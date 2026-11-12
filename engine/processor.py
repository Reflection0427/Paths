import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
import scipy.io
from layers.softmax_loss import CrossEntropyLabelSmooth
from modeling.modules.mab import MABDualMemory
import collections


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    centers = torch.stack(centers, dim=0)
    return centers




def _get_cfg_float(cfg, dotted_key, default):
    """Safely read a float config value from a yacs config using a dotted key."""
    node = cfg
    try:
        for key in dotted_key.split('.'):
            node = getattr(node, key)
        return float(node)
    except Exception:
        return float(default)


def cross_modal_supcon_loss(feat_rgb, feat_evt, target, temp=0.07):
    """
    Bidirectional supervised contrastive loss for RGB-Event identity alignment.

    Args:
        feat_rgb: Tensor with shape (B, D), RGB-side video descriptors.
        feat_evt: Tensor with shape (B, D), Event-side video descriptors.
        target:   Tensor with shape (B,), identity labels.
        temp:     Temperature for similarity logits.

    This uses all samples with the same identity in the current mini-batch as
    positives. It directly optimizes the cross-modal distance structure, which is
    not fully covered by local token alignment or prototype memory.
    """
    feat_rgb = F.normalize(feat_rgb, dim=1)
    feat_evt = F.normalize(feat_evt, dim=1)

    logits = torch.matmul(feat_rgb, feat_evt.t()) / temp  # (B, B)
    pos_mask = target.view(-1, 1).eq(target.view(1, -1)).float()

    # RGB -> Event
    log_prob_r2e = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_r2e = -(pos_mask * log_prob_r2e).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1.0)

    # Event -> RGB
    logits_t = logits.t()
    log_prob_e2r = logits_t - torch.logsumexp(logits_t, dim=1, keepdim=True)
    loss_e2r = -(pos_mask * log_prob_e2r).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1.0)

    return 0.5 * (loss_r2e.mean() + loss_e2r.mean())


def _forward_eval_batch(model, img, camids, target_view, return_pattern=3, img_path=None):
    """
    Forward helper for both normal RRS test clips and dense clip evaluation.

    Normal input: RGB/EVENT are (B, T, C, H, W).
    Dense input:  RGB/EVENT are (B, N, T, C, H, W), where N is the number of clips
    sampled from one tracklet. We flatten B*N clips, run the model once, then average
    features over N clips to obtain one video-level descriptor per tracklet.
    """
    rgb = img['RGB']
    event = img['EVENT']

    if rgb.dim() == 6:
        B, N, T, C, H, W = rgb.shape
        rgb = rgb.reshape(B * N, T, C, H, W)
        event = event.reshape(B * N, T, C, H, W)

        camids_dense = camids.view(B, 1).expand(B, N).reshape(B * N)
        view_dense = target_view.view(B, 1).expand(B, N).reshape(B * N)

        feat = model(
            {'RGB': rgb, 'EVENT': event},
            cam_label=camids_dense,
            view_label=view_dense,
            return_pattern=return_pattern,
            img_path=img_path,
        )
        return feat.view(B, N, -1).mean(dim=1)

    return model(
        {'RGB': rgb, 'EVENT': event},
        cam_label=camids,
        view_label=target_view,
        return_pattern=return_pattern,
        img_path=img_path,
    )

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_normal,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, num_classes):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("Paths.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    model.to(local_rank)


    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_frame = AverageMeter()
    loss_meter_frame = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent_frame = CrossEntropyLabelSmooth(num_classes=num_classes)

    use_mab = cfg.MODEL.USE_MAB
    frame_loss_weight = _get_cfg_float(cfg, 'MODEL.FRAME_LOSS_WEIGHT', 0.5)
    mab_loss_weight = _get_cfg_float(cfg, 'MODEL.MAB_LOSS_WEIGHT', 1.0)
    hmf_align_loss_weight = _get_cfg_float(cfg, 'MODEL.HMF_ALIGN_LOSS_WEIGHT', 1.0)
    cm_supcon_loss_weight = _get_cfg_float(cfg, 'MODEL.CM_SUPCON_LOSS_WEIGHT', 0.0)

    if use_mab:
        feat_dim = model.module.feat_dim if hasattr(model, "module") else model.feat_dim
        mab_memory = MABDualMemory(feat_dim, num_classes, temp=0.05,
                                   momentum=_get_cfg_float(cfg, 'MODEL.MAB_MOMENTUM', 0.1), alpha=0.1).to(device)
        logger.info("Generating initial MAB prototypes for RGB and Event...")
        model.eval()

        rgb_features_list = []
        event_features_list = []
        labels_list = []
        total_batches = len(train_loader_normal)
        log_interval = max(1, total_batches // 10)

        with torch.no_grad():
            for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader_normal):
                img_rgb = img['RGB'].to(device, non_blocking=True)  # (B, T, C, H, W)
                if len(img_rgb.size()) == 6:
                    # method = 'dense'
                    b, n, s, c, h, w = img_rgb.size()
                    assert (b == 1)
                    img_rgb = img_rgb.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                img_event = img['EVENT'].to(device, non_blocking=True)
                if len(img_event.size()) == 6:
                    # method = 'dense'
                    b, n, s, c, h, w = img_event.size()
                    assert (b == 1)
                    img_event = img_event.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                target = vid.to(device, non_blocking=True)
                target_cam = target_cam.to(device, non_blocking=True)
                target_view = target_view.to(device, non_blocking=True)

                # Extract features only (no classification scores) to build identity prototypes
                _, feats, _ = model({'RGB': img_rgb, 'EVENT': img_event},
                                    label=target, extract_features_only=True)

                # feats = [RGB_feat]
                rgb_feat = feats[0]  # (B, feat_dim)
                event_feat = feats[1]  # (B, feat_dim)
                rgb_feat = rgb_feat.view(-1, rgb_feat.size(1))
                rgb_feat = torch.mean(rgb_feat, 0, keepdim=True)  # 1,512
                event_feat = event_feat.view(-1, event_feat.size(1))
                event_feat = torch.mean(event_feat, 0, keepdim=True)  # 1,512
                # Collect features and labels
                for i, (rgb_f, event_f, label) in enumerate(zip(rgb_feat, event_feat, target)):
                    rgb_features_list.append(rgb_f.cpu())
                    event_features_list.append(event_f.cpu())
                    labels_list.append(label.cpu())

                if (n_iter + 1) % log_interval == 0 or (n_iter + 1) == total_batches:
                    progress = (n_iter + 1) / total_batches
                    logger.info(f"=> [MAB Init] feature extraction {n_iter + 1}/{total_batches} ({progress:.1%})")

        # Convert to tensors
        labels_tensor = torch.stack(labels_list, dim=0).cuda()  # (N,)
        rgb_features_tensor = torch.stack(rgb_features_list, dim=0).cuda()  # (N, D)
        event_features_tensor = torch.stack(event_features_list, dim=0).cuda()  # (N, D)

        # Compute identity prototypes
        logger.info(f"=> [MAB Init] Extracted features for {labels_tensor.size(0)} samples, computing identity prototypes...")
        cluster_features_rgb = generate_cluster_features(labels_tensor.cpu().numpy(), rgb_features_tensor)
        cluster_features_event = generate_cluster_features(labels_tensor.cpu().numpy(), event_features_tensor)

        # Initialize the MAB memory bank
        mab_memory.features_rgb = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
        mab_memory.features_event = F.normalize(cluster_features_event.repeat(2, 1), dim=1).cuda()

        logger.info("=> Initial MAB prototypes generated: RGB {} classes, Event {} classes".format(
            cluster_features_rgb.size(0), cluster_features_event.size(0)))
        logger.info("=> MAB memory bank will be updated dynamically during training via momentum")

        # Free GPU memory used by temporary tensors
        del labels_tensor, rgb_features_tensor, event_features_tensor, cluster_features_rgb, cluster_features_event
        torch.cuda.empty_cache()

    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    best_epoch = 0
    patience = int(_get_cfg_float(cfg, 'SOLVER.EARLY_STOP_PATIENCE', 0))
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img_rgb = img['RGB'].to(device, non_blocking=True)  # (B, T, C, H, W)
            img_event = img['EVENT'].to(device, non_blocking=True)
            img = {'RGB': img_rgb, 'EVENT': img_event}
            target = vid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)
            target_view = target_view.to(device, non_blocking=True)

            with amp.autocast(enabled=True):
                scores, feats, frame_score, align_loss = model(img, label=target, cam_label=target_cam,
                                                               view_label=target_view)
                B, T = img['RGB'].shape[0], img['RGB'].shape[1]

                # sequence-level loss (supports list inputs)
                loss_seq = loss_fn(score=scores, feat=feats, target=target, target_cam=target_cam)

                # frame-level loss (separate classifiers: [frame_score_r, frame_score_e])
                targetX = target.unsqueeze(1).expand(B, T).contiguous().view(B * T)
                loss_frame_r = xent_frame(frame_score[0].view(B * T, -1), targetX)
                loss_frame_e = xent_frame(frame_score[1].view(B * T, -1), targetX)
                loss_frame = (loss_frame_r + loss_frame_e) / 2.0

                loss_cm = 0
                if use_mab:
                    rgb_feat = feats[0]  # (B, feat_dim)
                    event_feat = feats[1]  # (B, feat_dim)

                    loss_cm = mab_memory(rgb_feat, event_feat, target)

                # Cross-modal supervised contrastive loss on HMF global features.
                # feats[2] = RGB_hmf_feat, feats[3] = EVENT_hmf_feat.
                if cm_supcon_loss_weight > 0:
                    loss_cm_sup = cross_modal_supcon_loss(feats[2], feats[3], target, temp=0.07)
                else:
                    loss_cm_sup = align_loss.new_zeros(())

                loss = (
                    loss_seq
                    + frame_loss_weight * loss_frame
                    + mab_loss_weight * loss_cm
                    + hmf_align_loss_weight * align_loss
                    + cm_supcon_loss_weight * loss_cm_sup
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (scores[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)
            frame_loss_avg = loss_frame.item()

            # Separate classifiers: average frame accuracy over the two modalities
            frame_acc_r = (frame_score[0].detach().view(B * T, -1).max(1)[1] == targetX).float().mean()
            frame_acc_e = (frame_score[1].detach().view(B * T, -1).max(1)[1] == targetX).float().mean()
            frame_acc = (frame_acc_r + frame_acc_e) / 2.0

            acc_meter_frame.update(frame_acc, 1)
            loss_meter_frame.update(frame_loss_avg, img['RGB'].shape[0])

            if (n_iter + 1) % log_period == 0:
                Acc_RGB = (scores[0].max(1)[1] == target).float().mean().item()
                Acc_Event = (scores[1].max(1)[1] == target).float().mean().item()

                log_str = f"Epoch{epoch:02d} ˶ᵒ ᜊ ˂˶…  [{n_iter + 1:04d}/{len(train_loader):04d}] | "
                log_str += f"LR: {scheduler._get_lr(epoch)[0]:.2e} | "
                log_str += f"Loss: {loss_meter.avg:.4f} | "
                log_str += f"Acc: RGB {Acc_RGB:.2%}, Event {Acc_Event:.2%} | "
                log_str += f"Frame: RGB {loss_frame_r.item() :.4f}, Event {loss_frame_e.item() / T:.4f}"

                if use_mab:
                    log_str += f" | MAB: {loss_cm.item():.4f}"
                if cm_supcon_loss_weight > 0:
                    log_str += f" | CM-SupCon: {loss_cm_sup.item():.4f}"
                log_str += f" | Align: {align_loss.item():.4f}"

                logger.info(log_str)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=3)
        if mAP >= best_index['mAP']:
            best_index['mAP'] = mAP
            best_index['Rank-1'] = cmc[0]
            best_index['Rank-5'] = cmc[4]
            best_index['Rank-10'] = cmc[9]
            best_epoch = epoch

            # Save the best model
            best_model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model (mAP: {mAP:.1%}) to {best_model_path}")

        logger.info("~" * 50)
        logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
        logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
        logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
        logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
        logger.info("~" * 50)

        if patience > 0 and epoch - best_epoch >= patience:
            logger.info(f"Early stopping at epoch {epoch}: mAP has not improved for {patience} epochs "
                        f"(best {best_index['mAP']:.1%} at epoch {best_epoch}).")
            break


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=3):
    evaluator.reset()
    model.eval()
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device), 'EVENT': img['EVENT'].to(device)}
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = _forward_eval_batch(model, img, camids, target_view, return_pattern=return_pattern)
            evaluator.update((feat, vid, camid, _))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))

    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 return_pattern=3,
                 vis_config=None):
    device = "cuda"
    logger = logging.getLogger("Paths.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    model.to(device)

    model.eval()
    img_path_list = []
    logger.info("~" * 50)
    logger.info("~" * 50)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            # print(imgpath)
            img = {'RGB': img['RGB'].to(device),
                   'EVENT': img['EVENT'].to(device)}
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = _forward_eval_batch(
                model, img, camids, target_view, return_pattern=return_pattern, img_path=imgpath
            )
            evaluator.update((feat, pid, camid, imgpath, target_view.cpu().numpy()))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    if vis_config and vis_config.get("enabled", False):
        from visualize.test_visualizations import run_test_visualizations

        logger.info("Generating test visualizations...")
        run_test_visualizations(
            cfg=cfg,
            model=model,
            evaluator=evaluator,
            distmat=distmat,
            qf=qf,
            gf=gf,
            vis_config=vis_config,
            device=device,
        )
        logger.info("Test visualizations saved to: %s", os.path.join(cfg.OUTPUT_DIR, "visualizations"))

    # Save distance matrix
    dist_save_dir = os.path.join(cfg.OUTPUT_DIR, 'distmat')
    os.makedirs(dist_save_dir, exist_ok=True)
    # Save as .mat file (for visualize_reid.py)
    dist_mat_path = os.path.join(dist_save_dir, f'dist.mat')
    scipy.io.savemat(dist_mat_path, {'distmat': distmat})
    logger.info("Distance matrix saved to: {}".format(dist_mat_path))
    return cmc[0], cmc[4]
