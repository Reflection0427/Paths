import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbone_router import BackboneRouter, weights_init_classifier, weights_init_kaiming
from modeling.modules.hmf import HMFGlobalConsensusPooling, HMFGlobalGraphFusion, HMFLocalTokenFusionHead
import torch


class Paths(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(Paths, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'dinov3_vitb16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'dinov3_vitl16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 1024
        else:
            raise ValueError(f"Unsupported TRANSFORMER_TYPE: {cfg.MODEL.TRANSFORMER_TYPE}")
        # Two separate backbones: RGB and EVENT do not share parameters
        self.BACKBONE_RGB = BackboneRouter(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.BACKBONE_EVENT = BackboneRouter(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.neck = cfg.MODEL.NECK
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.head = cfg.MODEL.HEAD
        self.use_hmf = cfg.MODEL.USE_HMF
        if self.use_hmf:
            self.hmf_global_rgb = HMFGlobalGraphFusion(
                dim=self.feat_dim, num_heads=8, k_intra=3, k_cross=3, temperature=0.05
            )
            self.hmf_global_event = HMFGlobalGraphFusion(
                dim=self.feat_dim, num_heads=8, k_intra=3, k_cross=3, temperature=0.05
            )
            self.hmf_global_pool = HMFGlobalConsensusPooling(dim=self.feat_dim)
            self.hmf_local_fusion = HMFLocalTokenFusionHead(in_dim=self.feat_dim)
        else:
            self.hmf_global_rgb = None
            self.hmf_global_event = None
            self.hmf_global_pool = None
            self.hmf_local_fusion = None

        # ==================== Decoupled bottlenecks / classifiers ====================
        # Spatial branch
        self.bottleneck_spatial_r = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_spatial_e = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_spatial_r.bias.requires_grad_(False)
        self.bottleneck_spatial_e.bias.requires_grad_(False)
        self.bottleneck_spatial_r.apply(weights_init_kaiming)
        self.bottleneck_spatial_e.apply(weights_init_kaiming)

        # HMF global branch
        self.bottleneck_hmf_global_r = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_hmf_global_e = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_hmf_global_r.bias.requires_grad_(False)
        self.bottleneck_hmf_global_e.bias.requires_grad_(False)
        self.bottleneck_hmf_global_r.apply(weights_init_kaiming)
        self.bottleneck_hmf_global_e.apply(weights_init_kaiming)

        # PST branch
        self.bottleneck_pst_r = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_pst_e = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_pst_r.bias.requires_grad_(False)
        self.bottleneck_pst_e.bias.requires_grad_(False)
        self.bottleneck_pst_r.apply(weights_init_kaiming)
        self.bottleneck_pst_e.apply(weights_init_kaiming)

        # HMF local branch
        self.bottleneck_hmf_local = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_hmf_local.bias.requires_grad_(False)
        self.bottleneck_hmf_local.apply(weights_init_kaiming)

        # Separate classifiers for spatial / HMF / PST features to avoid distribution conflicts
        self.classifier_spatial = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_hmf_global = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_spatial.apply(weights_init_classifier)
        self.classifier_hmf_global.apply(weights_init_classifier)

        self.classifier_pst = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_pst.apply(weights_init_classifier)

        self.classifier_hmf_local = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_hmf_local.apply(weights_init_classifier)

        # ==================== Final Fusion Branch ====================
        # Directly supervise the final retrieval descriptor instead of only
        # supervising each independent branch. This reduces descriptor mismatch
        # between training branches and the concatenated feature used at test time.
        self.fusion_branch_num = 7  # rgb_s, evt_s, rgb_hmf, evt_hmf, rgb_pst, evt_pst, local
        self.fusion_reduce = nn.Linear(self.feat_dim * self.fusion_branch_num, self.feat_dim, bias=False)
        nn.init.kaiming_normal_(self.fusion_reduce.weight, a=0, mode='fan_out')

        self.bottleneck_fusion = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_fusion.bias.requires_grad_(False)
        self.bottleneck_fusion.apply(weights_init_kaiming)

        self.classifier_fusion = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_fusion.apply(weights_init_classifier)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"=> Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None,
                extract_features_only=False):
        RGB = x['RGB']  # (B, T, C, H, W)
        EVENT = x['EVENT']  # (B, T, C, H, W)
        B, T, C, H, W = RGB.shape

        RGB_patch_tokens, RGB_feat_bt, RGB_pst = self.BACKBONE_RGB(RGB, cam_label=cam_label, view_label=view_label)
        EVENT_patch_tokens, EVENT_feat_bt, EVENT_pst = self.BACKBONE_EVENT(
            EVENT, cam_label=cam_label, view_label=view_label
        )

        if extract_features_only:
            return None, [RGB_feat_bt, EVENT_feat_bt], None

        # Some backbones do not produce PST features when USE_PST is disabled.
        if RGB_pst is None:
            RGB_pst = RGB_feat_bt
        if EVENT_pst is None:
            EVENT_pst = EVENT_feat_bt

        RGB_pst_feat = RGB_pst.view(B, T, -1).mean(dim=1)
        EVENT_pst_feat = EVENT_pst.view(B, T, -1).mean(dim=1)

        # Baseline sequence aggregation for each modality.
        RGB_feat_seq_base = RGB_feat_bt.view(B, T, -1).mean(dim=1)
        EVENT_feat_seq_base = EVENT_feat_bt.view(B, T, -1).mean(dim=1)

        if self.use_hmf:
            rgb_patch_tap = RGB_patch_tokens.view(B, T, -1, self.feat_dim).mean(dim=1)
            event_patch_tap = EVENT_patch_tokens.view(B, T, -1, self.feat_dim).mean(dim=1)

            if not self.training:
                feat_hmf_local, align_loss, pre_rgb_tokens, pre_evt_tokens, post_fused_tokens = self.hmf_local_fusion(
                    rgb_patch_tap, event_patch_tap, return_tokens=True
                )
                self.pre_fusion_rgb_tokens = pre_rgb_tokens
                self.pre_fusion_event_tokens = pre_evt_tokens
                self.post_fusion_tokens = post_fused_tokens
            else:
                feat_hmf_local, align_loss = self.hmf_local_fusion(
                    rgb_patch_tap, event_patch_tap
                )

            rgb_frame_feat = RGB_feat_bt.view(B, T, -1)
            evt_frame_feat = EVENT_feat_bt.view(B, T, -1)
            rgb_hmf_global_seq = self.hmf_global_rgb(
                x_query=rgb_frame_feat, x_intra_pool=rgb_frame_feat, x_cross_pool=evt_frame_feat
            )
            event_hmf_global_seq = self.hmf_global_event(
                x_query=evt_frame_feat, x_intra_pool=evt_frame_feat, x_cross_pool=rgb_frame_feat
            )
            RGB_hmf_feat, EVENT_hmf_feat = self.hmf_global_pool(rgb_hmf_global_seq, event_hmf_global_seq)
        else:
            feat_hmf_local = 0.5 * (RGB_feat_seq_base + EVENT_feat_seq_base)
            align_loss = RGB_feat_bt.new_zeros(())
            RGB_hmf_feat = RGB_feat_seq_base
            EVENT_hmf_feat = EVENT_feat_seq_base
            if not self.training:
                self.pre_fusion_rgb_tokens = None
                self.pre_fusion_event_tokens = None
                self.post_fusion_tokens = None

        if self.training:
            RGB_spatial_bn = self.bottleneck_spatial_r(RGB_feat_seq_base)
            EVENT_spatial_bn = self.bottleneck_spatial_e(EVENT_feat_seq_base)
            RGB_score_spatial = self.classifier_spatial(RGB_spatial_bn)
            EVENT_score_spatial = self.classifier_spatial(EVENT_spatial_bn)

            RGB_hmf_global_bn = self.bottleneck_hmf_global_r(RGB_hmf_feat)
            EVENT_hmf_global_bn = self.bottleneck_hmf_global_e(EVENT_hmf_feat)
            RGB_score_hmf_global = self.classifier_hmf_global(RGB_hmf_global_bn)
            EVENT_score_hmf_global = self.classifier_hmf_global(EVENT_hmf_global_bn)

            RGB_pst_bn = self.bottleneck_pst_r(RGB_pst_feat)
            EVENT_pst_bn = self.bottleneck_pst_e(EVENT_pst_feat)
            RGB_score_pst = self.classifier_pst(RGB_pst_bn)
            EVENT_score_pst = self.classifier_pst(EVENT_pst_bn)

            feat_hmf_local_bn = self.bottleneck_hmf_local(feat_hmf_local)
            score_hmf_local = self.classifier_hmf_local(feat_hmf_local_bn)

            fusion_raw = torch.cat([
                RGB_feat_seq_base, EVENT_feat_seq_base,
                RGB_hmf_feat, EVENT_hmf_feat,
                RGB_pst_feat, EVENT_pst_feat,
                feat_hmf_local,
            ], dim=-1)
            fusion_feat = self.fusion_reduce(fusion_raw)
            fusion_bn = self.bottleneck_fusion(fusion_feat)
            score_fusion = self.classifier_fusion(fusion_bn)

            RGB_frame_score = self.classifier_spatial(self.bottleneck_spatial_r(RGB_feat_bt)).view(B, T, -1)
            EVENT_frame_score = self.classifier_spatial(self.bottleneck_spatial_e(EVENT_feat_bt)).view(B, T, -1)

            scores = [RGB_score_spatial, EVENT_score_spatial,
                      RGB_score_hmf_global, EVENT_score_hmf_global,
                      RGB_score_pst, EVENT_score_pst,
                      score_hmf_local, score_fusion]

            feats = [RGB_feat_seq_base, EVENT_feat_seq_base,
                     RGB_hmf_feat, EVENT_hmf_feat,
                     RGB_pst_feat, EVENT_pst_feat,
                     feat_hmf_local, fusion_feat]

            frame_scores = [RGB_frame_score, EVENT_frame_score]

            return scores, feats, frame_scores, align_loss

        else:
            rgb_s = self.bottleneck_spatial_r(RGB_feat_seq_base)
            evt_s = self.bottleneck_spatial_e(EVENT_feat_seq_base)
            rgb_hmf = self.bottleneck_hmf_global_r(RGB_hmf_feat)
            evt_hmf = self.bottleneck_hmf_global_e(EVENT_hmf_feat)
            rgb_pst = self.bottleneck_pst_r(RGB_pst_feat)
            evt_pst = self.bottleneck_pst_e(EVENT_pst_feat)
            local = self.bottleneck_hmf_local(feat_hmf_local)

            fusion_raw = torch.cat([
                RGB_feat_seq_base, EVENT_feat_seq_base,
                RGB_hmf_feat, EVENT_hmf_feat,
                RGB_pst_feat, EVENT_pst_feat,
                feat_hmf_local,
            ], dim=-1)
            fusion_feat = self.fusion_reduce(fusion_raw)
            fusion_bn = self.bottleneck_fusion(fusion_feat)

            if return_pattern == 1:
                return torch.cat([rgb_s, rgb_hmf], dim=-1)
            elif return_pattern == 3:
                # Fusion descriptor + strong branch descriptors.
                # This usually performs better than raw branch concatenation because
                # fusion_bn is explicitly optimized by ID/Triplet losses during training.
                return torch.cat([
                    fusion_bn,
                    rgb_hmf, evt_hmf,
                    rgb_pst, evt_pst,
                    local,
                    ], dim=-1)
            elif return_pattern == 4:
                # Only the compact supervised fusion descriptor. Useful for ablation.
                return fusion_bn
            else:
                return torch.cat([
                    rgb_s, evt_s,
                    rgb_hmf, evt_hmf,
                    rgb_pst, evt_pst,
                    local,
                    ], dim=-1)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = Paths(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building Paths===========')
    return model
