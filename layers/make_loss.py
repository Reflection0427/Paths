# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False)  # center loss



    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)



    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':





        def loss_func(score, feat, target, target_cam, isprint=False):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    id_loss_fn = xent
                else:
                    id_loss_fn = F.cross_entropy



                # ID loss (separate classifiers, multiple branches supported)
                if isinstance(score, list):
                    loss_list = [id_loss_fn(s, target) for s in score]
                    ID_LOSS = sum(loss_list) / len(loss_list)
                else:
                    raise ValueError("Expected score to be a list for separate classification")






                # Triplet loss (separate classifiers, multiple branches supported)
                if isinstance(feat, list):
                    tri_list = [triplet(f, target)[0] for f in feat]
                    TRI_LOSS = sum(tri_list) / len(tri_list)
                else:
                    raise ValueError("Expected feat to be a list for separate classification")

                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                return loss








            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
