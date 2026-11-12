import os
import torch
import torch.nn as nn
from pathlib import Path
from timm.models.layers import trunc_normal_
from modeling.clip.make_model_clipreid import load_clip_to_cpu
from modeling.clip.LoRA import mark_only_lora_as_trainable as lora_train
from modeling.dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BackboneRouter(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim):
        super(BackboneRouter, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0
        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                            num_classes=num_classes,
                                                            camera=self.camera_num, view=self.view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
            self.clip = 0
            self.base.load_param(model_path)
            print('Loading pretrained model from ImageNet')
            if cfg.MODEL.FROZEN:
                lora_train(self.base)

        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE
            clip_model = load_clip_to_cpu(cfg, self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                          cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                          cfg.MODEL.STRIDE_SIZE)
            print('Loading pretrained model from CLIP')
            self.base = clip_model.visual
            if cfg.MODEL.FROZEN:
                lora_train(self.base)

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))
        elif cfg.MODEL.TRANSFORMER_TYPE in {'dinov3_vitb16', 'dinov3_vitl16'}:
            self.clip = 2
            pst_kwargs = dict(
                pst_enabled=getattr(cfg.MODEL, "USE_PST", False),
                pst_seq_len=getattr(cfg.MODEL, "PST_SEQ_LEN", 8),
                pst_prompt_len=getattr(cfg.MODEL, "PST_PROMPT_LEN", getattr(cfg.MODEL, "PST_SEQ_LEN", 8)),
                pst_start_layer=getattr(cfg.MODEL, "PST_START_LAYER", 0),
                pst_prompt_groups=getattr(cfg.MODEL, "PST_PROMPT_GROUPS", 2),
            )
            if cfg.MODEL.TRANSFORMER_TYPE == 'dinov3_vitb16':
                self.base = dinov3_vitb16(pretrained=False, **pst_kwargs)
            else:
                self.base = dinov3_vitl16(pretrained=False, **pst_kwargs)

            ckpt_path = Path(model_path).expanduser()
            if ckpt_path.exists():
                print(f'Loading pretrained model from DINOv3 local checkpoint: {ckpt_path}')
                state = torch.load(str(ckpt_path), map_location='cpu')
                if isinstance(state, dict):
                    if "teacher" in state and isinstance(state["teacher"], dict):
                        state = state["teacher"]
                    elif "model" in state and isinstance(state["model"], dict):
                        state = state["model"]
                    elif "state_dict" in state and isinstance(state["state_dict"], dict):
                        state = state["state_dict"]

                cleaned_state = {}
                for k, v in state.items():
                    nk = k
                    for prefix in ("module.", "backbone.", "model.", "teacher.backbone.", "teacher."):
                        if nk.startswith(prefix):
                            nk = nk[len(prefix):]
                    cleaned_state[nk] = v
                incompatible = self.base.load_state_dict(cleaned_state, strict=False)
                print(f"DINOv3 missing keys: {len(incompatible.missing_keys)}, "
                      f"unexpected keys: {len(incompatible.unexpected_keys)}")
            elif os.environ.get('PATHS_ALLOW_RANDOM_INIT') == '1':
                print(f'Warning: DINOv3 checkpoint not found at {ckpt_path}, using random init (smoke test only).')
            else:
                raise FileNotFoundError(
                    f'DINOv3 checkpoint not found at {ckpt_path}. Set MODEL.PRETRAIN_PATH_T, '
                    'or export PATHS_ALLOW_RANDOM_INIT=1 for a smoke test.')

            if cfg.MODEL.FROZEN:
                for p in self.base.parameters():
                    p.requires_grad = False

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, label=None, cam_label=None, view_label=None, modality=None):
        # Video ReID: inputs are sequences of shape (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Expand cam_label / view_label to match B*T
        if cam_label is not None:
            cam_label = cam_label.unsqueeze(1).repeat(1, T).view(B * T)
        if view_label is not None:
            view_label = view_label.unsqueeze(1).repeat(1, T).view(B * T)
        if self.clip == 0:
            x = self.base(x, cam_label=cam_label, view_label=view_label)
            xproj = x
            x_sta = None
        elif self.clip == 1:
            if self.cv_embed_sign and cam_label is not None:
                cv_embed = (self.sie_xishu * self.cv_embed[cam_label]).squeeze(1)
            else:
                cv_embed = None
            # Features returned by the backbone, including PST features
            ret = self.base(x, cv_embed, modality)
            if len(ret) == 3:
                x, xproj, x_sta = ret
            else:
                x, xproj = ret
                x_sta = None
        else:
            # DINOv3 returns dict with normalized cls/patch tokens.
            out = self.base.forward_features(x)
            cls_tok = out["x_norm_clstoken"].unsqueeze(1)
            patch_tok = out["x_norm_patchtokens"]
            x = torch.cat([cls_tok, patch_tok], dim=1)
            xproj = x
            x_sta = out.get("x_pst", None)


        global_feat = x[:, 0]  # (B*T, 768)
        # Match the CLIP feature dimension
        global_feat_proj = xproj[:, 0]  # (B*T, 512)

        patch_tokens = x[:, 1:]  # (B*T, 196, 768)

        return patch_tokens, global_feat, x_sta

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
