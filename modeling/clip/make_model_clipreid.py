import torch
from modeling.clip import clip


def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size):
    model_path = cfg.MODEL.CLIP_PATH

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(cfg, state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model
