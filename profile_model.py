import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from config import cfg
from data import make_dataloader
from modeling import make_model
from utils.flops import give_supported_ops


MB = 1024 ** 2


class FullProfileWrapper(nn.Module):
    def __init__(self, model, return_pattern):
        super().__init__()
        self.model = model
        self.return_pattern = return_pattern

    def forward(self, rgb, event, camids, viewids):
        inputs = {"RGB": rgb, "EVENT": event}
        return self.model(
            inputs,
            cam_label=camids,
            view_label=viewids,
            return_pattern=self.return_pattern,
        )


class BaselineProfileWrapper(nn.Module):
    """Baseline inference path without the extra fusion/attention modules."""

    def __init__(self, model):
        super().__init__()
        self.backbone_rgb = model.BACKBONE_RGB
        self.backbone_event = model.BACKBONE_EVENT
        self.bottleneck_spatial_r = model.bottleneck_spatial_r
        self.bottleneck_spatial_e = model.bottleneck_spatial_e

    def forward(self, rgb, event, camids, viewids):
        batch_size, seq_len = rgb.shape[:2]
        _, rgb_feat_bt, _ = self.backbone_rgb(rgb, cam_label=camids, view_label=viewids)
        _, event_feat_bt, _ = self.backbone_event(event, cam_label=camids, view_label=viewids)

        rgb_feat = rgb_feat_bt.view(batch_size, seq_len, -1).mean(dim=1)
        event_feat = event_feat_bt.view(batch_size, seq_len, -1).mean(dim=1)
        rgb_feat = self.bottleneck_spatial_r(rgb_feat)
        event_feat = self.bottleneck_spatial_e(event_feat)
        return torch.cat([rgb_feat, event_feat], dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MUA model complexity and forward latency")
    parser.add_argument("--config_file", default="", type=str, help="Path to config file")
    parser.add_argument("--weight", default="", type=str, help="Path to trained model weight")
    parser.add_argument("--batch-size", default=1, type=int, help="Dummy input batch size")
    parser.add_argument("--warmup", default=20, type=int, help="Warmup forward iterations")
    parser.add_argument("--iters", default=100, type=int, help="Timed forward iterations")
    parser.add_argument("--return-pattern", default=3, type=int, help="MUA inference return pattern")
    parser.add_argument(
        "--profile-mode",
        default="full",
        choices=("full", "baseline"),
        help="full runs MUA.forward; baseline skips MFA/hypergraph/consensus modules",
    )
    parser.add_argument(
        "--output-json",
        default="",
        type=str,
        help=(
            "Optional metrics JSON path. Defaults to cfg.OUTPUT_DIR/profile_metrics.json "
            "or profile_metrics_baseline.json."
        ),
    )
    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="Optional suffix for the default JSON filename, e.g. baseline, cm, cm_tac.",
    )
    parser.add_argument("--num-classes", default=None, type=int, help="Override class count and skip dataloader lookup")
    parser.add_argument("--camera-num", default=None, type=int, help="Override camera count and skip dataloader lookup")
    parser.add_argument("--view-num", default=None, type=int, help="Override view count and skip dataloader lookup")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Optional config overrides, e.g. MODEL.DEVICE_ID 0",
    )
    return parser.parse_args()


def count_params_and_buffers(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "trainable_params": trainable_params,
        "trainable_params_m": trainable_params / 1e6,
        "param_memory_mb": param_memory / MB,
        "buffer_memory_mb": buffer_memory / MB,
    }


def load_model_weight(model, weight_path):
    if not weight_path:
        print("[Profile] No full model weight provided; profiling current initialized model.")
        return None

    path = Path(weight_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    cleaned_state = {}
    for key, value in state.items():
        cleaned_key = key[len("module."):] if key.startswith("module.") else key
        cleaned_state[cleaned_key] = value

    incompatible = model.load_state_dict(cleaned_state, strict=False)
    print(f"[Profile] Loaded weight: {path}")
    print(
        "[Profile] load_state_dict strict=False, "
        f"missing_keys={len(incompatible.missing_keys)}, "
        f"unexpected_keys={len(incompatible.unexpected_keys)}"
    )
    return str(path)


def resolve_model_metadata(args):
    manual_values = (args.num_classes, args.camera_num, args.view_num)
    if all(value is not None for value in manual_values):
        return args.num_classes, args.camera_num, args.view_num

    print("[Profile] Inferring num_classes/camera_num/view_num from dataloader.")
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    return (
        args.num_classes if args.num_classes is not None else num_classes,
        args.camera_num if args.camera_num is not None else camera_num,
        args.view_num if args.view_num is not None else view_num,
    )


def build_dummy_inputs(batch_size, camera_num, view_num, device):
    seq_len = int(cfg.INPUT.SEQ_LEN)
    height, width = map(int, cfg.INPUT.SIZE_TEST)
    shape = (batch_size, seq_len, 3, height, width)
    rgb = torch.randn(shape, device=device)
    event = torch.randn(shape, device=device)

    safe_camera_num = max(int(camera_num), 1)
    safe_view_num = max(int(view_num), 1)
    camids = torch.arange(batch_size, device=device, dtype=torch.long) % safe_camera_num
    viewids = torch.arange(batch_size, device=device, dtype=torch.long) % safe_view_num
    return rgb, event, camids, viewids


def profile_flops(profiled_model, inputs, profile_mode):
    result = {
        "note": "FLOPs are torch-traced GFLOPs for model forward; CPU/SciPy Hungarian matching in MFA is not manually added.",
        "profile_mode": profile_mode,
    }
    if profile_mode == "baseline":
        result["note"] = (
            "FLOPs are torch-traced GFLOPs for baseline forward: RGB/EVENT backbones, "
            "temporal average pooling, spatial BN, and feature concatenation only."
        )
    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(profiled_model, inputs)
            supported_ops_args = []
            for op_name, handle in give_supported_ops().items():
                supported_ops_args.extend([op_name, handle])
            analysis.set_op_handle(*supported_ops_args)
            analysis.tracer_warnings("no_tracer_warning")
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)

            total_flops = analysis.total()
            result.update(
                {
                    "flops": total_flops,
                    "gflops": total_flops / 1e9,
                    "unsupported_ops": dict(analysis.unsupported_ops()),
                    "by_operator": dict(analysis.by_operator()),
                    "uncalled_modules_count": len(analysis.uncalled_modules()),
                }
            )
    except Exception as exc:
        result.update(
            {
                "flops": None,
                "gflops": None,
                "unsupported_ops": {},
                "error": repr(exc),
            }
        )
    return result


def percentile(values, q):
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, math.ceil((q / 100.0) * len(sorted_values)) - 1)
    return sorted_values[index]


def profile_latency_and_memory(profiled_model, inputs, batch_size, warmup, iters):
    rgb, event, camids, viewids = inputs

    with torch.inference_mode():
        for _ in range(warmup):
            profiled_model(rgb, event, camids, viewids)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        timings_ms = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            profiled_model(rgb, event, camids, viewids)
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(start.elapsed_time(end))

    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    p95_ms = percentile(timings_ms, 95)
    fps = batch_size * 1000.0 / mean_ms if mean_ms > 0 else None

    return {
        "latency": {
            "warmup_iters": warmup,
            "timed_iters": iters,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "fps": fps,
        },
        "memory": {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / MB,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / MB,
            "current_allocated_mb": torch.cuda.memory_allocated() / MB,
            "current_reserved_mb": torch.cuda.memory_reserved() / MB,
        },
    }


def print_summary(metrics):
    params = metrics["params"]
    flops = metrics["flops"]
    latency = metrics["latency"]
    memory = metrics["memory"]

    print("\n========== Profile Summary ==========")
    print(f"Profile mode     : {metrics['profile_mode']}")
    print(f"USE_MAB          : {metrics['config_flags']['USE_MAB']}")
    print(f"USE_PST          : {metrics['config_flags']['USE_PST']}")
    print(f"USE_HMF          : {metrics['config_flags']['USE_HMF']}")
    print(f"Input shape      : RGB/EVENT {metrics['input']['shape']}")
    print(f"Return pattern   : {metrics['input']['return_pattern']}")
    print(f"Param scope      : {metrics['params']['scope']}")
    print(f"Total params     : {params['total_params']:,} ({params['total_params_m']:.3f} M)")
    print(f"Trainable params : {params['trainable_params']:,} ({params['trainable_params_m']:.3f} M)")
    print(f"Param memory     : {params['param_memory_mb']:.2f} MB")
    print(f"Buffer memory    : {params['buffer_memory_mb']:.2f} MB")
    if flops.get("gflops") is None:
        print(f"GFLOPs           : unavailable ({flops.get('error', 'unknown error')})")
    else:
        print(f"GFLOPs           : {flops['gflops']:.4f}")
        print(f"Unsupported ops  : {len(flops.get('unsupported_ops', {}))}")
    print(f"Mean latency     : {latency['mean_ms']:.3f} ms")
    print(f"Median latency   : {latency['median_ms']:.3f} ms")
    print(f"P95 latency      : {latency['p95_ms']:.3f} ms")
    print(f"Throughput       : {latency['fps']:.2f} samples/s")
    print(f"Peak allocated   : {memory['peak_allocated_mb']:.2f} MB")
    print(f"Peak reserved    : {memory['peak_reserved_mb']:.2f} MB")
    if metrics["config_flags"]["USE_MAB"]:
        print("Note             : the MAB memory bank is used only during training and is not counted here.")
    print("=====================================\n")


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be greater than or equal to 0")
    if args.iters <= 0:
        raise ValueError("--iters must be greater than 0")

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script because latency uses torch.cuda.Event.")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes, camera_num, view_num = resolve_model_metadata(args)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    loaded_weight = load_model_weight(model, args.weight or cfg.TEST.WEIGHT)
    model.to(device)
    model.eval()

    if args.profile_mode == "baseline":
        profiled_model = BaselineProfileWrapper(model).to(device).eval()
        param_scope = "active baseline modules only"
    else:
        profiled_model = FullProfileWrapper(model, args.return_pattern).to(device).eval()
        param_scope = "full MUA model"
    inputs = build_dummy_inputs(args.batch_size, camera_num, view_num, device)

    params = count_params_and_buffers(profiled_model)
    params["scope"] = param_scope
    flops = profile_flops(profiled_model, inputs, args.profile_mode)
    timed = profile_latency_and_memory(profiled_model, inputs, args.batch_size, args.warmup, args.iters)

    seq_len = int(cfg.INPUT.SEQ_LEN)
    height, width = map(int, cfg.INPUT.SIZE_TEST)
    metrics = {
        "profile_mode": args.profile_mode,
        "config_file": args.config_file,
        "weight": loaded_weight,
        "device": torch.cuda.get_device_name(device),
        "metadata": {
            "num_classes": num_classes,
            "camera_num": camera_num,
            "view_num": view_num,
        },
        "config_flags": {
            "USE_MAB": bool(getattr(cfg.MODEL, "USE_MAB", False)),
            "USE_PST": bool(getattr(cfg.MODEL, "USE_PST", False)),
            "USE_HMF": bool(getattr(cfg.MODEL, "USE_HMF", False)),
            "PST_SEQ_LEN": int(cfg.MODEL.PST_SEQ_LEN),
            "PST_START_LAYER": int(cfg.MODEL.PST_START_LAYER),
            "PST_PROMPT_LEN": int(cfg.MODEL.PST_PROMPT_LEN),
            "PST_PROMPT_GROUPS": int(cfg.MODEL.PST_PROMPT_GROUPS),
        },
        "input": {
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "height": height,
            "width": width,
            "shape": [args.batch_size, seq_len, 3, height, width],
            "return_pattern": args.return_pattern if args.profile_mode == "full" else None,
        },
        "params": params,
        "flops": flops,
        **timed,
    }

    if args.profile_mode == "full":
        default_json_name = "profile_metrics.json"
    else:
        default_json_name = "profile_metrics_baseline.json"
    if args.tag:
        stem, suffix = os.path.splitext(default_json_name)
        default_json_name = f"{stem}_{args.tag}{suffix}"
    json_path = Path(args.output_json) if args.output_json else output_dir / default_json_name
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_summary(metrics)
    print(f"[Profile] Metrics saved to: {json_path}")


if __name__ == "__main__":
    main()
