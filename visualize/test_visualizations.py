import logging
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

from utils.metrics import visualize_pos_neg_similarity


LOGGER = logging.getLogger("Paths.visualize")


def _get_font(size: int):
    for name in ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf", "arial.ttf", "DejaVuSans.ttf", "simhei.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _load_image(path: str, size):
    try:
        return Image.open(path).convert("RGB").resize((size[0], size[1]), Image.BILINEAR)
    except Exception:
        return Image.new("RGB", size, (30, 30, 30))


def _pick_first_frames(paths: List[str], num_frames: int = 4) -> List[str]:
    if not paths:
        return []
    picked = list(paths[:num_frames])
    while len(picked) < num_frames:
        picked.append(picked[-1])
    return picked


def _tracklet_stack_image(paths: List[str], border_color, frame_size=(72, 144), offset=(10, 8)):
    picked = _pick_first_frames(paths, 4)
    if not picked:
        return Image.new("RGB", frame_size, (255, 255, 255))

    dx, dy = offset
    card_w = frame_size[0] + dx * 3 + 6
    card_h = frame_size[1] + dy * 3 + 6
    canvas = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(picked):
        frame = _load_image(path, frame_size)
        x = idx * dx + 3
        y = idx * dy + 3
        canvas.paste(frame, (x, y))
        draw.rectangle(
            [(x, y), (x + frame_size[0] - 1, y + frame_size[1] - 1)],
            outline=border_color,
            width=2,
        )

    return canvas


def _build_rank_column(title: str, tracklet: Dict[str, List[str]], border_color):
    rgb_stack = _tracklet_stack_image(tracklet["rgb_paths"], border_color)
    event_stack = _tracklet_stack_image(tracklet["event_paths"], border_color)
    width = max(rgb_stack.width, event_stack.width)
    title_h = 34
    gap = 12
    height = title_h + rgb_stack.height + gap + event_stack.height
    column = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(column)
    font = _get_font(18)
    draw.text((width // 2, 7), title, font=font, fill=(15, 15, 15), anchor="ma")
    column.paste(rgb_stack, ((width - rgb_stack.width) // 2, title_h))
    column.paste(event_stack, ((width - event_stack.width) // 2, title_h + rgb_stack.height + gap))
    return column


def _save_rank_list_for_query(query_idx, query_pid, query_camid, query_tracklet, gallery_infos, save_path):
    colors = {
        "query": (52, 152, 219),
        "match": (46, 204, 113),
        "mismatch": (231, 76, 60),
    }
    columns = [
        _build_rank_column(
            "Query",
            query_tracklet,
            colors["query"],
        )
    ]

    for rank_info in gallery_infos:
        color = colors["match"] if rank_info["is_match"] else colors["mismatch"]
        title = f"Rank-{rank_info['rank']}"
        columns.append(_build_rank_column(title, rank_info["tracklet"], color))

    label_w = 78
    gap = 16
    top_margin = 12
    rgb_stack_h = _tracklet_stack_image(query_tracklet["rgb_paths"], colors["query"]).height
    rgb_y = top_margin + 34
    event_y = rgb_y + rgb_stack_h + 12

    canvas_w = label_w + sum(col.width for col in columns) + gap * (len(columns) + 1)
    canvas_h = top_margin + max(col.height for col in columns) + 18
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _get_font(22)

    draw.text((18, rgb_y + 50), "RGB", fill=(0, 0, 0), font=label_font)
    draw.text((10, event_y + 50), "EVENT", fill=(0, 0, 0), font=label_font)

    current_x = label_w + gap
    current_y = top_margin
    for col in columns:
        canvas.paste(col, (current_x, current_y))
        current_x += col.width + gap

    canvas.save(save_path)


def _infer_hw(num_tokens: int):
    if num_tokens == 128:
        return 16, 8
    if num_tokens % 8 == 0:
        return num_tokens // 8, 8
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, side
    for width in range(int(math.sqrt(num_tokens)), 0, -1):
        if num_tokens % width == 0:
            return num_tokens // width, width
    return num_tokens, 1


def _preprocess_tracklet(tracklet, cfg, device):
    height, width = cfg.INPUT.SIZE_TRAIN
    mean = np.asarray(cfg.INPUT.PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(cfg.INPUT.PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)

    def _load_paths(paths):
        loaded = []
        raw = []
        for path in paths:
            img = _load_image(path, (width, height))
            raw.append(img)
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - mean) / std
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            loaded.append(tensor)
        return torch.stack(loaded, dim=0), raw

    rgb_tensor, rgb_raw = _load_paths(tracklet["rgb_paths"])
    event_tensor, event_raw = _load_paths(tracklet["event_paths"])
    return (
        rgb_tensor.unsqueeze(0).to(device),
        event_tensor.unsqueeze(0).to(device),
        rgb_raw,
        event_raw,
    )


def _tokens_to_cams(tokens):
    num_frames, num_tokens, channels = tokens.shape
    height, width = _infer_hw(num_tokens)
    maps = tokens.reshape(num_frames, height, width, channels).permute(0, 3, 1, 2).mean(dim=1)
    cams = []
    for cam in maps:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cams.append(cam.cpu().numpy())
    return cams


def _overlay_heatmap(image: Image.Image, cam_map: np.ndarray):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    if cam_map.shape != image_np.shape[:2]:
        cam_map = Image.fromarray(np.uint8(np.clip(cam_map, 0.0, 1.0) * 255.0), mode="L")
        cam_map = cam_map.resize((image.width, image.height), Image.BILINEAR)
        cam_map = np.asarray(cam_map).astype(np.float32) / 255.0
    heatmap = cm.jet(cam_map)[..., :3].astype(np.float32)
    overlay = np.clip(image_np * 0.45 + heatmap * 0.55, 0.0, 1.0)
    return Image.fromarray(np.uint8(overlay * 255))


def _save_cam_grid(tracklet, model, cfg, device, camid, viewid, title, save_path):
    rgb_tensor, event_tensor, rgb_raw, event_raw = _preprocess_tracklet(tracklet, cfg, device)
    cam_label = torch.tensor([camid], device=device)
    view_label = torch.tensor([viewid], device=device)

    with torch.no_grad():
        rgb_patch, _, _ = model.BACKBONE_RGB(rgb_tensor, cam_label=cam_label, view_label=view_label)
        event_patch, _, _ = model.BACKBONE_EVENT(event_tensor, cam_label=cam_label, view_label=view_label)

    seq_len = rgb_tensor.shape[1]
    rgb_patch = rgb_patch.view(1, seq_len, rgb_patch.shape[1], rgb_patch.shape[2])[0]
    event_patch = event_patch.view(1, seq_len, event_patch.shape[1], event_patch.shape[2])[0]

    rgb_cams = _tokens_to_cams(rgb_patch)[:4]
    event_cams = _tokens_to_cams(event_patch)[:4]
    rgb_frames = rgb_raw[:4]
    event_frames = event_raw[:4]

    frame_w, frame_h = rgb_frames[0].size
    inner_gap = 8
    pair_gap = 18
    row_gap = 18
    margin = 20
    label_w = 78
    pair_w = frame_w * 2 + inner_gap
    canvas_w = label_w + margin * 2 + pair_w * 4 + pair_gap * 3
    canvas_h = margin * 2 + frame_h * 2 + row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _get_font(22)
    y_rgb = margin
    y_event = y_rgb + frame_h + row_gap
    draw.text((18, y_rgb + frame_h // 2), "RGB", fill=(0, 0, 0), font=label_font, anchor="lm")
    draw.text((8, y_event + frame_h // 2), "EVENT", fill=(0, 0, 0), font=label_font, anchor="lm")

    for idx in range(4):
        pair_x = label_w + margin + idx * (pair_w + pair_gap)
        heatmap_x = pair_x + frame_w + inner_gap
        canvas.paste(rgb_frames[idx], (pair_x, y_rgb))
        canvas.paste(_overlay_heatmap(rgb_frames[idx], rgb_cams[idx]), (heatmap_x, y_rgb))
        canvas.paste(event_frames[idx], (pair_x, y_event))
        canvas.paste(_overlay_heatmap(event_frames[idx], event_cams[idx]), (heatmap_x, y_event))

    canvas.save(save_path)


def _resolve_query_indices(num_query, q_pids, vis_cfg):
    query_indices = vis_cfg.get("query_indices") or []
    query_pids = vis_cfg.get("pids") or []
    first_n_pids = int(vis_cfg.get("first_n_pids", 0) or 0)

    if query_indices:
        valid = sorted({idx for idx in query_indices if 0 <= idx < num_query})
        if len(valid) != len(query_indices):
            LOGGER.warning("Some vis query indices are out of range and were ignored.")
        if valid:
            return valid
        if num_query > 0:
            LOGGER.warning("No valid vis query indices matched the evaluation set. Falling back to the first query.")
            return [0]
        return []

    if query_pids:
        pid_set = set(query_pids)
        matched = [idx for idx, pid in enumerate(q_pids) if pid in pid_set]
        if matched:
            return matched
        if num_query > 0:
            unique_q_pids = np.unique(q_pids).tolist()
            LOGGER.warning(
                "None of the requested vis PIDs %s appeared in the query split. Falling back to the first query. "
                "Sample query PIDs: %s",
                sorted(pid_set),
                unique_q_pids[:10],
            )
            return [0]
        return []

    if first_n_pids > 0:
        unique_q_pids = list(dict.fromkeys(np.asarray(q_pids).tolist()))
        selected_pid_set = set(unique_q_pids[:first_n_pids])
        matched = [idx for idx, pid in enumerate(q_pids) if pid in selected_pid_set]
        if matched:
            return matched
        if num_query > 0:
            LOGGER.warning("Could not resolve the first %d query PIDs. Falling back to the first query.", first_n_pids)
            return [0]
        return []

    if num_query > 0:
        return [0]

    return []


def _gather_topk(dist_row, query_pid, query_camid, g_pids, g_camids, gallery_tracklets, topk):
    order = np.argsort(dist_row)
    results = []
    rank = 1
    for gallery_idx in order:
        if g_pids[gallery_idx] == query_pid and g_camids[gallery_idx] == query_camid:
            continue
        results.append({
            "rank": rank,
            "gallery_idx": int(gallery_idx),
            "pid": int(g_pids[gallery_idx]),
            "camid": int(g_camids[gallery_idx]),
            "dist": float(dist_row[gallery_idx]),
            "is_match": bool(g_pids[gallery_idx] == query_pid),
            "tracklet": gallery_tracklets[gallery_idx],
        })
        rank += 1
        if len(results) >= topk:
            break
    return results


def run_test_visualizations(cfg, model, evaluator, distmat, qf, gf, vis_config=None, vis_cfg=None, device="cuda"):
    # Accept both names during the transition so older/newer call sites won't break.
    vis_cfg = vis_config if vis_config is not None else vis_cfg
    if vis_cfg is None:
        vis_cfg = {}

    vis_root = os.path.join(cfg.OUTPUT_DIR, "visualizations")
    os.makedirs(vis_root, exist_ok=True)

    all_features = torch.cat([qf, gf], dim=0)
    all_pids = np.asarray(evaluator.pids)

    if vis_cfg.get("similarity"):
        sim_dir = os.path.join(vis_root, "similarity")
        fig = visualize_pos_neg_similarity(
            features=all_features,
            labels=torch.from_numpy(all_pids).long().to(all_features.device),
            method_name="Paths\nw/ MFA",
            writer=None,
            epoch=None,
            figure_size=(10, 8),
            save_dir=sim_dir,
        )
        if fig is not None:
            plt.close(fig)
        LOGGER.info("Saved similarity visualization to %s", sim_dir)

    if vis_cfg.get("tsne"):
        tsne_dir = os.path.join(vis_root, "tsne")
        os.makedirs(tsne_dir, exist_ok=True)
        q_pids_for_vis = np.asarray(evaluator.pids[:evaluator.num_query])
        if vis_cfg.get("pids"):
            draw_label = vis_cfg.get("pids")
        elif int(vis_cfg.get("first_n_pids", 0) or 0) > 0:
            draw_label = list(dict.fromkeys(q_pids_for_vis.tolist()))[: int(vis_cfg.get("first_n_pids", 0))]
        else:
            draw_label = np.unique(all_pids)[: min(10, len(np.unique(all_pids)))].tolist()
        evaluator.showPointMultiModal(all_features.cpu().numpy(), all_pids, list(draw_label), save_path=tsne_dir)
        LOGGER.info("Saved t-SNE visualization to %s", tsne_dir)

    if not vis_cfg.get("rank_list") and not vis_cfg.get("cam"):
        return

    q_pids = np.asarray(evaluator.pids[:evaluator.num_query])
    q_camids = np.asarray(evaluator.camids[:evaluator.num_query])
    q_viewids = np.asarray(evaluator.viewids[:evaluator.num_query])
    g_pids = np.asarray(evaluator.pids[evaluator.num_query:])
    g_camids = np.asarray(evaluator.camids[evaluator.num_query:])
    g_viewids = np.asarray(evaluator.viewids[evaluator.num_query:])
    query_tracklets = evaluator.tracklet_paths[:evaluator.num_query]
    gallery_tracklets = evaluator.tracklet_paths[evaluator.num_query:]

    selected_queries = _resolve_query_indices(evaluator.num_query, q_pids, vis_cfg)
    if not selected_queries:
        LOGGER.warning("Rank_list/CAM requested, but no valid query is available in the evaluation set.")
        return
    if not (vis_cfg.get("query_indices") or vis_cfg.get("pids") or vis_cfg.get("first_n_pids")):
        LOGGER.info("No vis query was specified. Falling back to the first query (index 0).")

    topk = int(vis_cfg.get("topk", 10))

    for query_idx in selected_queries:
        query_pid = int(q_pids[query_idx])
        query_camid = int(q_camids[query_idx])
        query_viewid = int(q_viewids[query_idx])
        query_tracklet = query_tracklets[query_idx]
        gallery_infos = _gather_topk(
            distmat[query_idx],
            query_pid,
            query_camid,
            g_pids,
            g_camids,
            gallery_tracklets,
            topk,
        )

        query_name = f"q{query_idx:04d}_pid{query_pid}_cam{query_camid}"

        if vis_cfg.get("rank_list"):
            rank_dir = os.path.join(vis_root, "rank_list")
            os.makedirs(rank_dir, exist_ok=True)
            rank_path = os.path.join(rank_dir, f"{query_name}.png")
            _save_rank_list_for_query(query_idx, query_pid, query_camid, query_tracklet, gallery_infos, rank_path)

        if vis_cfg.get("cam"):
            cam_dir = os.path.join(vis_root, "cam", query_name)
            os.makedirs(cam_dir, exist_ok=True)
            _save_cam_grid(
                query_tracklet,
                model,
                cfg,
                device,
                query_camid,
                query_viewid,
                f"Query {query_idx} | PID {query_pid} | Cam {query_camid}",
                os.path.join(cam_dir, "query.png"),
            )
            for gallery_info in gallery_infos:
                gallery_idx = gallery_info["gallery_idx"]
                gallery_viewid = int(g_viewids[gallery_idx])
                title = f"Rank {gallery_info['rank']} | PID {gallery_info['pid']} | Cam {gallery_info['camid']}"
                save_name = f"rank{gallery_info['rank']:02d}_pid{gallery_info['pid']}_cam{gallery_info['camid']}.png"
                _save_cam_grid(
                    gallery_info["tracklet"],
                    model,
                    cfg,
                    device,
                    int(gallery_info["camid"]),
                    gallery_viewid,
                    title,
                    os.path.join(cam_dir, save_name),
                )
