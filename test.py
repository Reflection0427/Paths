import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paths Testing")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--vis-enable", action="store_true", help="enable test-time visualizations")
    parser.add_argument("--vis-pids", type=int, nargs="+", default=None, help="query PIDs to visualize")
    parser.add_argument("--vis-first-n-pids", type=int, default=0, help="visualize the first N unique query PIDs")
    parser.add_argument("--vis-query-indices", type=int, nargs="+", default=None, help="query indices to visualize")
    parser.add_argument("--vis-topk", type=int, default=10, help="top-k gallery results for rank list")
    parser.add_argument("--vis-cam", action="store_true", help="generate CAM visualizations")
    parser.add_argument("--vis-rank-list", action="store_true", help="generate rank list visualizations")
    parser.add_argument("--vis-tsne", action="store_true", help="generate t-SNE visualization")
    parser.add_argument("--vis-similarity", action="store_true", help="generate positive/negative similarity visualization")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("Paths", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.eval()

    weight_path = cfg.TEST.WEIGHT
    if weight_path == "":
        weight_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_best.pth")
    logger.info("Loading test weight from: %s", weight_path)
    model.load_param(weight_path)
    vis_flags = [args.vis_cam, args.vis_rank_list, args.vis_tsne, args.vis_similarity]
    vis_requested = (
        args.vis_enable
        or any(vis_flags)
        or args.vis_pids is not None
        or args.vis_query_indices is not None
        or args.vis_first_n_pids > 0
    )
    if vis_requested and not any(vis_flags):
        args.vis_cam = True
        args.vis_rank_list = True
        args.vis_tsne = True
        args.vis_similarity = True

    vis_config = {
        "enabled": vis_requested,
        "pids": args.vis_pids,
        "first_n_pids": args.vis_first_n_pids,
        "query_indices": args.vis_query_indices,
        "topk": args.vis_topk,
        "cam": args.vis_cam,
        "rank_list": args.vis_rank_list,
        "tsne": args.vis_tsne,
        "similarity": args.vis_similarity,
    }
    logger.info("Visualization config: %s", vis_config)
    do_inference(cfg, model, val_loader, num_query, return_pattern=3, vis_config=vis_config)
