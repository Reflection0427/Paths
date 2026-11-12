import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import seaborn as sns
from sklearn import manifold
import torch.nn.functional as F
import pandas as pd
import json

# Import re_ranking; fall back to a stub if it is unavailable
try:
    from utils.reranking import re_ranking
except ImportError:
    print("[Warning] 'utils.reranking' not found. Re-ranking functionality will be disabled.")
    re_ranking = None


class R1_mAP_eval:
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.viewids = []
        self.img_paths = []  # Store absolute paths of RGB images
        self.tracklet_paths = []

    @staticmethod
    def _to_tracklet_paths(img_paths):
        tracklet_paths = []
        if img_paths is None:
            return tracklet_paths

        if isinstance(img_paths, np.ndarray):
            batch_items = [img_paths]
        elif isinstance(img_paths, (list, tuple)):
            batch_items = list(img_paths)
        else:
            return tracklet_paths

        for item in batch_items:
            if isinstance(item, np.ndarray):
                item = item.tolist()

            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[0], (list, tuple, np.ndarray))
                and isinstance(item[1], (list, tuple, np.ndarray))
            ):
                rgb_paths = [str(p) for p in list(item[0])]
                event_paths = [str(p) for p in list(item[1])]
            elif isinstance(item, (list, tuple)):
                rgb_paths = [str(p) for p in list(item)]
                event_paths = [str(p) for p in list(item)]
            else:
                path = str(item)
                rgb_paths = [path]
                event_paths = [path]

            tracklet_paths.append({
                "rgb_paths": rgb_paths,
                "event_paths": event_paths,
            })

        return tracklet_paths

    def update(self, output):
        if len(output) == 4:
            feat, pid, camid, img_paths = output
            viewids = None
        else:
            feat, pid, camid, img_paths, viewids = output

        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        if viewids is None:
            self.viewids.extend([0] * len(np.asarray(pid)))
        else:
            self.viewids.extend(np.asarray(viewids).astype(int).tolist())

        tracklet_batch = self._to_tracklet_paths(img_paths)
        self.tracklet_paths.extend(tracklet_batch)
        self.img_paths.extend([
            tracklet["rgb_paths"][0] if tracklet["rgb_paths"] else ""
            for tracklet in tracklet_batch
        ])

    def compute(self):
        feats = torch.cat(self.feats, dim=0)

        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # Split Query and Gallery
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking and re_ranking is not None:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        # Optional: Visualize Top-10 results (can be called externally or here)
        # self.visualize_ranked_results(distmat, topk=10, save_dir='vis_results')

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    def load_image(self, path):
        """Helper to safely load images."""
        if not os.path.exists(path):
            print(f"[Warning] Image not found: {path}")
            return np.zeros((256, 128, 3), dtype=np.uint8)

        try:
            image = Image.open(path).convert('RGB')
            return np.array(image)
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            return np.zeros((256, 128, 3), dtype=np.uint8)

    def visualize_ranked_results(self, distmat, topk=10, save_dir='vis_results'):
        """
        Visualize Top-K ranking results for RGB-Event ReID.
        Shows query image and ranking scores.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Visualizing top-{topk} results...")

        # Limit visualization to prevent long wait times (e.g., first 20 queries)
        num2vis = min(self.num_query, len(self.img_paths), 20)

        for i in range(num2vis):
            # check that the query path is valid
            if i >= len(self.img_paths):
                print(f"[Warning] Query index {i} out of range for img_paths")
                continue

            query_path = self.img_paths[i]
            query_pid = self.pids[i]
            query_camid = self.camids[i]

            # Load Query Image (RGB or Event)
            query_img = self.load_image(query_path)

            # Sort indices
            index = np.argsort(distmat[i])

            ranked_indices = []
            ranked_distances = []
            for idx in index:
                # Convert relative gallery index to global index
                gallery_real_idx = idx + self.num_query

                # check that the index is valid
                if gallery_real_idx >= len(self.pids) or gallery_real_idx >= len(self.camids):
                    continue

                # Market1501 protocol: Remove junk (same ID, same Camera)
                if self.camids[gallery_real_idx] == query_camid and self.pids[gallery_real_idx] == query_pid:
                    continue

                ranked_indices.append(gallery_real_idx)
                ranked_distances.append(distmat[i][idx])
                if len(ranked_indices) >= topk:
                    break

            # PIDs and distances of the ranked gallery
            gallery_pids_vis = [self.pids[g_idx] for g_idx in ranked_indices]

            # draw the query and its ranking
            self.plot_query_with_ranking(
                query_img,
                gallery_pids_vis,
                ranked_distances,
                query_pid,
                save_path=os.path.join(save_dir, f'query_{i}_pid{query_pid}.png')
            )

    def plot_query_with_ranking(self, query_img, gallery_pids, distances, query_pid, save_path=None):
        """
        Draw Query image and ranking information.
        query_img: single image (RGB or Event)
        gallery_pids: list of PIDs in ranking order
        distances: list of distances
        """
        topk = len(gallery_pids)

        # Create figure: query image on left, ranking info on right
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Plot Query Image
        ax1.imshow(query_img)
        ax1.set_title(f"Query (PID: {query_pid})", fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Add ranking information as text
        ranking_text = "Top-K Ranking Results:\n\n"
        for rank, (pid, dist) in enumerate(zip(gallery_pids, distances), 1):
            match_status = "✓ MATCH" if pid == query_pid else "✗ MISMATCH"
            ranking_text += f"Rank {rank:2d}: PID {pid:4d} | Dist: {dist:.4f} | {match_status}\n"

        # Add text to the second subplot
        ax2.text(0.05, 0.95, ranking_text, transform=ax2.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def plot_images(self, query_imgs, gallery_imgs, gallery_pids, query_pid, save_path=None):
        """
        Draw Query (col 0) and Gallery (col 1-N).
        query_imgs: [rgb_img, event_img]
        gallery_imgs: [[rgb_list...], [event_list...]]
        """
        if not gallery_imgs or not gallery_imgs[0]:
            return

        num_results = len(gallery_imgs[0])
        modalities = ['RGB', 'EVENT']

        # Dynamic figure size
        fig, axs = plt.subplots(2, num_results + 1, figsize=(2 * (num_results + 1), 5))

        # 1. Plot Query
        for row, (img, modal) in enumerate(zip(query_imgs, modalities)):
            axs[row, 0].imshow(img)
            axs[row, 0].set_title(f"Query {modal}\n{query_pid}", fontsize=10)
            axs[row, 0].axis('off')

        # 2. Plot Gallery
        for row, (imgs_list, modal) in enumerate(zip(gallery_imgs, modalities)):
            for col, (img, pid) in enumerate(zip(imgs_list, gallery_pids)):
                ax = axs[row, col + 1]
                ax.imshow(img)
                ax.axis('off')

                # Title only on top row
                if row == 0:
                    ax.set_title(f"Rank {col + 1}\n{pid}", fontsize=10)

                # Border: Green = Correct, Red = Wrong
                color = 'green' if pid == query_pid else 'red'
                rect = patches.Rectangle((0, 0), img.shape[1] - 1, img.shape[0] - 1,
                                         linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            # print(f"Saved to {save_path}")
        plt.close()

    def showPointMultiModal(self, features, real_label, draw_label, save_path='../TSNE'):
        """t-SNE Visualization helper"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        full_save_path = os.path.join(save_path, str(draw_label) + ".png")
        print("Draw points of features to {}".format(full_save_path))

        indices = find_label_indices(real_label, draw_label, max_indices_per_label=25)
        if not indices:
            print(f"No indices found for label {draw_label}")
            return

        feat = features[indices]
        num_samples = feat.shape[0]
        if num_samples < 2:
            print(f"Skip t-SNE for {draw_label}: need at least 2 samples, got {num_samples}")
            return

        perplexity = min(30, num_samples - 1)
        tsne = manifold.TSNE(
            n_components=2,
            init='pca',
            random_state=1,
            learning_rate=100,
            perplexity=perplexity,
        )
        features_tsne = tsne.fit_transform(feat)

        plt.figure(figsize=(8, 8), facecolor='white')
        label_sequence = list(dict.fromkeys(draw_label))
        cmap_name = 'tab20' if len(label_sequence) <= 20 else 'gist_ncar'
        cmap = plt.get_cmap(cmap_name, max(len(label_sequence), 1))
        for label_idx, label in enumerate(label_sequence):
            label_mask = np.asarray(real_label)[indices] == label
            if not np.any(label_mask):
                continue
            plt.scatter(
                features_tsne[label_mask, 0],
                features_tsne[label_mask, 1],
                s=90,
                color=cmap(label_idx),
                marker='*',
                alpha=0.65,
                linewidths=0.6,
            )

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_facecolor('white')

        plt.tight_layout(pad=0.4)
        plt.savefig(full_save_path, format='png', bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close()


# --- Independent Helper Functions ---

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def find_label_indices(label_list, target_labels, max_indices_per_label=1):
    indices = []
    counts = {label: 0 for label in target_labels}
    for index, label in enumerate(label_list):
        if label in target_labels and counts[label] < max_indices_per_label:
            indices.append(index)
            counts[label] += 1
    sorted_indices = sorted(indices, key=lambda index: (label_list[index], index))
    return sorted_indices


def _calculate_pos_neg_similarities(features, labels):
    """
    Cosine similarities of positive (same ID) and negative (different ID) pairs.

    Args:
        features (torch.Tensor): features of shape (N, D)
        labels (torch.Tensor): identity labels of shape (N,)

    Returns:
        pos_sims (np.ndarray): similarities of positive pairs
        neg_sims (np.ndarray): similarities of negative pairs
    """
    # 1. L2-normalize so that the dot product is the cosine similarity
    features = F.normalize(features, p=2, dim=1)

    # 2. pairwise similarity matrix (N, N)
    sim_matrix = torch.matmul(features, features.T)

    # 3. positive / negative masks
    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T)  # True for the same ID
    neg_mask = ~pos_mask  # True for different IDs

    # 4. drop the diagonal (self-similarity is always 1)
    mask_self = torch.eye(labels.size(0), dtype=torch.bool, device=features.device)
    pos_mask.fill_diagonal_(False)

    # 5. gather the similarities
    pos_sims = sim_matrix[pos_mask].cpu().detach().numpy()
    neg_sims = sim_matrix[neg_mask].cpu().detach().numpy()

    return pos_sims, neg_sims


def visualize_pos_neg_similarity(features, labels, method_name="Ours",
                                 writer=None, epoch=None, figure_size=(8, 8), save_dir=None):
    """
    KDE plot of positive / negative pair similarities (paper visualization).

    Args:
        features: test features (N, D)
        labels: identity labels (N,)
        method_name: label drawn on the left of the plot
        save_dir: output directory for the figure and values
    """
    # positive / negative similarities
    pos_sims, neg_sims = _calculate_pos_neg_similarities(features, labels)

    # statistics
    mean_pos = np.mean(pos_sims)
    mean_neg = np.mean(neg_sims)
    std_pos = np.std(pos_sims)
    std_neg = np.std(neg_sims)
    diff = mean_pos - mean_neg

    # plot style
    plt.rcParams.update({'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=figure_size)

    # 1. KDE curves
    sns.kdeplot(pos_sims, color='#7CAE7A', fill=True, alpha=0.4,
                label='Positive Pairs (Same ID)', ax=ax, linewidth=1)
    sns.kdeplot(neg_sims, color='#F07B7A', fill=True, alpha=0.4,
                label='Negative Pairs (Different ID)', ax=ax, linewidth=1)

    # 2. dashed mean lines
    ax.axvline(mean_pos, color='#3B7A33', linestyle='--', linewidth=1.5,
               label=f'Mean Positive Similarity: {mean_pos:.4f}')
    ax.axvline(mean_neg, color='#E23D3D', linestyle='--', linewidth=1.5,
               label=f'Mean Negative Similarity: {mean_neg:.4f}')

    # axis limits, used to place text and arrows
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    # 3. mean values at the bottom of the x-axis
    ax.text(mean_pos, ymin + (ymax - ymin) * 0.01, f'{mean_pos:.4f}',
            color='#3B7A33', ha='center', va='bottom', fontsize=14)
    ax.text(mean_neg, ymin + (ymax - ymin) * 0.01, f'{mean_neg:.4f}',
            color='#E23D3D', ha='center', va='bottom', fontsize=14)

    # 4. double-headed arrow
    arrow_y = ymax * 0.65  # arrow at 65% of the plot height
    ax.annotate('', xy=(mean_neg, arrow_y), xytext=(mean_pos, arrow_y),
                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=4, mutation_scale=2.0))

    # 5. gap value in red
    ax.text((mean_pos + mean_neg) / 2, arrow_y - (ymax * 0.03), f'{diff:.4f}',
            color='#E64B35', fontsize=36, fontweight='bold', ha='center', va='top')

    # 6. method name on the left
    ax.text(xmin + (xmax - xmin) * 0.20, ymax * 0.45, method_name,
            fontsize=40, fontweight='bold', color='black', ha='center', va='center')

    # 7. layout
    ax.set_xlabel('Cosine Similarity', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # light dashed grid
    ax.grid(True, linestyle='--', alpha=0.4)

    # x-axis range (similarities are usually in [-0.2, 1.0])
    ax.set_xlim(-0.1, 1.1)

    # legend in the upper left
    ax.legend(loc='upper left', fontsize=14, framealpha=1.0, edgecolor='black')

    plt.tight_layout()

    # ================== save values ==================
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        fig.savefig(
            os.path.join(save_dir, 'pos_neg_similarity.png'),
            dpi=150,
            bbox_inches='tight'
        )

        # 1. raw similarities as NPZ
        np.savez(
            os.path.join(save_dir, 'similarity_data.npz'),
            pos_sims=pos_sims,
            neg_sims=neg_sims
        )

        # 2. statistics as CSV
        stats_data = {
            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Count'],
            'Positive Pairs': [
                mean_pos,
                std_pos,
                np.min(pos_sims),
                np.max(pos_sims),
                len(pos_sims)
            ],
            'Negative Pairs': [
                mean_neg,
                std_neg,
                np.min(neg_sims),
                np.max(neg_sims),
                len(neg_sims)
            ],
            'Difference': [diff, np.nan, np.nan, np.nan, np.nan]
        }

        import pandas as pd
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(os.path.join(save_dir, 'similarity_stats.csv'), index=False)

        # 3. statistics as JSON
        import json
        json_data = {
            'method_name': method_name,
            'positive_pairs': {
                'mean': float(mean_pos),
                'std': float(std_pos),
                'min': float(np.min(pos_sims)),
                'max': float(np.max(pos_sims)),
                'count': int(len(pos_sims))
            },
            'negative_pairs': {
                'mean': float(mean_neg),
                'std': float(std_neg),
                'min': float(np.min(neg_sims)),
                'max': float(np.max(neg_sims)),
                'count': int(len(neg_sims))
            },
            'difference': float(diff)
        }

        with open(os.path.join(save_dir, 'similarity_stats.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"✓ Similarity data saved to {save_dir}")
        print(f"  - similarity_data.npz (raw arrays)")
        print(f"  - similarity_stats.csv (statistics)")
        print(f"  - similarity_stats.json (JSON format)")
    # =============================================

    # log to TensorBoard
    if writer is not None:
        writer.add_figure(f'Similarity_Pos_Neg', fig, epoch)

    return fig
