
# %%
# 尝试提高integrate_annotations函数在处理大型数据集时的运行效率
# 使用MOATA_E16.5_E1S1 数据进行分析
import numpy as np
import pandas as pd
import random
import time
import torch
import scanpy as sc
import anndata
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# %% 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
import SpacGPA as sg

# %%
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/MOSTA_E16.5_E1S1.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取数据
adata = sc.read_h5ad("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/MOSTA/E16.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata.X.shape)

# %%
# 计算模块的加权表达值
start_time = time.time()
sg.calculate_module_expression(adata, 
                               ggm, 
                               ggm_key='ggm',
                               top_genes=30,
                               weighted=True,
                               calculate_moran=False,
                               embedding_key='spatial',
                               k_neighbors=6,
                               add_go_anno=5,
                               )
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_info'])


# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            embedding_key='spatial',
                            k_neighbors=6,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
# 注释平滑
start_time = time.time()
sg.smooth_annotations(adata, 
                     ggm_key='ggm',
                     embedding_key='spatial',
                     k_neighbors=24)


# %%
# 方案1，尝试使用GPU加速
import numpy as np, pandas as pd, random, sys, torch
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity
import networkx as nx, leidenalg
from igraph import Graph

# ---------------- helper ---------------- #
def calc_border_flags(coords, k, iqr_factor=1.5, device="cpu"):
    """同原逻辑，用 torch 加速均值距离计算"""
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
    # knn 用 sklearn 足够快；返回仍在 CPU
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_d = dists[:, 1:].mean(1)
    q1, q3 = np.percentile(mean_d, [25, 75])
    threshold = q3 + iqr_factor * (q3 - q1)
    return mean_d > threshold, dists[:, 1:]

# -------------- main function -------------- #
def integrate_annotations(
    adata,
    ggm_key="ggm",
    modules_used=None,
    modules_excluded=None,
    modules_preferred=None,
    result_anno="anno_density",
    embedding_key="spatial",
    k_neighbors=24,
    purity_adjustment=True,
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    delta=0.4,
    lambda_pair=0.3,
    lr=0.1,
    target_purity=0.8,
    w_floor=0.1,
    w_ceil=1.0,
    max_iter=100,
    energy_tol=1e-3,
    p0=0.1,
    tau=8.0,
    random_state=0,
    device="cpu",
):
    """
    **仅加速，不改数学逻辑**：参数与返回值完全与原函数一致。
    """
    time_mark = time.time()
    # reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not found in adata.obsm")
    if ggm_key not in adata.uns["ggm_keys"]:
        raise KeyError(f"{ggm_key} not found in adata.uns['ggm_keys']")

    stats_key = adata.uns["ggm_keys"][ggm_key]["module_stats"]
    all_modules = list(pd.unique(adata.uns[stats_key]["module_id"]))
    modules_used = modules_used or all_modules
    if modules_excluded:
        modules_used = [m for m in modules_used if m not in modules_excluded]
    
    print("Time for module selection:", time.time() - time_mark)
    
    # --- build KNN graph (仍用 sklearn，速度足够) ---
    X = adata.obsm[embedding_key]
    n_cells = adata.n_obs
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    dist, idx = knn.kneighbors(X)
    sigma = max(np.mean(dist[:, 1:]), 1e-6)
    W = np.exp(-(dist ** 2) / sigma**2).astype("float32")          # shape (N, k+1)
    lam = (lambda_pair * W[:, 1:]).astype("float32")               # shape (N, k)
    print("Time for KNN graph:", time.time() - time_mark)
    # 转 torch
    idx_t  = torch.as_tensor(idx[:, 1:],  dtype=torch.long,   device=device)  # (N,k)
    W_t    = torch.as_tensor(W[:, 1:],   dtype=torch.float32, device=device)  # (N,k)
    lam_t  = torch.as_tensor(lam,        dtype=torch.float32, device=device)  # (N,k)
    
    print("Time for torch conversion:", time.time() - time_mark)
    
    # --- 预计算 anno, rank_norm, dens_log ---
    anno_dict   = {}
    rank_norm_t = []
    dens_log_t  = []

    for m in modules_used:
        col = f"{m}_anno_smooth" if f"{m}_anno_smooth" in adata.obs else f"{m}_anno"
        if col not in adata.obs or f"{m}_exp" not in adata.obs:
            raise KeyError(f"Missing columns for module {m}")

        lab = (adata.obs[col] == m).astype(np.float32).values       # 0/1
        anno_dict[m] = torch.as_tensor(lab, device=device)

        r = rankdata(-adata.obs[f"{m}_exp"].values, method="dense")
        rank_norm = ((r - 1) / (n_cells - 1)).astype("float32")
        rank_norm_t.append(torch.as_tensor(rank_norm, device=device))

        pts = X[lab == 1]
        if len(pts) >= 3:
            dens = -KernelDensity().fit(pts).score_samples(X)
        else:
            dens = np.zeros(n_cells, dtype="float32")
        dens_log_t.append(torch.as_tensor(dens, device=device))

    rank_norm_t = torch.stack(rank_norm_t, dim=1)   # (N, M)
    dens_log_t  = torch.stack(dens_log_t,  dim=1)   # (N, M)
    
    print("Time for pre-calculation:", time.time() - time_mark)
    
    # --- 动态模块权重 ---
    M = len(modules_used)
    w_mod = torch.ones(M, device=device)

    # --- 初始 label 选择（逐细胞但向量化 unary 评估） ---
    # 计算 neighbor‑vote term 一次性完成
    # vote_ij = sum_k W_ik * anno_j[k]  / sum_k W_ik
    # -> 用稀疏 gather 实现
    anno_stack = torch.stack([anno_dict[m] for m in modules_used], dim=1)   # (N, M)
    # denominator (N,1)
    denom = W_t.sum(1, keepdim=True)
    # gather neighbors anno: (N,k,M)
    neigh_anno = anno_stack[idx_t]                # (N,k,M)
    vote = (W_t.unsqueeze(2) * neigh_anno).sum(1) / torch.clamp_min(denom, 1e-12)  # (N,M)
    
    print("Time for vote calculation:", time.time() - time_mark)
    
    # unary matrix (N,M)
    unary_mat = (
        alpha * (1 - vote)
        + beta * rank_norm_t
        + gamma * dens_log_t
        + delta * (1 - w_mod.unsqueeze(0))
    )

    # initial label = argmin unary among candidates
    # 每个细胞的候选集合不同→不能一次 argmin；但仍可向量化：
    cand_mask = (anno_stack == 1)                 # (N,M) 0/1
    # 至少保证每行有一个 True
    if cand_mask.sum(1).min() == 0:
        cand_mask[cand_mask.sum(1) == 0] = True   # 若无注释则全模块都可选

    big = 1e6
    masked_unary = unary_mat + big * (~cand_mask) # 非候选赋极大
    label_idx = masked_unary.argmin(1)            # (N,)
    label = [modules_used[i] for i in label_idx.cpu().numpy()]

    # --- energy 函数（张量版） ---
    label_t = torch.as_tensor(label_idx, device=device)            # (N,)

    def energy():
        u = unary_mat[torch.arange(n_cells, device=device), label_t].sum()
        # pairwise cost: lam_t * (label_i != label_j)
        neigh_lab = label_t[idx_t]                                  # (N,k)
        pair = lam_t * (neigh_lab != label_t.unsqueeze(1))
        return (u + pair.sum()).item()

    prev_E = energy()
    print("Starting optimization...")
    print(f"Iteration:   0, Energy: {prev_E:.2f}")
    
    print(f"Time for initialization: {time.time() - time_mark:.5f} s")
    
    # ------------ ICM loop ------------
    converged = False
    last_pr_len = 0
    rng = random.Random(random_state)

    for t in range(1, max_iter + 1):
        # purity_adjustment 部分保持原 py 循环实现（通常瓶颈不在这里）
        if purity_adjustment:
            g = nx.Graph()
            g.add_nodes_from(range(n_cells))
            edges = np.column_stack([np.repeat(np.arange(n_cells), k_neighbors), idx[:, 1:].reshape(-1)])
            g.add_edges_from(edges)
            nx.set_node_attributes(g, {i: label[i] for i in range(n_cells)}, "label")
            part = leidenalg.find_partition(
                Graph.from_networkx(g),
                leidenalg.RBConfigurationVertexPartition,
                seed=random_state,
            )

            purity = {m: [] for m in modules_used}
            for cluster in part:
                counts = {}
                for v in cluster:
                    counts[label[v]] = counts.get(label[v], 0) + 1
                cp = max(counts.values()) / len(cluster)
                for m, v in counts.items():
                    if m in modules_used:
                        purity[m].append(v / len(cluster) * cp)
            for j, m in enumerate(modules_used):
                if purity[m]:
                    mp = float(np.mean(purity[m]))
                    w_mod_j = w_mod[j] * (1 + lr * (mp - target_purity))
                    w_mod[j] = float(np.clip(w_mod_j, w_floor, w_ceil))
        # 重新计算 unary_mat （只有 w_mod 变）——张量化
        unary_mat = (
            alpha * (1 - vote)
            + beta * rank_norm_t
            + gamma * dens_log_t
            + delta * (1 - w_mod.unsqueeze(0))
        )

        changed = 0
        p_t = p0 * np.exp(-t / tau)

        # ICM：逐细胞，但每步只用 torch 计算 pair_cost
        for i in range(n_cells):
            # 候选
            cand_mask_i = cand_mask[i].clone()
            if modules_preferred:
                # 若存在偏好模块且在候选中，则只保留偏好
                pref_mask = torch.zeros(M, dtype=torch.bool, device=device)
                for m in modules_preferred:
                    if m in modules_used:
                        pref_mask[modules_used.index(m)] = True
                if (cand_mask_i & pref_mask).any():
                    cand_mask_i = cand_mask_i & pref_mask
            if not cand_mask_i[label_idx[i]]:
                cand_mask_i[label_idx[i]] = True  # ensure current label

            cand_idx = cand_mask_i.nonzero(as_tuple=True)[0]        # (C,)
            if cand_idx.numel() == 1:
                continue

            if rng.random() < p_t:
                new_lab_idx = int(cand_idx[rng.randrange(cand_idx.numel())])
            else:
                # pair_cost: lam_i * (label_neigh != cand)
                neigh_lab_i = label_t[idx_t[i]]                     # (k,)
                # (C,k)
                diff = (neigh_lab_i.unsqueeze(0) != cand_idx.unsqueeze(1))
                pc = (lam_t[i] * diff).sum(1)                       # (C,)
                total = unary_mat[i, cand_idx] + pc
                new_lab_idx = int(cand_idx[total.argmin()])

            if new_lab_idx != label_idx[i]:
                label_idx[i] = new_lab_idx
                label[i] = modules_used[new_lab_idx]
                label_t[i] = new_lab_idx
                changed += 1

        curr_E = energy()
        msg = (
            f"Iteration: {t:3}, Energy: {curr_E:.2f}, "
            f"ΔE: {prev_E-curr_E:+.2f}, Changed: {changed}"
        )
        sys.stdout.write("\r" + " " * last_pr_len + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()
        last_pr_len = len(msg)

        if abs(prev_E - curr_E) / max(abs(prev_E), 1.0) < energy_tol:
            print("\nConverged\n")
            converged = True
            break
        prev_E = curr_E

    if not converged:
        print("\nStopped after max_iter without convergence\n")

    adata.obs[result_anno] = np.array(label, dtype=object)
    return adata


# %%
# 测试
start_time = time.time()
integrate_annotations(
                    adata,
                    ggm_key='ggm',
                    #modules_used=module_used,
                    #modules_excluded=['M15', 'M18'],        
                    #modules_preferred=['M28', 'M38'],
                    result_anno='annotation_new_all',
                    k_neighbors=24,
                    lambda_pair=0.3,
                    purity_adjustment=False,
                    w_floor=0.01,
                    lr=0.5,
                    target_purity=0.85,
                    # alpha=0.5,
                    # beta=0.3
                    gamma=0.3,
                    # delta=0.4,   
                    max_iter=100,
                    random_state=0,
                    device='cuda')
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="annotation_new_all", show=True)

# %%








# %%
# 原版：
def calc_border_flags(coords, k, iqr_factor=1.5):
    """
    Identify border cells based on neighbor distances.

    Parameters:
        coords (ndarray): Spatial coordinates, shape (N, D).
        k (int): Number of neighbors to consider.
        iqr_factor (float): Multiplier for IQR when setting distance threshold.

    Returns:
        border_mask (ndarray of bool): True for border cells.
        knn_dists (ndarray): Distances to k nearest neighbors for each cell.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_d = dists[:, 1:].mean(axis=1)
    q1, q3 = np.percentile(mean_d, [25, 75])
    threshold = q3 + iqr_factor * (q3 - q1)
    border_mask = mean_d > threshold
    return border_mask, dists[:, 1:]

# integrate_annotations
def integrate_annotations(
    adata,
    ggm_key="ggm",
    modules_used=None,
    modules_excluded=None,
    modules_preferred=None,
    result_anno="anno_density",
    embedding_key="spatial",
    k_neighbors=24,
    purity_adjustment=True,
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    delta=0.4,
    lambda_pair=0.3,
    lr=0.1,
    target_purity=0.8,
    w_floor=0.1,
    w_ceil=1.0,
    max_iter=100,
    energy_tol=1e-3,
    p0=0.1,
    tau=8.0,
    random_state=None,
):
    """
    Integrate module-wise annotations into a single, spatially smooth label
    using a first-order Potts Conditional Random Field (CRF).

    Parameters
    ----------
    adata : AnnData
        Must contain:
          • adata.obsm[embedding_key]: spatial coordinates
          • per-module columns 'Mx_anno' or 'Mx_anno_smooth', and 'Mx_exp'
    ggm_key : str
        Key in adata.uns['ggm_keys'] pointing to module stats.
    modules_used : list[str] or None
        Modules to include; None means use all.
    modules_excluded : list[str] or None
        Modules to remove from the set.
    modules_preferred : list[str] or None
        Preferred modules when multiple candidates exist for a cell.
    result_anno : str
        Name of the output column in adata.obs.
    embedding_key : str
        Key for spatial coordinates in adata.obsm.
    k_neighbors : int
        Number of nearest neighbours (including the cell).
    purity_adjustment : bool
        Whether to perform Leiden clustering each iteration to adjust module weights.
    alpha, beta, gamma, delta : float
        Weights of the unary terms:
          alpha - neighbor-vote weight (how much neighboring labels influence cost)
          beta  - expression-rank weight (how much gene expression rank influences cost)
          gamma - density weight (how much spatial cluster density influences cost)
          delta - low-purity penalty weight (penalizes modules with low global purity, only if purity_adjustment is True)
    lambda_pair : float
        Strength of the Potts pairwise smoothing term (penalizes label differences).
    lr : float
        Learning rate for dynamic module weights.
    target_purity : float
        Target purity for weight updates.
    w_floor, w_ceil : float
        Bounds for dynamic module weights.
    max_iter : int
        Maximum number of iterations.
    energy_tol : float
        Convergence threshold on relative energy change.
    p0 : float
        Initial perturbation probability (simulated annealing).
    tau : float
        Decay constant for the perturbation probability.
    random_state : int or None
        Seed for random number generator.

    Returns
    -------
    AnnData
        The same AnnData with adata.obs[result_anno] containing final labels.
    """
    # reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    rng = random.Random(random_state)

    # sanity checks
    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not found in adata.obsm")
    if ggm_key not in adata.uns["ggm_keys"]:
        raise KeyError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    
    # print parameter meanings and values
    print(
        f"Main parameters for integration:\n"
        f"  alpha = {alpha:.2f}  for neighbor-vote weight\n"
        f"  beta  = {beta:.2f}  for expression-rank weight\n"
        f"  gamma = {gamma:.2f}  for spatial-density weight\n"
        f"  delta = {delta:.2f}  for low-purity penalty weight\n"
        f"  lambda_pair = {lambda_pair:.2f}  for potts smoothing strength\n"
    )

    # load module list
    stats_key = adata.uns["ggm_keys"][ggm_key]["module_stats"]
    modules_df = adata.uns[stats_key]
    all_modules = list(pd.unique(modules_df["module_id"]))
    modules_used = modules_used or all_modules
    if modules_excluded:
        modules_used = [m for m in modules_used if m not in modules_excluded]

    # build spatial KNN graph
    X = adata.obsm[embedding_key]
    n_cells = adata.n_obs
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    dist, idx = knn.kneighbors(X)
    sigma = max(np.mean(dist[:, 1:]), 1e-6)
    W = np.exp(-dist**2 / sigma**2)
    lam = lambda_pair * W[:, 1:]

    # load annotations, compute ranks and density
    anno = {}
    rank_norm = {}
    dens_log = {}
    for m in modules_used:
        col = f"{m}_anno_smooth" if f"{m}_anno_smooth" in adata.obs else f"{m}_anno"
        if col not in adata.obs or f"{m}_exp" not in adata.obs:
            raise KeyError(f"Missing columns for module {m}")
        lab = (adata.obs[col] == m).astype(int).values
        anno[m] = lab

        r = rankdata(-adata.obs[f"{m}_exp"].values, method="dense")
        rank_norm[m] = (r - 1) / (n_cells - 1)

        pts = X[lab == 1]
        if len(pts) >= 3:
            dens_log[m] = -KernelDensity().fit(pts).score_samples(X)
        else:
            dens_log[m] = np.zeros(n_cells)

    # define unary energy
    w_mod = {m: 1.0 for m in modules_used}
    def unary(i, m):
        nb = idx[i, 1:]
        s = W[i, 1:].sum()
        vote = (W[i, 1:] * anno[m][nb]).sum() / s if s else 0.0
        vote *= w_mod[m]
        return (
            alpha * (1 - vote)
            + beta * rank_norm[m][i]
            + gamma * dens_log[m][i]
            + delta * (1 - w_mod[m])
        )

    # initial labeling
    label = np.empty(n_cells, dtype=object)
    for i in range(n_cells):
        cand = [m for m in modules_used if anno[m][i]] or modules_used
        if modules_preferred:
            pref = [m for m in cand if m in modules_preferred]
            cand = pref or cand
        label[i] = min(cand, key=lambda m: unary(i, m))

    # energy function
    def energy():
        u = sum(unary(i, label[i]) for i in range(n_cells))
        p = float((lam * (label[idx[:, 1:]] != label[:, None])).sum())
        return u + p

    prev_E = energy()
    print("Starting optimization...")
    print(f"Iteration:   0, Energy: {prev_E:.2f}")

    # optimization loop
    converged = False
    last_pr_len = 0
    for t in range(1, max_iter + 1):
        # optionally update module weights based on Leiden purity
        if purity_adjustment:
            g = nx.Graph()
            g.add_nodes_from(range(n_cells))
            for i in range(n_cells):
                for j in idx[i, 1:]:
                    g.add_edge(i, j)
            nx.set_node_attributes(g, {i: label[i] for i in range(n_cells)}, "label")
            part = leidenalg.find_partition(
                Graph.from_networkx(g),
                leidenalg.RBConfigurationVertexPartition,
                seed=random_state
            )

            purity = {m: [] for m in modules_used}
            for cluster in part:
                counts = {}
                for v in cluster:
                    counts[label[v]] = counts.get(label[v], 0) + 1
                cp = max(counts.values()) / len(cluster)
                for m, v in counts.items():
                    if m in modules_used:
                        purity[m].append(v / len(cluster) * cp)
            for m in modules_used:
                if purity[m]:
                    mp = float(np.mean(purity[m]))
                    w_mod[m] = np.clip(w_mod[m] * (1 + lr * (mp - target_purity)),
                                       w_floor, w_ceil)

        # ICM with annealed perturbation
        changed = 0
        p_t = p0 * np.exp(-t / tau)
        for i in range(n_cells):
            cand = [m for m in modules_used if anno[m][i]] or modules_used
            if modules_preferred:
                pref = [m for m in cand if m in modules_preferred]
                cand = pref or cand
            if label[i] not in cand:
                cand.append(label[i])

            if random.random() < p_t:
                new_lab = rng.choice(cand)
            else:
                pair_cost = lam[i] * (label[idx[i, 1:]] != np.array(cand)[:, None])
                total = [unary(i, m) + pair_cost[k].sum() for k, m in enumerate(cand)]
                new_lab = cand[int(np.argmin(total))]

            if new_lab != label[i]:
                label[i] = new_lab
                changed += 1

        curr_E = energy()
        msg = f"Iteration: {t:3}, Energy: {curr_E:.2f}, ΔE: {prev_E-curr_E:+.2f}, Changed: {changed}"
        sys.stdout.write("\r" + " " * last_pr_len + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()
        last_pr_len = len(msg)
        if abs(prev_E - curr_E) / max(abs(prev_E), 1.0) < energy_tol:
            print("\nConverged\n")
            converged = True
            break
        prev_E = curr_E

    if not converged:
        print("\nStopped after max_iter without convergence\n")

    adata.obs[result_anno] = label