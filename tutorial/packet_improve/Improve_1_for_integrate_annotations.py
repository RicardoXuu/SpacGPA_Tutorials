
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

adata.X = adata.layers['count']

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
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
# 分析模块类型
start_time = time.time()
sg.classify_modules(adata, 
                    ggm_key='ggm',
                    ref_anno='annotation',
                    #ref_cluster_method='leiden',
                    #ref_resolution=0.5,
                    skew_threshold=2,
                    top1pct_threshold=2,
                    Moran_I_threshold=0.2,
                    min_dominant_cluster_fraction=0.2,
                    anno_overlap_threshold=0.4)

# %%
adata.uns['module_filtering']['type_tag'].value_counts()

# %%
# 设置配色方案
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors
from scanpy.plotting.palettes import default_20, vega_10, vega_20

def assign_module_colors(adata, ggm_key='ggm', seed=1):
    """
    Generate and store a consistent color palette for modules based on the number of modules,
    using Scanpy/Squidpy‐style discrete palettes for ≤100 modules and XKCD names for >100 modules,
    with optional random shuffling controlled by `seed`.

    The palette is automatically selected based on module count:
      - ≤10    modules: vega_10
      - ≤20    modules: default_20
      - ≤40    modules: vega_20
      - ≤60    modules: default_20 + vega_10 + first 8 of tab20b
      - ≤100   modules: merged [default_20, vega_20, tab20b, tab20c, Set3, Pastel2]
      - >100   modules: use matplotlib’s XKCD_COLORS, filtered, sorted by HSV, then sampled

    If seed != 0, sampling and final order are shuffled reproducibly; if seed == 0, the order is deterministic.

    Parameters:
      adata: AnnData object containing module statistics in adata.uns['ggm_keys'][ggm_key]['module_stats'].
      ggm_key: str, key for the GGM metadata in adata.uns.
      seed: int, random seed (0 = no shuffle; otherwise reproducible sampling + shuffle).

    Returns:
      dict mapping module IDs to color names or hex codes.
    """
    # ——前置检查——
    ggm_meta = adata.uns.get('ggm_keys', {}).get(ggm_key)
    if ggm_meta is None:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = ggm_meta.get('module_stats')
    mod_col_val   = ggm_meta.get('module_colors')

    module_stats = adata.uns.get(mod_stats_key)
    if module_stats is None:
        raise ValueError(f"Module statistics not found in adata.uns['{mod_stats_key}']")
    df = module_stats if isinstance(module_stats, pd.DataFrame) else pd.DataFrame(module_stats)
    module_ids = df['module_id'].tolist()

    n_all = len(module_ids)
    if n_all == 0:
        return {}
    n_modules = min(n_all, 806)

    # 初始化 RNG（seed=0 不使用）
    rng = np.random.RandomState(seed) if seed != 0 else None

    # ——选择配色方案——
    if n_modules <= 10:
        colors = vega_10[:n_modules]
    elif n_modules <= 20:
        colors = default_20[:n_modules]
    elif n_modules <= 40:
        colors = vega_20[:n_modules]
    elif n_modules <= 60:
        tab20b = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20b').colors[:8]]
        combo = default_20 + vega_10 + tab20b
        colors = combo[:n_modules]
    elif n_modules <= 100:
        tab20b  = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20b').colors]
        tab20c  = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20c').colors]
        set3    = [mpl.colors.to_hex(c) for c in plt.get_cmap('Set3').colors]
        pastel2 = [mpl.colors.to_hex(c) for c in plt.get_cmap('Pastel2').colors]
        palette_combo = default_20 + vega_20 + tab20b + tab20c + set3 + pastel2
        colors = palette_combo[:n_modules]
    else:
        # >100 modules: use XKCD_COLORS filtered & HSV-sampled
        xkcd_colors = mcolors.XKCD_COLORS
        to_remove = ['gray','grey','black','white','light',
                     'lawngreen','silver','gainsboro','snow',
                     'mintcream','ivory','fuchsia','cyan']
        filtered = {
            name: col for name, col in xkcd_colors.items()
            if not any(key in name for key in to_remove)
        }
        # sort by HSV using tuples
        sorted_hsv = sorted(
            ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(col)[:3])), name))
            for name, col in filtered.items()
        )
        sorted_names = [name for hsv, name in sorted_hsv]
        if rng is not None:
            # random sampling without replacement if possible
            if len(sorted_names) >= n_modules:
                colors = list(rng.choice(sorted_names, size=n_modules, replace=False))
            else:
                base = sorted_names.copy()
                rem = n_modules - len(base)
                vir = plt.cm.viridis
                extra = [mpl.colors.to_hex(vir(i/(rem-1))) for i in range(rem)]
                colors = base + extra
        else:
            # uniformly spaced sampling along sorted list
            L = len(sorted_names)
            step = L / n_modules
            colors = [sorted_names[int(i * step)] for i in range(n_modules)]

    # ——处理超过806的额外模块——
    if n_all > 806:
        extra_count = n_all - 806
        if rng is not None:
            extra = list(rng.choice(colors, size=extra_count, replace=True))
        else:
            extra = [colors[i % len(colors)] for i in range(extra_count)]
        colors += extra

    # ——全局打乱——
    if rng is not None:
        perm = rng.permutation(len(colors))
        colors = [colors[i] for i in perm]

    # ——构建映射并存入 adata.uns——
    color_dict = {module_ids[i]: colors[i] for i in range(n_all)}
    if isinstance(mod_col_val, str):
        adata.uns.setdefault(mod_col_val, {}).update(color_dict)
    else:
        raise ValueError(f"Invalid module color value in adata.uns['{mod_col_val}']: {mod_col_val}")


# %%
# 生成配色方案
assign_module_colors(adata, ggm_key='ggm', seed=1)  

# %%
# 原始注释
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="annotation", show=True,
              save="/annotation_raw.pdf")

# %%
# 旧版本
start_time = time.time()
module_used = adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cell_identity_module']['module_id'].tolist()
sg.integrate_annotations_noweight(
                    adata,
                    ggm_key='ggm',
                    result_anno='annotation_old_all',)
sg.integrate_annotations_noweight(
                    adata,
                    ggm_key='ggm',
                    result_anno='annotation_old_all_no_neighbor',
                    neighbor_similarity_ratio=0)
sg.integrate_annotations_noweight(
                    adata,
                    ggm_key='ggm',
                    modules_used=module_used,
                    result_anno='annotation_old_id',)
sg.integrate_annotations_noweight(
                    adata,
                    ggm_key='ggm',
                    modules_used=module_used,
                    result_anno='annotation_old_id_no_neighbor',
                    neighbor_similarity_ratio=0)
print(f"Time: {time.time() - start_time:.5f} s")


# %%
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_old_all", save="/annotation_old.pdf")
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_old_all_no_neighbor", save="/annotation_old_no_neighbor.pdf")
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_old_id", save="/annotation_old_id.pdf")
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_old_id_no_neighbor", save="/annotation_old_id_no_neighbor.pdf")

# %%
# 方案1，加速 KDE 计算，整体计算逻辑依然较慢
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KernelDensity
from scipy.stats import rankdata
import networkx as nx
import leidenalg
from igraph import Graph
import sys

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
    （与原版完全一致，仅修复 dens_log 计算中的 n_jobs 错误）
    """
    # reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

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

    # load annotations, compute ranks
    anno = {}
    rank_norm = {}
    dens_log = {}

    # 加速 KDE 的设置
    max_kde = 2000
    sigma_d = float(np.std(dist[:, 1:])) + 1e-8

    def compute_density(pts):
        """子采样 + KDE 计算（不使用 n_jobs）"""
        if pts.shape[0] > max_kde:
            idx_sub = np.random.choice(pts.shape[0], max_kde, replace=False)
            pts = pts[idx_sub]
        kde = KernelDensity(
            bandwidth=sigma_d,
            kernel='gaussian',
            algorithm='ball_tree',
            leaf_size=40,
            metric='euclidean'
        )
        kde.fit(pts)
        return -kde.score_samples(X)

    for m in modules_used:
        col = f"{m}_anno_smooth" if f"{m}_anno_smooth" in adata.obs else f"{m}_anno"
        exp_col = f"{m}_exp"
        if col not in adata.obs or exp_col not in adata.obs:
            raise KeyError(f"Missing columns for module {m}")

        # binary annotation vector
        lab = (adata.obs[col] == m).astype(int).values
        anno[m] = lab

        # expression rank normalization
        r = rankdata(-adata.obs[exp_col].values, method="dense")
        rank_norm[m] = (r - 1) / (n_cells - 1)

        # accelerated density log
        pts = X[lab == 1]
        if len(pts) >= 3:
            dens_log[m] = compute_density(pts)
        else:
            dens_log[m] = np.zeros(n_cells, dtype="float32")

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
    rng = random.Random(random_state)

    for t in range(1, max_iter + 1):
        # purity_adjustment
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

            if rng.random() < p_t:
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


# %%
# 测试1，全部模块
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
                    random_state=0)
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_new_all", save="/annotation_new_all.pdf")

# %%
# 测试2， identify_modules
start_time = time.time()
module_used = adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cell_identity_module']['module_id'].tolist()
integrate_annotations(
                    adata,
                    ggm_key='ggm',
                    modules_used=module_used,
                    #modules_excluded=['M15', 'M18'],        
                    #modules_preferred=['M28', 'M38'],
                    result_anno='annotation_new_id',
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
                    random_state=0)
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'], 
              color="annotation_new_id", save="/annotation_new_id.pdf")




# %%
# 测试3， identify_modules, 不迭代优化
start_time = time.time()
module_used = adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cell_identity_module']['module_id'].tolist()
integrate_annotations(
                    adata,
                    ggm_key='ggm',
                    modules_used=module_used,
                    #modules_excluded=['M15', 'M18'],        
                    #modules_preferred=['M28', 'M38'],
                    result_anno='annotation_new_id_no_optimization',
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
                    max_iter=0,
                    random_state=0)
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_new_id_no_optimization", save="/annotation_new_id_no_optimization.pdf")



# %%
# 计算各组结果的ARI和NMI
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Define the columns of interest
clust_columns = ['annotation', 'annotation_old_all', 'annotation_old_all_no_neighbor', 'annotation_old_id', 'annotation_old_id_no_neighbor', 
                 'annotation_new_all', 'annotation_new_id', 'annotation_new_id_no_optimization']
# Extract clustering labels from adata.obs
df = adata.obs[clust_columns]
# Initialize empty DataFrames for the ARI and NMI matrices
ari_matrix = pd.DataFrame(np.zeros((len(clust_columns), len(clust_columns))), index=clust_columns, columns=clust_columns)
nmi_matrix = pd.DataFrame(np.zeros((len(clust_columns), len(clust_columns))), index=clust_columns, columns=clust_columns)
# Loop over each pair of clustering columns and compute metrics.
for col1 in clust_columns:
    for col2 in clust_columns:
        # Convert to strings to ensure proper handling (especially if the columns are categorical)
        labels1 = df[col1].astype(str).values
        labels2 = df[col2].astype(str).values
        
        # Calculate ARI and NMI
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)
        
        ari_matrix.loc[col1, col2] = ari
        nmi_matrix.loc[col1, col2] = nmi
# Display the ARI matrix
print("Adjusted Rand Index (ARI) Matrix:")
print(ari_matrix)
# Display the NMI matrix
print("\nNormalized Mutual Information (NMI) Matrix:")
print(nmi_matrix)
















# %%
# 方案2，尝试GPU加速：
import random, sys
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.stats import rankdata
import networkx as nx, leidenalg
from igraph import Graph

def integrate_annotations_gpu(
    adata,
    ggm_key="ggm",
    modules_used=None,
    modules_excluded=None,
    modules_preferred=None,
    result_anno="anno_density",
    embedding_key="spatial",
    k_neighbors=24,
    purity_adjustment=True,
    alpha=0.4, beta=0.3, gamma=0.3, delta=0.4,
    lambda_pair=0.3,
    lr=0.1, target_purity=0.8, w_floor=0.1, w_ceil=1.5,
    max_iter=100, energy_tol=1e-3,
    p0=0.1, tau=8.0, random_state=None,
    dens_k=24
):
    """
    原始 Potts-CRF + ICM + 模拟退火 算法实现，
    仅在密度计算时用 KDTree 近邻均距近似 KDE，
    其余逻辑逐细胞循环完全一致。
    """
    # reproducibility
    rnd = random.Random(random_state)
    np.random.seed(random_state)

    # sanity checks
    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not in adata.obsm")
    if ggm_key not in adata.uns["ggm_keys"]:
        raise KeyError(f"{ggm_key} not in adata.uns['ggm_keys']")
    stats_key = adata.uns["ggm_keys"][ggm_key]["module_stats"]

    # module list
    all_modules = pd.unique(adata.uns[stats_key]["module_id"]).tolist()
    modules_used = modules_used or all_modules
    if modules_excluded:
        modules_used = [m for m in modules_used if m not in modules_excluded]
    if modules_preferred:
        for m in modules_preferred:
            if m not in modules_used:
                modules_used.append(m)
    M = len(modules_used)
    if M == 0:
        raise ValueError("No modules to integrate")

    # spatial KNN graph (exclude self)
    X = adata.obsm[embedding_key].astype(np.float32)
    n_cells = adata.n_obs
    knn = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X)
    dist_full, idx_full = knn.kneighbors(X)
    dist = dist_full[:, 1:]    # (N, k_neighbors)
    idx  = idx_full[:, 1:]     # (N, k_neighbors)
    sigma = max(np.mean(dist), 1e-6)
    W = np.exp(-(dist**2) / sigma**2).astype(np.float32)    # (N, k)
    W_sum = W.sum(axis=1)                                   # (N,)
    lam = (lambda_pair * W).astype(np.float32)              # (N, k)

    # precompute rank_norm and anno and dens_log
    anno = {}
    rank_norm = {}
    dens_log = {}

    for m in modules_used:
        # annotation vector
        col_smooth = f"{m}_anno_smooth"
        col_raw    = f"{m}_anno"
        col = col_smooth if col_smooth in adata.obs else col_raw
        if col not in adata.obs or f"{m}_exp" not in adata.obs:
            raise KeyError(f"Missing columns for module {m}")
        lab = (adata.obs[col] == m).astype(int).values   # (N,)
        anno[m] = lab

        # expression rank normalization
        r = rankdata(-adata.obs[f"{m}_exp"].values, method="dense")
        rank_norm[m] = (r - 1) / (n_cells - 1)

        # dens_log via KDTree
        pts = X[lab==1]
        if pts.shape[0] >= dens_k:
            tree = KDTree(pts, leaf_size=32)
            dists, _ = tree.query(X, k=dens_k)
            dens_log[m] = dists.mean(axis=1).astype(np.float32)
        else:
            dens_log[m] = np.zeros(n_cells, dtype=np.float32)

    # dynamic module weights
    w_mod = {m: 1.0 for m in modules_used}

    # unary energy
    def unary(i, m):
        nb = idx[i]    # (k_neighbors,)
        vote = (W[i] * anno[m][nb]).sum() / (W_sum[i] + 1e-12)
        vote *= w_mod[m]
        return (alpha * (1 - vote)
                + beta  * rank_norm[m][i]
                + gamma * dens_log[m][i]
                + delta * (1 - w_mod[m]))

    # initial labeling
    rng = random.Random(random_state)
    label = np.empty(n_cells, dtype=object)
    for i in range(n_cells):
        candidates = [m for m in modules_used if anno[m][i]] or modules_used.copy()
        if modules_preferred:
            pref = [m for m in candidates if m in modules_preferred]
            candidates = pref or candidates
        label[i] = min(candidates, key=lambda m: unary(i, m))

    # energy function
    def energy():
        U = sum(unary(i, label[i]) for i in range(n_cells))
        diffs = (label[idx] != label[:, None]).astype(np.int8)
        P = (lam * diffs).sum()
        return U + float(P)

    prev_E = energy()
    print(f"Iteration   0, Energy: {prev_E:.2f}")

    # main loop: ICM + simulated annealing
    for t in range(1, max_iter+1):
        changed = 0
        p_t = p0 * np.exp(-t / tau)

        # purity adjustment
        if purity_adjustment:
            g = nx.Graph()
            edges_src = np.repeat(np.arange(n_cells), k_neighbors)
            edges_dst = idx.reshape(-1)
            g.add_edges_from(zip(edges_src, edges_dst))
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
                        purity[m].append(v/len(cluster)*cp)
            for m in modules_used:
                if purity[m]:
                    mp = float(np.mean(purity[m]))
                    w_mod[m] = np.clip(w_mod[m] * (1 + lr*(mp-target_purity)),
                                       w_floor, w_ceil)

        # cell-wise update
        for i in range(n_cells):
            candidates = [m for m in modules_used if anno[m][i]] or modules_used.copy()
            if modules_preferred:
                pref = [m for m in candidates if m in modules_preferred]
                candidates = pref or candidates
            if label[i] not in candidates:
                candidates.append(label[i])

            if rng.random() < p_t:
                new_label = rng.choice(candidates)
            else:
                # compute pairwise costs correctly
                neigh_labels = label[idx[i]]         # (k_neighbors,)
                diff = np.array([neigh_labels != m for m in candidates], dtype=np.float32)  # (C,k)
                pair_costs = lam[i][None, :] * diff                                           
                totals = [unary(i, m) + pair_costs[j].sum()
                          for j, m in enumerate(candidates)]
                new_label = candidates[int(np.argmin(totals))]

            if new_label != label[i]:
                label[i] = new_label
                changed += 1

        curr_E = energy()
        print(f"Iteration: {t:3d}, Energy: {curr_E:.2f}, ΔE: {prev_E-curr_E:+.2f}, Changed: {changed}")
        if abs(prev_E - curr_E)/max(abs(prev_E),1.0) < energy_tol:
            print("Converged")
            break
        prev_E = curr_E

    adata.obs[result_anno] = label
    return adata



# %%
# 测试
start_time = time.time()
module_used = adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cell_identity_module']['module_id'].tolist()
integrate_annotations_gpu(
                    adata,
                    ggm_key='ggm',
                    modules_used=module_used,
                    #modules_excluded=['M15', 'M18'],        
                    #modules_preferred=['M28', 'M38'],
                    result_anno='annotation_gpu_id',
                    k_neighbors=24,
                    lambda_pair=0.3,
                    purity_adjustment=True,
                    w_floor=0.01,
                    w_ceil=1.5,
                    lr=0.1,
                    target_purity=0.85,
                    # alpha=0.5,
                    # beta=0.3
                    gamma=0.3,
                    # delta=0.4,   
                    max_iter=3,
                    random_state=0,
                    #device="cuda:0"
                    )
print(f"Time: {time.time() - start_time:.5f} s")
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_gpu_id", save="/annotation_gpu_id.pdf")


# %%
for i in range(10):
    assign_module_colors(adata, ggm_key='ggm', seed=i) 
    sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, show=True, palette=adata.uns['module_colors'],
              color="annotation_old_all_no_neighbor", save=f"/annotation_old_all_no_neighbor_test_{i}.pdf")
# %%




# %%
# 保存adata
adata.write("data/MOSTA_E16.5_E1S1_ggm_anno.h5ad")



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