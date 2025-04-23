
# %%
# 设计函数分析模块之间的相关性以及模块在leiden聚类上的表达分布
# 使用人类肺癌 Xenium Human_Lung_Cancer_FFPE_5K 数据集
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
ggm = sg.load_ggm("data/Human_Lung_Cancer_5K.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取处理后的adata对象
adata = sc.read_h5ad("data/Human_Lung_Cancer_5K_ggm_anno.h5ad")


# %%
# 模块相关性分析
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.transforms import Affine2D
import itertools

def draw_module_dendrogram(
    adata,
    ggm_key='ggm',
    corr_method='pearson',
    use_smooth=True,
    plot=True,
    fig_width=8,
    fig_height=8,
    linkage_method='average',
    tick_fontsize=8,
    axis_labelsize=10,
    cbar_ticksize=8,
    cbar_labelsize=10
):
    # --- Data prep ---
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} missing")
    stat_key = ggm_keys[ggm_key]['module_stats']
    expr_key = ggm_keys[ggm_key]['module_expression']
    stats_df = adata.uns[stat_key]
    module_expr = pd.DataFrame(
        adata.obsm[expr_key],
        index=adata.obs_names,
        columns=stats_df['module_id']
    )
    modules = list(module_expr.columns)

    # --- Compute correlation ---
    if corr_method not in ['pearson','spearman','kendall']:
        raise ValueError("corr_method must be 'pearson','spearman' or 'kendall'")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_df = module_expr.corr(method=corr_method)
    corr = corr_df.values
    n = corr.shape[0]

    # --- Mask out lower triangle ---
    corr_ut = np.triu(corr, k=1)
    corr_ut[corr_ut < 0] = 0  # enforce non-negative if desired

    # --- Hierarchical clustering order ---
    dist = 1 - corr
    Z = linkage(squareform(dist, checks=False), method=linkage_method)
    leaves = leaves_list(Z)
    ordered = [modules[i] for i in leaves]
    corr_ord = corr_df.loc[ordered, ordered].values
    corr_ut_ord = np.triu(corr_ord, k=1)

    # --- Figure & axes ---
    if not plot:
        return pd.DataFrame({
            'module_a':[], 'module_b':[], 
            'correlation':[], 'jaccard_index':[]
        })
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0,0,1,1])
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Heatmap rotated -45° ---
    # build a square image of size n×n
    im = ax.imshow(
        corr_ut_ord,
        cmap='Reds', 
        vmin=0, vmax=np.nanmax(corr_ut_ord),
        interpolation='none',
        origin='lower',
        extent=[0, n, 0, n]
    )
    # apply rotation about the center
    trans = (Affine2D()
             .translate(-n/2, -n/2)
             .rotate_deg(-45)
             .translate(n/2, n/2)
             + ax.transData)
    im.set_transform(trans)

    # --- Dendrogram along new diagonal ---
    # create a second invisible axis for the tree
    ax2 = fig.add_axes([0,0,1,1], sharex=ax, sharey=ax)
    ax2.axis('off')
    dendrogram(
        Z,
        orientation='right',
        labels=ordered,
        no_labels=True,
        link_color_func=lambda *args, **kwargs: 'black',
        ax=ax2
    )
    # rotate that axis the same way
    for line in ax2.get_lines():
        line.set_transform(trans)

    # --- Colorbar ---
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=cbar_ticksize)
    cb.set_label('Correlation', fontsize=cbar_labelsize)

    # --- Done ---
    plt.show()

    # --- Return empty DataFrame as stats no longer relevant here ---
    return pd.DataFrame({
        'module_a':[], 'module_b':[], 
        'correlation':[], 'jaccard_index':[]
    })



# %%
# 测试
mod_cor = draw_module_dendrogram(adata,
                                ggm_key='ggm',
                                use_smooth=True,
                                corr_method='pearson',
                                plot=True,
                                linkage_method='average',
                                fig_height=16,
                                fig_width=15,
                                #dendrogram_height=0.1,
                                tick_fontsize=15,
                                axis_labelsize=15,
                                cbar_ticksize=12,
                                cbar_labelsize=15,
                                )


# %%
mod_cor['correlation'].describe()


# %%



# %% 
# 参考版本
# 版本1，实现了基本功能
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import itertools

def draw_module_dendrogram(
    adata,
    ggm_key='ggm',
    corr_method='pearson',    # 'pearson', 'spearman', or 'kendall'
    use_smooth=True,
    plot=True,
    figsize=(15, 15),
    linkage_method='average'
):
    """
    Compute pairwise module–module correlation (by expression) and Jaccard index (by annotation),
    optionally plot a clustered heatmap of the correlations, and return a summary table.
    """
    # 1. retrieve keys
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    stat_key = ggm_keys[ggm_key]['module_stats']
    expr_key = ggm_keys[ggm_key]['module_expression']

    # 2. fetch expression + modules
    if expr_key not in adata.obsm:
        raise ValueError(f"{expr_key} not found in adata.obsm")
    stats_df = adata.uns.get(stat_key)
    module_expr = pd.DataFrame(
        adata.obsm[expr_key],
        index=adata.obs_names,
        columns=stats_df['module_id'].values
    )

    # 3. compute correlation, cast to float64 and silence overflow warnings
    print(f"Calculating Correlation of {len(module_expr.columns)} modules...")
    if corr_method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError(f"Invalid correlation method: {corr_method}. Choose from 'pearson', 'spearman', or 'kendall'.")
    module_expr = module_expr.astype('float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_df = module_expr.corr(method=corr_method)

    corr_mat = corr_df.values
    modules = list(corr_df.columns)

    # 4. build annotation dict for Jaccard
    anno_dict = {}
    for mod in modules:
        anno_col = f"{mod}_anno_smooth" if use_smooth and f"{mod}_anno_smooth" in adata.obs else f"{mod}_anno"
        if anno_col not in adata.obs:
            raise ValueError(f"Annotation column not found for module {mod}")
        anno_dict[mod] = (adata.obs[anno_col] == mod).astype(int).values

    # 5. compute pairwise Jaccard + correlation records
    print(f"Calculating Jaccard index of {len(modules)} modules...")
    records = []
    for i, j in itertools.combinations(range(len(modules)), 2):
        m1, m2 = modules[i], modules[j]
        a, b = anno_dict[m1], anno_dict[m2]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        jaccard = inter / union if union > 0 else np.nan
        records.append({
            'module_a': m1,
            'module_b': m2,
            'correlation': float(corr_mat[i, j]),
            'jaccard_index': jaccard
        })

    result_df = pd.DataFrame(records)
    result_df.sort_values(['module_a', 'correlation'], ascending=[True, False], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # 6. optional clustered heatmap
    if plot:
        print("Plotting heatmap of module correlation...")
        dist = 1 - corr_mat
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method=linkage_method)

        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        g = sns.clustermap(
            corr_df,
            row_linkage=Z, col_linkage=Z,
            cmap=cmap, center=0,
            figsize=figsize,
            dendrogram_ratio=0.1,
            cbar_pos=(0.02, 0.2, 0.03, 0.4)
        )
        # rotate x‑ticks 45°
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        g.ax_heatmap.set_xlabel('Module')
        g.ax_heatmap.set_ylabel('Module')
        plt.show()
        
    return result_df
# %%
