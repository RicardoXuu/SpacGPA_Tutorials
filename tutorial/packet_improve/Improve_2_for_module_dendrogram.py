
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
import itertools

def draw_module_dendrogram(
    adata,
    ggm_key='ggm',
    corr_method='pearson',
    use_smooth=True,
    plot=True,
    fig_width=15,               # figure width in inches
    fig_height=15,              # figure height in inches
    linkage_method='average',
    dendrogram_height=0.2,      # fraction of figure height for the dendrogram
    tick_fontsize=8,            # fontsize for tick labels
    axis_labelsize=10,          # fontsize for axis titles
    cbar_ticksize=8,            # fontsize for colorbar tick labels
    cbar_labelsize=10           # fontsize for colorbar titles
):
    """
    Compute module–module Pearson/Spearman/Kendall correlation and Jaccard index,
    plot a heatmap (upper=correlation, lower=Jaccard) with a single black dendrogram on top,
    colorbars outside, and return a DataFrame of pairwise stats.
    """
    # --- Data prep ---
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} missing in adata.uns['ggm_keys']")
    stat_key = ggm_keys[ggm_key]['module_stats']
    expr_key = ggm_keys[ggm_key]['module_expression']

    stats_df = adata.uns[stat_key]
    module_expr = pd.DataFrame(
        adata.obsm[expr_key],
        index=adata.obs_names,
        columns=stats_df['module_id']
    )
    modules = list(module_expr.columns)

    # --- Correlation ---
    if corr_method not in ['pearson','spearman','kendall']:
        raise ValueError("corr_method must be 'pearson','spearman' or 'kendall'")
    module_expr = module_expr.astype('float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_df = module_expr.corr(method=corr_method)
    corr_mat = corr_df.values

    # --- Build annotation dict for Jaccard ---
    anno_dict = {}
    for mod in modules:
        col = f"{mod}_anno_smooth" if use_smooth and f"{mod}_anno_smooth" in adata.obs else f"{mod}_anno"
        anno_dict[mod] = (adata.obs[col] == mod).astype(int).values

    # --- Pairwise stats ---
    records = []
    for i,j in itertools.combinations(range(len(modules)), 2):
        m1, m2 = modules[i], modules[j]
        a, b = anno_dict[m1], anno_dict[m2]
        inter = np.logical_and(a,b).sum()
        union = np.logical_or(a,b).sum()
        jacc = inter / union if union > 0 else np.nan
        records.append({
            'module_a': m1,
            'module_b': m2,
            'correlation': float(corr_mat[i,j]),
            'jaccard_index': jacc
        })
    result_df = pd.DataFrame(records)
    result_df.sort_values(['module_a','correlation'], ascending=[True,False], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    if plot:
        # --- Clustering order ---
        dist = 1 - corr_mat
        Z = linkage(squareform(dist, checks=False), method=linkage_method)
        order = leaves_list(Z)
        ordered = [modules[i] for i in order]

        corr_ord = corr_df.loc[ordered, ordered].values
        jacc_mat = pd.DataFrame(np.nan, index=modules, columns=modules)
        for r in records:
            jacc_mat.loc[r['module_a'], r['module_b']] = r['jaccard_index']
            jacc_mat.loc[r['module_b'], r['module_a']] = r['jaccard_index']
        jacc_ord = jacc_mat.loc[ordered, ordered].values

        # --- Figure layout with top dendrogram + heatmap ---
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(2, 1,
                              height_ratios=(dendrogram_height, 1 - dendrogram_height),
                              hspace=0.0)
        ax_dend = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0])

        # Top dendrogram in black, no frame
        dendrogram(Z, ax=ax_dend, labels=ordered, orientation='top',
                   no_labels=True, color_threshold=None,
                   link_color_func=lambda *args, **kwargs: 'black')
        ax_dend.axis('off')

        # --- Heatmap: set negative correlations to zero ---
        corr_plot = corr_ord.copy()
        corr_plot[corr_plot < 0] = 0  # ← clip negative to zero

        mask_low = np.tril(np.ones_like(corr_plot, bool), -1)
        mask_up  = np.triu(np.ones_like(jacc_ord, bool), 1)

        cmap_corr = plt.get_cmap('bwr')
        max_val = np.nanmax(corr_plot)
        sns.heatmap(corr_plot, mask=mask_low,
                    cmap=cmap_corr, vmin=0, vmax=max_val,
                    xticklabels=ordered, yticklabels=ordered,
                    cbar=False, ax=ax_heat)

        cmap_jacc = sns.light_palette("navy", as_cmap=True)
        sns.heatmap(jacc_ord, mask=mask_up, cmap=cmap_jacc, vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False,
                    cbar=False, ax=ax_heat)

        # --- Ticks and labels ---
        n = len(ordered)
        ax_heat.set_xticks(np.arange(n) + 0.5)
        ax_heat.set_yticks(np.arange(n) + 0.5)
        ax_heat.set_xticklabels(ordered, rotation=45, ha='right', fontsize=tick_fontsize)
        ax_heat.set_yticklabels(ordered, rotation=0, fontsize=tick_fontsize)
        ax_heat.set_xlabel('Module', fontsize=axis_labelsize)
        ax_heat.set_ylabel('Module', fontsize=axis_labelsize)

        # --- Colorbars outside ---
        cb1 = fig.add_axes([0.92, 0.4, 0.015, 0.2])
        sm1 = plt.cm.ScalarMappable(cmap=cmap_corr,
                                    norm=plt.Normalize(vmin=0, vmax=max_val))
        cbar1 = fig.colorbar(sm1, cax=cb1)
        # ← replace the '0' tick label with '<0'
        ticks = cbar1.get_ticks()
        labels = [('<0' if t == 0 else f'{t:.2f}') for t in ticks]
        cbar1.ax.set_yticklabels(labels, fontsize=cbar_ticksize)
        cbar1.set_label('Correlation', fontsize=cbar_labelsize)

        cb2 = fig.add_axes([0.92, 0.15, 0.015, 0.2])
        sm2 = plt.cm.ScalarMappable(cmap=cmap_jacc,
                                    norm=plt.Normalize(vmin=0, vmax=1))
        cbar2 = fig.colorbar(sm2, cax=cb2)
        cbar2.ax.tick_params(labelsize=cbar_ticksize)
        cbar2.set_label('Jaccard index', fontsize=cbar_labelsize)

        plt.show()

    return result_df


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
                                dendrogram_height=0.1,
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