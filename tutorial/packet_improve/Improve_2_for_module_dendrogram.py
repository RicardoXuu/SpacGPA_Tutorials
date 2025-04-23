
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
from matplotlib.colors import ListedColormap
import itertools

def calculating_module_similarity(
    adata,
    ggm_key='ggm',
    use_smooth=True,
    corr_method='pearson',
    linkage_method='average',
    return_summary=True,
    plot_heatmap=True,
    heatmap_metric='correlation',   # 'correlation' or 'jaccard'
    fig_height=17,
    fig_width=15,
    dendrogram_height=0.15,
    dendrogram_hspace=0.1,
    axis_fontsize=12,
    axis_labelsize=15,
    legend_fontsize=12,
    legend_labelsize=15,
    cmap_name='bwr',               # must be one of the 12 diverging maps
    save_as=None
):
    """
    Compute module–module Pearson/Spearman/Kendall correlation and Jaccard index,
    then plot either the upper‐triangle heatmap of correlation or of Jaccard index
    with a dendrogram above.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with module expression and annotations.
    ggm_key : str
        Key in adata.uns['ggm_keys'].
    use_smooth : bool
        Whether to use smoothed annotation columns.
    corr_method : {'pearson','spearman','kendall'}
        Correlation method.
    linkage_method : str
        Hierarchical clustering linkage.
    plot : bool
        Whether to draw the figure.
    fig_height, fig_width : float
        Figure dimensions in inches.
    dendrogram_height : float
        Fraction of figure height for the dendrogram.
    dendrogram_hspace : float
        Vertical space between dendrogram and heatmap.
    axis_fontsize : int
        Tick label font size.
    axis_labelsize : int
        Axis label font size.
    legend_fontsize : int
        Colorbar tick label font size.
    legend_labelsize : int
        Colorbar title font size.
    cmap_name : str
        Diverging colormap name (one of the 12 allowed).
    heatmap_metric : str
        Which matrix to plot: 'correlation' or 'jaccard'.

    Returns
    -------
    pd.DataFrame
        Columns: module_a, module_b, correlation, jaccard_index.
    """
    # --- Validate colormap and metric choice ---
    allowed_cmaps = [
        'PiYG','PRGn','BrBG','PuOr','RdGy','RdBu',
        'RdYlBu','RdYlGn','Spectral','coolwarm','bwr','seismic',
        'PiYG_r','PRGn_r','BrBG_r','PuOr_r','RdGy_r','RdBu_r',
        'RdYlBu_r','RdYlGn_r','Spectral_r','coolwarm_r','bwr_r','seismic_r'
    ]
    if cmap_name not in allowed_cmaps:
        raise ValueError(f"cmap_name must be one of {allowed_cmaps}")
    if heatmap_metric not in ['correlation','jaccard']:
        raise ValueError("heatmap_metric must be 'correlation' or 'jaccard'")

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

    # --- Pairwise stats & Jaccard matrix ---
    if use_smooth:
        anno_dict = {
            mod: (adata.obs.get(f"{mod}_anno_smooth", adata.obs[f"{mod}_anno"]) == mod).astype(int).values
            for mod in modules
        }
    else:
        anno_dict = {
            mod: (adata.obs[f"{mod}_anno"] == mod).astype(int).values
            for mod in modules
        }
    records = []
    jacc_mat = pd.DataFrame(np.nan, index=modules, columns=modules)
    for i, j in itertools.combinations(range(len(modules)), 2):
        a, b = anno_dict[modules[i]], anno_dict[modules[j]]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        jacc = inter / union if union > 0 else np.nan
        records.append({
            'module_a': modules[i],
            'module_b': modules[j],
            'correlation': float(corr_mat[i, j]),
            'jaccard_index': jacc
        })
        jacc_mat.iloc[i, j] = jacc_mat.iloc[j, i] = jacc
    result_df = pd.DataFrame(records).sort_values(
        by=['module_a','correlation'],
        ascending=[True, False]
    ).reset_index(drop=True)

    # --- Plotting ---
    if plot_heatmap == False and save_as is None:
        if return_summary:
            return result_df
        else:
            return
    else:
        # Hierarchical clustering
        dist = 1 - corr_mat
        Z = linkage(squareform(dist, checks=False), method=linkage_method)
        order = leaves_list(Z)
        ordered = [modules[k] for k in order]

        # Prepare data matrix
        if heatmap_metric == 'correlation':
            data_df = corr_df.loc[ordered, ordered]
            vmin, vmax = np.nanmin(data_df.values), 1
            cbar_label = 'Correlation'
        else:  # jaccard
            data_df = jacc_mat.loc[ordered, ordered]
            vmin, vmax = 0, 1
            cbar_label = 'Jaccard index'

        # Mask lower triangle
        mask = np.tril(np.ones(data_df.shape, bool), -1)

        # Build colormap slice
        full_cmap = plt.get_cmap(cmap_name, 200)
        colors = full_cmap(np.arange(200))
        # if heatmap_metric == 'correlation':
        #     min_val = vmin
        #     frac = (min_val + 1) / 2  if min_val < 0 else 0.5
        #     idx = int(np.floor(frac * 199))
        # else:
        #     idx = 0
        # sub_cmap = ListedColormap(colors[idx:], name=f'{cmap_name}_slice')

        colors = full_cmap(np.arange(200))
        min_val = vmin
        if min_val < 0:
            frac = (min_val + 1) / 2
            idx = int(np.floor(frac * 199))
        else:
            idx = 100
        sub_cmap = ListedColormap(colors[idx:], name=f'{cmap_name}_slice')

        # Figure + gridspec
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(2, 1,
                                height_ratios=(dendrogram_height, 1 - dendrogram_height),
                                hspace=dendrogram_hspace)
        ax_dend = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0])

        # Dendrogram
        dendrogram(Z, ax=ax_dend, labels=ordered, orientation='top',
                    no_labels=True, link_color_func=lambda *args, **kwargs: 'black')
        ax_dend.axis('off')

        # Heatmap
        sns.heatmap(
            data_df.values,
            mask=mask,
            cmap=sub_cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=ordered,
            yticklabels=ordered,
            square=False,
            cbar=False,
            ax=ax_heat
        )

        # Axis adjustments
        ax_heat.xaxis.tick_top()
        ax_heat.xaxis.set_label_position('top')
        ax_heat.yaxis.tick_right()
        ax_heat.yaxis.set_label_position('right')
        n = len(ordered)
        ax_heat.set_xticks(np.arange(n) + 0.5)
        ax_heat.set_yticks(np.arange(n) + 0.5)
        ax_heat.set_xticklabels(ordered, rotation=90, ha='center', fontsize=axis_fontsize)
        ax_heat.set_yticklabels(ordered, rotation=0, fontsize=axis_fontsize)
        ax_heat.set_ylabel('Module', fontsize=axis_labelsize)

        # Colorbar
        cax = fig.add_axes([0.14, 0.14, 0.02, 0.2])
        sm = plt.cm.ScalarMappable(cmap=sub_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(data_df.values)
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=legend_fontsize)
        cb.set_label(cbar_label, fontsize=legend_labelsize)

        # --- Save to file(s) if requested ---
        if save_as is not None:
            format = save_as.split('.')[-1]
            if format not in ['pdf', 'png']:
                raise ValueError("only 'pdf' and 'png' formats are supported for saving.")
            if format == 'pdf':
                fig.savefig(save_as, format=format, bbox_inches='tight')
            if format == 'png':
                fig.savefig(save_as, format=format, bbox_inches='tight', dpi=300)    
        if plot_heatmap:
            plt.show()
        if return_summary:
            return result_df
        else:
            return


# %%
# 测试
mod_cor = calculating_module_similarity(adata,
                                        ggm_key='ggm',
                                        use_smooth=True,
                                        corr_method='pearson',
                                        linkage_method='average',
                                        return_summary=True,
                                        plot_heatmap=True,
                                        heatmap_metric='jaccard',   # 'correlation' or 'jaccard'
                                        fig_height=17,
                                        fig_width=15,
                                        dendrogram_height=0.15,
                                        dendrogram_hspace=0.08,
                                        axis_fontsize=12,
                                        axis_labelsize=15,
                                        legend_fontsize=12,
                                        legend_labelsize=15,
                                        cmap_name='bwr',               # must be one of the 24 diverging maps
                                        save_as="figures/module_dendrogram.png"  # or "figures/module_dendrogram.png"
                                        )


# %%
mod_cor


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
# 测试
mod_cor = draw_module_dendrogram(adata,
                                ggm_key='ggm',
                                corr_method='pearson',    # 'pearson', 'spearman', or 'kendall'
                                use_smooth=True,
                                plot=True,
                                figsize=(15, 15),
                                linkage_method='average'
                                )
                                 
# %%
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns

def draw_module_dendrogram(adata,
                          expression_key = 'module_expression',
                          figsize=(15,15)
                          ):
    # Assuming adata is your AnnData object with module_expression in obsm
    module_expression = adata.obsm[expression_key]

    n_samples = module_expression.shape[1]
    sample_names = [f'M{i+1}' for i in range(n_samples)]

    # 1. Compute correlation matrix
    corr_matrix = np.corrcoef(module_expression, rowvar=False)  # Correlation between columns (modules)

    # 2. Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - corr_matrix

    # 3. Perform hierarchical clustering
    # Convert to condensed form (required by linkage)
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')  # You can try 'ward', 'complete', etc.

    # 4. Plot dendrogram
    #plt.figure(figsize=(10, 5))
    #dendrogram(Z, labels=sample_names)
    #plt.title('Dendrogram of Module Expression')
    #plt.xlabel('Modules')
    #plt.ylabel('Distance (1 - correlation)')
    #plt.show()

    # 5. Create 2D dendrogram-heatmap
    # Create a DataFrame for the heatmap
    df_corr = pd.DataFrame(corr_matrix, index=sample_names,columns=sample_names)

    # Plot clustered heatmap
    g = sns.clustermap(df_corr, 
                       row_linkage=Z, 
                       col_linkage=Z,
                       cmap='coolwarm', 
                       center=0,
                       figsize=figsize,
                       dendrogram_ratio=0.1)
    g.ax_heatmap.set_xlabel('Module')
    g.ax_heatmap.set_ylabel('Module')
    #g.ax_heatmap.set_title('Module Expression Correlation Heatmap with Dendrogram')
    plt.show()

# %%
draw_module_dendrogram(adata,
                       expression_key='module_expression')  
# %%
