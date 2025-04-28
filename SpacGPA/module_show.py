import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.colors import ListedColormap
from typing import List
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from matplotlib.lines import Line2D


# get_module_edges 
def get_module_edges(self, module_id):
    """
    Extract edges within a module.
    Parameters:
        module_id: The ID of the module to extract.
    Returns:
        module_edges: The edges within the module.
    """
    module_list = self.modules['module_id'].unique()
    if module_id not in module_list:
        raise ValueError("Module ID not found.")
    if self.modules is None:
        raise ValueError("Please run find_modules first.")
    genes_in_module = self.modules.loc[self.modules["module_id"] == module_id, "gene"].unique()

    mask = self.SigEdges["GeneA"].isin(genes_in_module) & self.SigEdges["GeneB"].isin(genes_in_module)
    module_edges = self.SigEdges[mask].copy()
    module_edges.index = range(len(module_edges))
    module_edges.insert(0, "module_id", module_id)
    
    return module_edges

# get_module_anno 
def get_module_anno(self, module_id, add_enrich_info=True, top_n=None, term_id=None):
    """
    Get the annotation information of a specific module.
    Parameters:
        self: GGM object
        module_id: str, a module id in the modules_summary
        add_enrich_info: bool, whether to add GO and MP enrichment information to the module annotation
        top_n: int, the top n GO or MP terms to add to the module annotation, default as None
            use when add_enrich_info is True and too many GO or MP terms are enriched in the module
        term_id: a list of GO or MP term ids to add to the module annotation, default as None
            use for specific GO or MP terms to add to the module annotation
    """
    if term_id is not None and top_n is not None:
        raise ValueError("term_id and top_n cannot be specified at the same time.")
    
    if self.modules is None:
        raise ValueError("No modules found. run find_modules function first.")
    
    if module_id not in self.modules['module_id'].values:
        raise ValueError(f"{module_id} not found in modules.")

    module_anno = self.modules[self.modules['module_id'] == module_id].copy()
    if add_enrich_info:
        if self.go_enrichment is not None:
            go_df = self.go_enrichment[self.go_enrichment['module_id'] == module_id].copy()
            if go_df.empty:
                print(f"No significant enrichment GO term found for {module_id}.")
            else:
                if top_n is not None:
                    go_df = go_df.head(top_n)
                for _, row in go_df.iterrows():
                    go_id = row['go_id']
                    go_term = row['go_term']
                    gene_list = row['genes_with_go_in_module'].split("/")
                    module_anno[go_id] = module_anno['gene'].apply(lambda g: go_term if g in gene_list else None)

        if self.mp_enrichment is not None:
            mp_df = self.mp_enrichment[self.mp_enrichment['module_id'] == module_id].copy()
            if mp_df.empty:
                print(f"No significant enrichment MP term found for {module_id}.")
            else:
                if top_n is not None:
                    mp_df = mp_df.head(top_n)
                for _, row in mp_df.iterrows():
                    mp_id = row['mp_id']
                    mp_term = row['mp_term']
                    gene_list = row['genes_with_mp_in_module'].split("/")
                    module_anno[mp_id] = module_anno['gene'].apply(lambda g: mp_term if g in gene_list else None)
    if term_id is not None:
        save_id = np.concatenate((go_df['go_id'].values, mp_df['mp_id'].values))
        remove_id = [x for x in save_id if x not in term_id]
        keep_id = [x for x in term_id if x in save_id]
        wrong_id = [x for x in term_id if x not in save_id]
        if len(keep_id) == 0:
            print(f"Make sure the term_id listed are in the enrichment information.")
        else:    
            if len(wrong_id) > 0:
                print(f"The term_id {wrong_id} not in the enrichment information.")
            module_anno = module_anno.drop(columns=remove_id)  

    return module_anno


# calculating_module_similarity 
def calculating_module_similarity(
    adata,
    ggm_key='ggm',
    modules_used=None,
    modules_excluded=None,
    use_smooth=True,
    corr_method='pearson',
    linkage_method='average',
    return_summary=True,
    plot_heatmap=True,
    show_full_heatmap = True,
    heatmap_metric='correlation',
    fig_height=15,
    fig_width=16,
    dendrogram_height=0.15,
    dendrogram_space=0.08,
    axis_fontsize=12,
    axis_labelsize=15,
    legend_fontsize=12,
    legend_labelsize=15,
    cmap_name='bwr',
    save_plot_as=None
):
    """
    Compute pairwise module similarity and optionally visualize as a clustered heatmap.

    This function calculates correlation and Jaccard index between modules based on
    expression profiles and annotations stored in an AnnData object. It can produce
    either a half-matrix (upper triangle only) or a full-matrix heatmap with
    hierarchical clustering dendrograms.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing module expression in .obsm and annotations in .obs.
    ggm_key : str, default 'ggm'
        Key to access module stats and expression arrays in adata.uns['ggm_keys'].
    modules_used : list of str or None
        Module IDs to include; all modules if None.
    modules_excluded : list of str or None
        Module IDs to exclude; none if None.
    use_smooth : bool, default True
        Use smoothed annotations if available.
    corr_method : {'pearson','spearman','kendall'}, default 'pearson'
        Correlation method for expression profiles.
    linkage_method : str, default 'average'
        Linkage method for hierarchical clustering.
    return_summary : bool, default True
        Return DataFrame of module-pair metrics if True.
    plot_heatmap : bool, default True
        Display the heatmap and dendrogram plot if True.
    show_full_heatmap : bool, default True
        Plot full symmetric matrix if True; otherwise upper triangle only.
    heatmap_metric : {'correlation','jaccard'}, default 'correlation'
        Metric to visualize in the heatmap.
    fig_height, fig_width : float
        Figure dimensions in inches.
    dendrogram_height : float
        Fraction of figure allocated to each dendrogram.
    dendrogram_space : float
        Spacing between dendrogram and heatmap axes.
    axis_fontsize, axis_labelsize : int
        Font sizes for tick labels and axis labels.
    legend_fontsize, legend_labelsize : int
        Font sizes for colorbar and legend text.
    cmap_name : str
        Name of diverging colormap.
    save_plot_as : str or None
        File path to save the figure; supports '.pdf' or '.png'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['module_a','module_b','correlation','jaccard_index']
        if return_summary is True and plotting is disabled; otherwise None.
    """
    # validate colormap and metric
    allowed_cmaps = [
        'PiYG','PRGn','BrBG','PuOr','RdGy','RdBu',
        'RdYlBu','RdYlGn','Spectral','coolwarm','bwr','seismic'
    ] + [c + '_r' for c in [
        'PiYG','PRGn','BrBG','PuOr','RdGy','RdBu',
        'RdYlBu','RdYlGn','Spectral','coolwarm','bwr','seismic'
    ]]
    if cmap_name not in allowed_cmaps:
        raise ValueError(f"cmap_name must be one of {allowed_cmaps}")
    if heatmap_metric not in ['correlation','jaccard']:
        raise ValueError("heatmap_metric must be 'correlation' or 'jaccard'")

    # prepare module expression matrix
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} missing in adata.uns['ggm_keys']")
    stat_key = ggm_keys[ggm_key]['module_stats']
    expr_key = ggm_keys[ggm_key]['module_expression']
    stats_df = adata.uns[stat_key]
    module_expr = pd.DataFrame(
        adata.obsm[expr_key], index=adata.obs_names,
        columns=stats_df['module_id']
    )

    # filter modules
    modules = list(module_expr.columns) if modules_used is None else [
        m for m in modules_used if m in module_expr
    ]
    if modules_excluded:
        modules = [m for m in modules if m not in modules_excluded]
    if len(modules) < 2:
        raise ValueError("At least two modules are required")
    module_expr = module_expr[modules]

    # compute correlation matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_df = module_expr.astype(float).corr(method=corr_method)
    corr_mat = corr_df.values

    # compute Jaccard index matrix
    if use_smooth:
        anno_dict = {
            m: (adata.obs.get(f"{m}_anno_smooth", adata.obs[f"{m}_anno"]) == m).astype(int).values
            for m in modules
        }
    else:
        anno_dict = {
            m: (adata.obs[f"{m}_anno"] == m).astype(int).values
            for m in modules
        }
    records = []
    jacc_mat = pd.DataFrame(np.nan, index=modules, columns=modules)
    for i, j in itertools.combinations(range(len(modules)), 2):
        a = anno_dict[modules[i]]
        b = anno_dict[modules[j]]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        jacc = inter / union if union > 0 else np.nan
        records.append({
            'module_a': modules[i],
            'module_b': modules[j],
            'correlation': float(corr_mat[i, j]),
            'jaccard_index': jacc
        })
        jacc_mat.iat[i, j] = jacc_mat.iat[j, i] = jacc
    result_df = pd.DataFrame(records)

    # return summary if no plotting
    if not plot_heatmap and save_plot_as is None:
        return result_df if return_summary else None

    # hierarchical clustering
    Z = linkage(squareform(1 - corr_mat, checks=False), method=linkage_method)
    ordered = [modules[i] for i in leaves_list(Z)]
    n = len(ordered)

    # select heatmap data
    if heatmap_metric == 'correlation':
        data_df = corr_df.loc[ordered, ordered]
        vmin, vmax, cbar_label = np.nanmin(data_df.values), 1, 'Correlation'
    else:
        data_df = jacc_mat.loc[ordered, ordered].copy()
        np.fill_diagonal(data_df.values, 1.0)
        vmin, vmax, cbar_label = 0, 1, 'Jaccard index'
    mask = None if show_full_heatmap else np.tril(np.ones_like(data_df, bool), -1)

    # prepare colormap slice
    full_cmap = plt.get_cmap(cmap_name, 200)
    sub_cmap = ListedColormap(
        full_cmap(np.arange(200))[int(((vmin+1)/2 if heatmap_metric=='correlation' and vmin<0 else 0.5)*199):],
        name=f'{cmap_name}_slice'
    )

    # plotting
    if not show_full_heatmap:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(2, 1,
                              height_ratios=(dendrogram_height, 1-dendrogram_height),
                              hspace=dendrogram_space)
        ax_dend = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0])

        dendrogram(Z, ax=ax_dend, labels=ordered,
                   orientation='top', no_labels=True,
                   link_color_func=lambda *a, **k: 'black')
        ax_dend.axis('off')

        sns.heatmap(data_df.values, mask=mask, cmap=sub_cmap,
                    vmin=vmin, vmax=vmax,
                    xticklabels=ordered, yticklabels=ordered,
                    square=False, cbar=False, ax=ax_heat)

        # adjust axes for half-matrix
        ax_heat.xaxis.tick_top()
        ax_heat.xaxis.set_label_position('top')
        ax_heat.yaxis.tick_right()
        ax_heat.yaxis.set_label_position('right')
        ticks = np.arange(n) + 0.5
        ax_heat.set_xticks(ticks)
        ax_heat.set_yticks(ticks)
        ax_heat.set_xticklabels(ordered, rotation=90, ha='center', fontsize=axis_fontsize)
        ax_heat.set_yticklabels(ordered, rotation=0, fontsize=axis_fontsize)
        ax_heat.set_ylabel('Module', fontsize=axis_labelsize)

        # colorbar for half-matrix
        cax = fig.add_axes([0.14, 0.14, 0.02, 0.2])
        sm = plt.cm.ScalarMappable(cmap=sub_cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(data_df.values)
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=legend_fontsize)
        cb.set_label(cbar_label, fontsize=legend_labelsize)

    else:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(2, 2,
                              width_ratios=(dendrogram_height, 1-dendrogram_height),
                              height_ratios=(dendrogram_height, 1-dendrogram_height),
                              wspace=dendrogram_space, hspace=0)
        ax_row = fig.add_subplot(gs[1, 0])
        ax_col = fig.add_subplot(gs[0, 1])
        ax_heat = fig.add_subplot(gs[1, 1])

        dendrogram(Z, ax=ax_col, labels=ordered,
                   orientation='top', no_labels=True,
                   link_color_func=lambda *a, **k: 'black')
        ax_col.axis('off')

        dendrogram(Z, ax=ax_row, labels=ordered,
                   orientation='left', no_labels=True,
                   link_color_func=lambda *a, **k: 'black')
        ax_row.axis('off')
        ax_row.invert_yaxis()

        sns.heatmap(data_df.values, mask=mask, cmap=sub_cmap,
                    vmin=vmin, vmax=vmax,
                    xticklabels=ordered, yticklabels=ordered,
                    square=False, cbar=False, ax=ax_heat)

        # adjust axes for full-matrix
        ticks = np.arange(n) + 0.5
        ax_heat.xaxis.tick_bottom()
        ax_heat.set_xticks(ticks)
        ax_heat.set_xticklabels(ordered, rotation=45,
                                ha='right', va='top',
                                fontsize=axis_fontsize,
                                rotation_mode='anchor')
        ax_heat.yaxis.tick_left()
        ax_heat.set_yticks(ticks)
        ax_heat.set_yticklabels(ordered, fontsize=axis_fontsize)
        ax_heat.tick_params(axis='both', pad=1)
        ax_heat.set_xlabel('Module', fontsize=axis_labelsize)
        ax_heat.set_ylabel('')

        # colorbar for full-matrix
        cax = fig.add_axes([0.92, 0.3, 0.02, 0.2])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax),
                                   cmap=sub_cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.set_ylabel(cbar_label, fontsize=legend_labelsize)
        cb.ax.tick_params(labelsize=legend_fontsize)

    # save figure if requested
    if save_plot_as:
        ext = save_plot_as.split('.')[-1]
        if ext not in ['pdf', 'png']:
            raise ValueError("save_plot_as must end with .pdf or .png")
        dpi = 300 if ext == 'png' else None
        fig.savefig(save_plot_as, dpi=dpi, bbox_inches='tight')

    if plot_heatmap:
        plt.show()

    return result_df if return_summary else None


# module_dot_plot
def module_dot_plot(
    adata,
    ggm_key: str = 'ggm',
    modules_used: List[str] = None,
    modules_excluded: List[str] = None,
    groupby: str = None,
    show_dendrogram: bool = True,
    dendrogram_height: float = 0.15,
    dendrogram_space: float = 0.1,
    scale: bool = False,
    corr_method: str = 'pearson',
    linkage_method: str = 'average',
    fig_height: float = 8,
    fig_width: float = 10,
    dot_max_size: float = 200,
    cmap: str = 'Reds',
    axis_fontsize: int = 12,
    axis_labelsize: int = 15,
    legend_fontsize: int = 12,
    legend_labelsize: int = 15,
    save_plot_as: str = None,
    return_df: bool = False
):
    """
    Generate a dot plot of module expression across defined cell groups or cluster with optional dendrograms.

    This function clusters cell groups and modules based on correlation of expression,
    computes mean expression and percentage of expressing cells per module-group pair,
    and visualizes the results. dot size indicates percentage expressing; color indicates mean expression.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression in .X, metadata in .obs,
        module statistics in .uns, and module expressions in .obsm.
    ggm_key : str
        Key in adata.uns['ggm_keys'] that references module stats and expression.
    modules_used : List[str], optional
        Modules to include. If None, all available modules are used.
    modules_excluded : List[str], optional
        Modules to exclude from analysis.
    groupby : str
        Column name in adata.obs for grouping cells.
    show_dendrogram : bool, default True
        Whether to display hierarchical clustering dendrograms.
    dendrogram_height : float, default 0.15
        Fraction of figure height allocated to each dendrogram.
    dendrogram_space : float, default 0.1
        Horizontal gap between the left dendrogram and the y-axis tick labels when
        show_dendrogram is True (0 → flush; larger → wider gap).    
    scale : bool, default False
        Scale mean expression values for each module to [0,1].
    corr_method : str, default 'pearson'
        Correlation method for distance computation.
    linkage_method : str, default 'average'
        Linkage method for hierarchical clustering.
    fig_height : float, default 10
        Figure height in inches.
    fig_width : float, default 8
        Figure width in inches.
    dot_max_size : float, default 200
        Maximum bubble size for 100% expression.
    cmap : str, default 'Reds'
        Colormap for mean expression.
    axis_fontsize : int, default 12
        Font size for axis ticks.
    axis_labelsize : int, default 15
        Font size for axis labels.
    legend_fontsize : int, default 12
        Font size for legend text.
    legend_labelsize : int, default 15
        Font size for legend title.
    save_plot_as : str, optional
        Path to save figure (formats: 'pdf', 'png').
    return_df : bool, default False
        If True, return DataFrame of computed metrics.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['group','module','mean_expr','pct_expr'] if return_df is True; else None.
    """
    # Check if groupby is provided
    if groupby is None:
        raise ValueError("groupby must be specified")
    if groupby not in adata.obs:
        raise ValueError(f"{groupby} not found in adata.obs")
    keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in keys:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    
    # 1) Cluster cell groups
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    grp_means = expr.groupby(adata.obs[groupby]).mean()
    grp_corr = grp_means.T.corr(method=corr_method)
    Zg = linkage(squareform(1 - grp_corr.values), method=linkage_method)
    ordered_groups = list(grp_corr.index[leaves_list(Zg)])

    # 2) Cluster modules
    stats_key = keys[ggm_key]['module_stats']
    expr_key = keys[ggm_key]['module_expression']
    stats_df = adata.uns[stats_key]
    mod_expr = pd.DataFrame(
        adata.obsm[expr_key], index=adata.obs_names,
        columns=stats_df['module_id']
    )
    modules = list(mod_expr.columns) if modules_used is None else [m for m in modules_used if m in mod_expr]
    if modules_excluded:
        modules = [m for m in modules if m not in modules_excluded]
    if len(modules) < 2:
        raise ValueError('At least two modules are required')
    mod_expr = mod_expr[modules]
    mod_corr = mod_expr.corr(method=corr_method)
    Zm = linkage(squareform(1 - mod_corr.values), method=linkage_method)
    ordered_modules = [modules[i] for i in leaves_list(Zm)]

    # 3) Compute mean expression and percent expressing
    trim_cols = [f"{m}_exp_trim" for m in ordered_modules]
    df_trim = adata.obs[[groupby] + trim_cols]
    grp = df_trim.groupby(groupby)
    mean_expr = grp[trim_cols].mean().loc[ordered_groups, trim_cols]
    pct_expr = (
        grp[trim_cols]
        .apply(lambda x: (x > 0).sum() / len(x) * 100)
        .loc[ordered_groups, trim_cols]
    )
    if scale:
        mn, mx = mean_expr.min(), mean_expr.max()
        rng = (mx - mn).replace(0, 1)
        mean_expr = (mean_expr - mn) / rng

    plot_df = pd.DataFrame({
        'group': np.repeat(ordered_groups, len(ordered_modules)),
        'module': ordered_modules * len(ordered_groups),
        'mean_expr': mean_expr.values.flatten(),
        'pct_expr': pct_expr.values.flatten(),
    })
    g2i = {g: i for i, g in enumerate(ordered_groups)}
    m2i = {m: i for i, m in enumerate(ordered_modules)}
    xs = plot_df['module'].map(m2i)
    ys = plot_df['group'].map(g2i)

    # 4) Create figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height))
    if show_dendrogram:
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=(dendrogram_height, 1 - dendrogram_height),
            height_ratios=(dendrogram_height, 1 - dendrogram_height),
            hspace=0, wspace=dendrogram_space
        )
        ax_row = fig.add_subplot(gs[1, 0])
        ax_col = fig.add_subplot(gs[0, 1])
        ax_sc = fig.add_subplot(gs[1, 1])
        dendrogram(Zg, ax=ax_row, labels=ordered_groups[::-1], orientation='left', no_labels=True,
                   link_color_func=lambda *args, **kwargs: 'black')
        ax_row.axis('off')
        dendrogram(Zm, ax=ax_col, labels=ordered_modules, orientation='top', no_labels=True,
                   link_color_func=lambda *args, **kwargs: 'black')
        ax_col.axis('off')
    else:
        ax_sc = fig.add_subplot(1, 1, 1)

    # 5) Plot bubbles
    sc = ax_sc.scatter(
        xs, ys,
        s=plot_df['pct_expr'] / 100 * dot_max_size,
        c=plot_df['mean_expr'], cmap=cmap, edgecolors='none'
    )
    ax_sc.set_xlim(-0.5, len(ordered_modules) - 0.5)
    ax_sc.set_ylim(-0.5, len(ordered_groups) - 0.5)
    ax_sc.set_xticks(range(len(ordered_modules)))
    ax_sc.set_xticklabels(ordered_modules, rotation=45, 
                          ha='right', va='top', rotation_mode='anchor', 
                          fontsize=axis_fontsize)
    ax_sc.set_yticks(range(len(ordered_groups)))
    ax_sc.set_yticklabels(ordered_groups, fontsize=axis_fontsize)
    ax_sc.tick_params(axis='both', pad=1) 
    ax_sc.set_xlabel('Module', fontsize=axis_labelsize)
    if not show_dendrogram:
        ax_sc.set_ylabel(groupby, fontsize=axis_labelsize)

    pos = ax_sc.get_position()          
    pad = 0.02                         
    cb_x = pos.x1 + pad
    lg_x = pos.x1 + pad

    # 6) Add colorbar
    cax = fig.add_axes([cb_x, 0.4, 0.02, 0.2])
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.ax.set_title('Average Expression', fontsize=legend_labelsize, pad=10, loc='left')
    cbar.ax.tick_params(labelsize=legend_fontsize)

    # 7) Add size legend
    lax = fig.add_axes([lg_x, 0.0, 0.1, 0.30])
    lax.axis('off')
    sizes = [25, 50, 75, 100]
    handles = [
        Line2D([0], [0], linestyle='', marker='o', color='black',
               markersize=np.sqrt(s / 100 * dot_max_size))
        for s in sizes
    ]
    labels = [f"{s}%" for s in sizes]
    lax.legend(handles, labels, loc='upper left', frameon=False,
               handletextpad=0.5, labelspacing=0.8, borderaxespad=0)
    lax.text(0, 1.0, 'Percent Expressed', transform=lax.transAxes,
             va='bottom', ha='left', fontsize=legend_labelsize)

    # 8) Save figure if requested
    if save_plot_as:
        fmt = save_plot_as.split('.')[-1]
        if fmt not in ['pdf', 'png']:
            raise ValueError("Supported formats are 'pdf' and 'png'.")
        save_kwargs = {'bbox_inches': 'tight'}
        if fmt == 'png':
            save_kwargs['dpi'] = 300
        fig.savefig(save_plot_as, format=fmt, **save_kwargs)

    if return_df:
        return plot_df
    return