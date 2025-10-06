
# %%
# Analyze MOSTA E16.5 E2S1 Spatial Transcriptomics Data using SpacGPA
# Data source: https://db.cngb.org/stomics/mosta/
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

# %%
# import SpacGPA
import SpacGPA as sg

# %%
# Set working directory
workdir = '..'
os.chdir(workdir)

# %%
# Load Spatial Transcriptomics data
adata = sc.read_h5ad("data/Stereo-seq/MOSTA/E16.5_E2S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata)

# %%
# Data preprocessing
adata.X = adata.layers['count']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
print(adata.X.shape)

# %%
# Calculate Coexpression Network using SpacGPA
ggm = sg.create_ggm(adata,project_name = "E16.5_E2S1")  

# %%
# show significant co-expression gene pairs
print(ggm.SigEdges.head(5))

# %%
# identify gene programs via MCL-Hub algorithm
ggm.find_modules(methods='mcl-hub')

# %%
# show identified programs
print(ggm.modules_summary.head(5))

# %%
# visualize network of program M1 via sg.module_network_plot
M1_edges = ggm.get_module_edges('M1')

# %%
M1_anno = ggm.get_module_anno('M1')

# %%
sg.module_network_plot(
    nodes_edges = M1_edges,
    nodes_anno = M1_anno
) 
# %%

# %%
# Show Relationship between degree and Moran's I of genes within program M1 via sg.module_degree_vs_moran_plot



# %%
# GO Enrichment Analysis
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
# MP Enrichment Analysis
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)


# %%
# visualize GO enrichment results for programs M1-M5 via sg.module_go_enrichment_plot

# %%
# visualize MP enrichment results for programs M1-M5 via sg.module_mp_enrichment_plot



# %%
# Save GGM object as h5 file
sg.save_ggm(ggm, "data/MOSTA_E16.5_E1S1_r3.ggm.h5")

# %%
# Load GGM object
ggm = sg.load_ggm("data/MOSTA_E16.5_E1S1_r3.ggm.h5")


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
# 删除GGM对象，释放内存
del ggm
gc.collect()

# %%
# 使用leiden聚类和louvain聚类基于模块表达矩阵归一化矩阵进行聚类
start_time = time.time()
sc.pp.neighbors(adata, n_neighbors=18, use_rep='module_expression_scaled',n_pcs=adata.obsm['module_expression_scaled'].shape[1])
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_ggm')
sc.tl.leiden(adata, resolution=1, key_added='leiden_1_ggm')
sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_ggm')
sc.tl.louvain(adata, resolution=1, key_added='louvan_1_ggm')
print(f"Time: {time.time() - start_time:.5f} s")


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
adata.uns['module_stats'].to_csv("data/MOSTA_E16.5_E1S1_ggm_module_stats_r3.csv")

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                      ggm_key='ggm',
                      embedding_key='spatial',
                      k_neighbors=24,
                      min_drop_neighbors=1,
                      min_add_neighbors='half',
                      max_weight_ratio=1.5,
                      border_iqr_factor=1.5,
                      )
print(f"Time: {time.time() - start_time:.5f} s")    


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
                    Moran_I_threshold=0.1,
                    min_dominant_cluster_fraction=0.2,
                    anno_overlap_threshold=0.4)

# %%
adata.uns['module_filtering']['type_tag'].value_counts()


# %%
# 计算并可视化模块之间的相似性
mod_cor = sg.module_similarity_plot(adata,
                                    ggm_key='ggm',
                                    use_smooth=True,
                                    corr_method='pearson',
                                    linkage_method='average',
                                    return_summary=True,
                                    plot_heatmap=True,
                                    heatmap_metric='correlation',   
                                    fig_height=26,
                                    fig_width=28,
                                    dendrogram_height=0.1,
                                    dendrogram_space=0.05,
                                    axis_fontsize=12,
                                    axis_labelsize=15,
                                    legend_fontsize=12,
                                    legend_labelsize=15,
                                    cmap_name='coolwarm',
                                    save_plot_as="figures/MOSTA_E16.5_E1S1_module_corr_similarity_r3.pdf",  
                                    )


# %%
# 可视化模块在各个leiden分群里的表达气泡图
sg.module_dot_plot(
    adata,
    ggm_key='ggm',
    groupby= 'leiden_0.5_ggm', 
    scale = True,
    corr_method='pearson',
    linkage_method='average',
    show_dendrogram = True,
    dendrogram_height = 0.1,
    dendrogram_space=0.03,
    fig_height = 8,
    fig_width = 25,
    dot_max_size=200,
    cmap='Reds',
    axis_labelsize=12,
    axis_fontsize=10,
    return_df=False,
    save_plot_as="figures/MOSTA_E16.5_E1S1_leiden_0_5_ggm_module_dotplot_r3.pdf",
)

# %%
# 可视化模块在各个原始注释分群里的表达气泡图
sg.module_dot_plot(
    adata,
    ggm_key='ggm',
    groupby= 'annotation', 
    scale = True,
    corr_method='pearson',
    linkage_method='average',
    show_dendrogram = True,
    dendrogram_height = 0.1,
    dendrogram_space=0.18,
    fig_height = 10,
    fig_width = 25,
    dot_max_size=200,
    cmap='Reds',
    axis_labelsize=12,
    axis_fontsize=10,
    return_df=False,
    save_plot_as="figures/MOSTA_E16.5_E1S1_raw_cell_type_module_dotplot_r3.pdf",
)



# %%
# 合并注释（考虑空间坐标和模块表达值）
# 1. 使用全部模块
sg.integrate_annotations_noweight(adata,
                        ggm_key='ggm',
                        result_anno='ggm_annotation',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )
# 2. 使用经过鉴别的模块
sg.integrate_annotations_noweight(adata,
                        ggm_key='ggm',
                        modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity']== True]['module_id'],
                        result_anno='ggm_annotation_filtered',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )
# 3. 不使用activity模块
sg.integrate_annotations_noweight(adata,
                        ggm_key='ggm',
                        modules_excluded=adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cellular_activity_module']['module_id'],
                        result_anno='ggm_annotation_no_activity',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )

# %%
# 合并注释（不考虑空间坐标）
sg.integrate_annotations_noweight(adata,
                        ggm_key='ggm',
                        result_anno='ggm_annotation_no_spatial',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
# 2. 使用经过鉴别的模块
sg.integrate_annotations_noweight(adata,
                        ggm_key='ggm',
                        modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity']== True]['module_id'],
                        result_anno='ggm_annotation_filtered_no_spatial',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
# 3. 不使用activity模块
sg.integrate_annotations_noweight(adata,
                        ggm_key='ggm',
                        modules_excluded=adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cellular_activity_module']['module_id'],
                        result_anno='ggm_annotation_no_activity_no_spatial',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )


# %%
# 保存注释结果
adata.obs.to_csv("data/MOSTA_E16.5_E1S1_ggm_annotation_r3.csv")


# %%
# 保存adata
adata.write("data/MOSTA_E16.5_E1S1_ggm_anno_r3.h5ad")




# %%
module_used = adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cell_identity_module']['module_id'].tolist()

# %%
# 测试新版整合函数
start_time = time.time()
sg.integrate_annotations(
    adata,
    ggm_key='ggm',
    modules_used=module_used,
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
# 测试新版函数
start_time = time.time()
sg.integrate_annotations(
    adata,
    ggm_key='ggm',
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
