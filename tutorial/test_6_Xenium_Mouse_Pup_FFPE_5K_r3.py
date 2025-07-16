# %%
# r2版本，使用新的基因选择方案，当2500<基因数<10000时，每次循环采样1/5的基因。
# 使用SpacGPA对 Xenium Mouse_Pup_5K 数据集进行分析
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
# 读取空转数据
adata = sc.read_10x_h5('/dta/ypxu/ST_GGM/Raw_Datasets/Xenium/Mouse_Pup_5K/cell_feature_matrix.h5')
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
meta = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/Xenium/Mouse_Pup_5K/cells.csv.gz')
adata.obs = meta
adata.obsm['spatial'] = adata.obs[['x_centroid','y_centroid']].values*[-1,-1]

sc.pp.log1p(adata)
print(adata.X.shape)

sc.pp.filter_cells(adata, min_genes=100)
print(adata.X.shape)

sc.pp.filter_genes(adata,min_cells=10)
print(adata.X.shape)

# %%
# 使用 GPU 计算GGM，double_precision=False
ggm = sg.create_ggm(adata,
                    project_name = "Mouse_Pup_5K", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.05,
                    auto_adjust=True,
                    )  
print(ggm.SigEdges)

# %%
# 调整Pcor阈值
cut_pcor = ggm.fdr.summary[ggm.fdr.summary['FDR'] <= 0.05]['Pcor'].min()
if cut_pcor < 0.02:
    cut_pcor = 0.02
print("Adjust cutoff pcor:", cut_pcor)
ggm.adjust_cutoff(pcor_threshold=cut_pcor)

# %%
# 使用改进的mcl聚类识别共表达模块
start_time = time.time()
ggm.find_modules(methods='mcl-hub',
                 expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                 min_module_size=10, topology_filtering=True, 
                 convert_to_symbols=True, species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.modules_summary)


# %%
# GO富集分析
start_time = time.time()
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.go_enrichment)

# %%
# MP富集分析
start_time = time.time()
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.mp_enrichment)

# %%
# 打印GGM信息
ggm

# %%
# 保存GGM
start_time = time.time()
sg.save_ggm(ggm, "data/Mouse_Pup_5K_r3.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/Mouse_Pup_5K_r3.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
ggm

# %%
ggm.modules_summary.to_csv("data/Mouse_Pup_5K_ggm_modules_summary_r3.csv")


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
sc.pp.neighbors(adata, 
                use_rep='module_expression_scaled',
                n_pcs=adata.obsm['module_expression_scaled'].shape[1])
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
adata.uns['module_stats'].to_csv("data/Mouse_Pup_5K_ggm_module_stats_r3.csv")

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
                    ref_anno='leiden_0.5_ggm',
                    #ref_cluster_method='leiden',
                    #ref_cluster_resolution=0.5,
                    skew_threshold=2,
                    top1pct_threshold=2,
                    Moran_I_threshold=0.1,
                    min_dominant_cluster_fraction=0.3,
                    anno_overlap_threshold=0.5)
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
                                    heatmap_metric='correlation',   # 'correlation' or 'jaccard'
                                    fig_height=20,
                                    fig_width=21,
                                    dendrogram_height=0.1,
                                    dendrogram_space=0.06,
                                    axis_fontsize=12,
                                    axis_labelsize=15,
                                    legend_fontsize=12,
                                    legend_labelsize=15,
                                    cmap_name='coolwarm',               # must be one of the 24 diverging maps
                                    save_plot_as="figures/Mouse_Pup_5K_module_corr_similarity_r3.pdf"  
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
    dendrogram_space= 0.05,
    fig_height = 12,
    fig_width = 24,
    dot_max_size=300,
    cmap='Reds',
    axis_labelsize=12,
    axis_fontsize=10,
    return_df=False,
    save_plot_as="figures/Mouse_Pup_5K_leiden_0_5_ggm_module_dotplot_r3.pdf" 
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
adata.obs.to_csv("data/Mouse_Pup_5K_ggm_annotation_r3.csv")

# %%
# 保存adata
adata.write("data/Mouse_Pup_5K_ggm_anno_r3.h5ad")


# %%
