
# %%
# 一些问题修复
# 使用 CytAssist_FreshFrozen_Mouse_Brain_Rep2 数据
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

# %%
# 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
#from SpacGPA import *
import SpacGPA as sg

# %%
# 读取 ggm
start_time = time.time()
ggm = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm.h5")
print(f"Read ggm: {time.time() - start_time:.5f} s")
# 读取联合分析的ggm
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
print("=====================================")
print(ggm)
print("=====================================")
print(ggm_mulit_intersection)
print("=====================================")
print(ggm_mulit_union)


# %%
# 读取数据
adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                       count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

# %%
# 读取原始数据集提供的两种注释
graph_cluster = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_graphclust/clusters.csv',
                            header=0, sep=',', index_col=0)
kmeans_10_clusters = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv',
                                header=0, sep=',', index_col=0)

adata.obs['graph_cluster'] = graph_cluster.loc[adata.obs_names, 'Cluster']
adata.obs['graph_cluster'] = adata.obs['graph_cluster'].astype('category')

adata.obs['kmeans_10_clusters'] = kmeans_10_clusters.loc[adata.obs_names, 'Cluster']
adata.obs['kmeans_10_clusters'] = adata.obs['kmeans_10_clusters'].astype('category')

# %%
#使用leiden聚类和louvain聚类基于模块表达矩阵归一化矩阵进行聚类
start_time = time.time()
sc.pp.neighbors(adata, n_neighbors=18, use_rep='module_expression_scaled',n_pcs=adata.obsm['module_expression_scaled'].shape[1])
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_ggm')
sc.tl.leiden(adata, resolution=1, key_added='leiden_1_ggm')
sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_ggm')
sc.tl.louvain(adata, resolution=1, key_added='louvan_1_ggm')
print(f"Time: {time.time() - start_time:.5f} s")


# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="graph_cluster", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="kmeans_10_clusters", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="leiden_0.5_ggm", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="leiden_1_ggm", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="louvan_0.5_ggm", show=True)
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="louvan_1_ggm", show=True)

# %%
ggm.adjust_cutoff(pcor_threshold=0.059)
best_inf, _ = sg.find_best_inflation(ggm, min_inflation=1.1, phase=3, show_plot=True)
ggm.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=best_inf, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm.modules_summary.shape)

# %%
sg.annotate_with_ggm(adata, ggm,
                     ggm_key='ggm')

# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        result_anno='annotation',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=18,
                        neighbor_similarity_ratio=0.0,
                        )
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)


# %%
sg.classify_modules(adata, 
                 ggm_key='ggm',
                 ref_anno='graph_cluster',
                 #ref_cluster_method='leiden', 
                 #ref_cluster_method='none',  
                 #ref_cluster_resolution=0.5, 
                 skew_threshold=2,
                 top1pct_threshold=2,
                 Moran_I_threshold=0.3,
                 min_dominant_cluster_fraction=0,
                 anno_overlap_threshold=0.4)

# %%
adata.uns['module_filtering']['type_tag'].value_counts()

# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag'] == 'cell_identity_module']['module_id'].values,
                        result_anno='annotation_1',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=18,
                        neighbor_similarity_ratio=0.0,
                        )
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_1", show=True)

# %%
# 计算ARI和NMI, 并用矩阵的格式打印结果
adata.obs['annotation_re'] = adata.obs['annotation'].astype('str')
adata.obs['annotation_re'].fillna("Others", inplace=True)
adata.obs['annotation_re'] = adata.obs['annotation_re'].astype('category')

adata.obs['annotation_1_re'] = adata.obs['annotation_1'].astype('str')
adata.obs['annotation_1_re'].fillna("Others", inplace=True)
adata.obs['annotation_1_re'] = adata.obs['annotation_1_re'].astype('category')

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
columns = ['graph_cluster', 'kmeans_10_clusters', 'leiden_1_ggm',  'louvan_1_ggm', 
           'annotation_re','annotation_1_re']
rows = ['graph_cluster', 'kmeans_10_clusters', 'leiden_1_ggm',  'louvan_1_ggm',
        'annotation_re','annotation_1_re']
df_ari = pd.DataFrame(index=rows, columns=columns)
df_nmi = pd.DataFrame(index=rows, columns=columns)
for i in range(len(rows)):
    for j in range(len(columns)):
        if rows[i] == columns[j]:
            df_ari.iloc[i, j] = 1
            df_nmi.iloc[i, j] = 1
        else:
            df_ari.iloc[i, j] = adjusted_rand_score(adata.obs[rows[i]], adata.obs[columns[j]])
            df_nmi.iloc[i, j] = normalized_mutual_info_score(adata.obs[rows[i]], adata.obs[columns[j]])

# %%
print(df_ari)
print(df_nmi)


# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id']
pdf_file = "figures/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_all_modules_Anno.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, size=1.6, alpha_img=0.5, frameon = False, color_map="Reds", 
                  color=[f"{module}_exp",f"{module}_exp_trim",f"{module}_anno",f"{module}_anno_smooth"],show=False)
    show_png_file = f"figures/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_{module}_Anno.png"
    plt.savefig(show_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(show_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    