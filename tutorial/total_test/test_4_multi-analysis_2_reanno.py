# %%
# 尝试根据联合分析的结果优化注释，使用CytAssist_FreshFrozen_Mouse_Brain_Rep2数据集
import numpy as np
import pandas as pd
import random
import time
import torch
import scanpy as sc
import anndata
import os
import gc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# %%
# 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
from ST_GGM_dev_1 import *

# %%
# 读取数据
adata_combined = sc.read_h5ad('data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_Combined_Anno.h5ad')
adata_self = sc.read_h5ad('data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_Self_Anno.h5ad')


# %%
# 使用module表达矩阵代替PCA进行降维和聚类
sc.pp.neighbors(adata_combined, use_rep='module_expression_scaled', n_neighbors=18)
sc.tl.umap(adata_combined)
sc.tl.louvain(adata_combined, key_added='combined_exp_louvain', resolution=0.5)
sc.tl.leiden(adata_combined, key_added='combined_exp_leiden', resolution=0.5)

sc.pp.neighbors(adata_self, use_rep='module_expression_scaled', n_neighbors=18)
sc.tl.umap(adata_self)
sc.tl.louvain(adata_self, key_added='self_exp_louvain', resolution=0.5)
sc.tl.leiden(adata_self, key_added='self_exp_leiden', resolution=0.5)

# %%
# 读取原始数据集提供的两种注释
graph_cluster = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_graphclust/clusters.csv',
                            header=0, sep=',', index_col=0)
kmeans_10_clusters = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv',
                                header=0, sep=',', index_col=0)

adata_self.obs['graph_cluster'] = graph_cluster.loc[adata_self.obs_names, 'Cluster']
adata_self.obs['graph_cluster'] = adata_self.obs['graph_cluster'].astype('category')

adata_self.obs['kmeans_10_clusters'] = kmeans_10_clusters.loc[adata_self.obs_names, 'Cluster']
adata_self.obs['kmeans_10_clusters'] = adata_self.obs['kmeans_10_clusters'].astype('category')

adata_combined.obs['graph_cluster'] = graph_cluster.loc[adata_combined.obs_names, 'Cluster']
adata_combined.obs['graph_cluster'] = adata_combined.obs['graph_cluster'].astype('category')

adata_combined.obs['kmeans_10_clusters'] = kmeans_10_clusters.loc[adata_combined.obs_names, 'Cluster']
adata_combined.obs['kmeans_10_clusters'] = adata_combined.obs['kmeans_10_clusters'].astype('category')


# %%
# 可视化原数据集提供的注释
# 1. Cluster 配色方案（14个类别）
cluster_colors = {
    1: "#1b9e77",   # Cluster 1
    2: "#d95f02",   # Cluster 2
    3: "#7570b3",   # Cluster 3
    4: "#e7298a",   # Cluster 4
    5: "#66a61e",   # Cluster 5
    6: "#e6ab02",   # Cluster 6
    7: "#a6761d",   # Cluster 7
    8: "#666666",   # Cluster 8
    9: "#8dd3c7",   # Cluster 9
    10: "#ffffb3",  # Cluster 10
    11: "#bebada",  # Cluster 11
    12: "#fb8072",  # Cluster 12
    13: "#80b1d3",  # Cluster 13
    14: "#fdb462"   # Cluster 14
}

sc.pl.spatial(adata_self, alpha_img = 0.5, size = 1.6, title= "graph_cluster", frameon = False, color="graph_cluster", palette=cluster_colors,show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_graph_cluster.pdf")
sc.pl.spatial(adata_self, alpha_img = 0.5, size = 1.6, title= "kmeans_10_clusters", frameon = False, color="kmeans_10_clusters",show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_kmeans_10_clusters.pdf")

# %%
# 可视化单独切片分析的注释
sc.pl.spatial(adata_self, alpha_img = 0.5, size = 1.6, title= "self_exp_louvain", frameon = False, color="self_exp_louvain", show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_self_exp_louvain.pdf")
sc.pl.spatial(adata_self, alpha_img = 0.5, size = 1.6, title= "self_exp_leiden", frameon = False, color="self_exp_leiden", show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_self_exp_leiden.pdf")
sc.pl.spatial(adata_self, alpha_img = 0.5, size = 1.6, title= "annotation_self", frameon = False, color="annotation_self", show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_self_annotation.pdf")

# %%
# 可视化联合分析的注释
sc.pl.spatial(adata_combined, alpha_img = 0.5, size = 1.6, title= "combined_exp_louvain", frameon = False, color="combined_exp_louvain", show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_combined_exp_louvain.pdf")
sc.pl.spatial(adata_combined, alpha_img = 0.5, size = 1.6, title= "combined_exp_leiden", frameon = False, color="combined_exp_leiden", show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_combined_exp_leiden.pdf")
sc.pl.spatial(adata_combined, alpha_img = 0.5, size = 1.6, title= "annotation_combined", frameon = False, color="annotation_combined", show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_combined_annotation.pdf")

# %%
# 可视化模块
for module in adata_combined.uns['module_stats']['module_id'].unique()[55:]:
    print(module)
    sc.pl.spatial(adata_combined, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"], color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True,
                  save=f"/{module}_of_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Combined_Anno.pdf")


# %%
# 计算模块重叠度
overlap_records = calculate_module_overlap(adata_combined, 
                                           module_list = adata_combined.uns['module_stats']['module_id'].unique())

# %%
# 查找最相似的模块
# 筛选出所有模块注释列，假设模块列以 "M" 开头且包含 "anno"
module_cols = [col for col in adata_combined.obs.columns if col.startswith("M") and "anno_smooth" in col]

# 保存结果：字典，键为 cluster，值为列表（每项为 (模块列名, 相似度)），排序后取前三
similarity_results = {}

# 遍历 graph_cluster 中的所有类别（假设为类别变量）
for cluster in adata_combined.obs['graph_cluster'].cat.categories:
    # 构造当前 cluster 的布尔 mask
    cluster_mask = adata_combined.obs['graph_cluster'] == cluster
    
    sim_list = []
    # 遍历每个模块注释列
    for mod in module_cols:
        # 模块注释中值为1表示被注释的样本
        module_mask = adata_combined.obs[mod] == 1
        
        # 计算交集和并集
        intersection = np.logical_and(cluster_mask, module_mask).sum()
        union = np.logical_or(cluster_mask, module_mask).sum()
        jaccard = intersection / union if union > 0 else 0
        
        sim_list.append((mod, jaccard))
    
    # 按相似度降序排序，并取前三
    sim_list_sorted = sorted(sim_list, key=lambda x: x[1], reverse=True)[:3]
    similarity_results[cluster] = sim_list_sorted

# 打印结果
for cl, sim_list in similarity_results.items():
    print(f"Cluster {cl}:")
    for mod, sim in sim_list:
        print(f"  模块 {mod}, Jaccard 相似度: {sim:.3f}")  

##########
# Cluster 1:
#   模块 M069_anno_smooth, Jaccard 相似度: 0.314
#   模块 M065_anno_smooth, Jaccard 相似度: 0.228
#   模块 M080_anno_smooth, Jaccard 相似度: 0.195 ****
# ----------------------------------------
# Cluster 2:
#   模块 M051_anno_smooth, Jaccard 相似度: 0.453 
#   模块 M031_anno_smooth, Jaccard 相似度: 0.397 ****
#   模块 M038_anno_smooth, Jaccard 相似度: 0.342 ****
# ----------------------------------------
# Cluster 3:
#   模块 M027_anno_smooth, Jaccard 相似度: 0.768 ****
#   模块 M028_anno_smooth, Jaccard 相似度: 0.719
#   模块 M024_anno_smooth, Jaccard 相似度: 0.548
# ----------------------------------------
# Cluster 4:
#   模块 M037_anno_smooth, Jaccard 相似度: 0.688
#   模块 M018_anno_smooth, Jaccard 相似度: 0.679 ****
#   模块 M010_anno_smooth, Jaccard 相似度: 0.646
# ----------------------------------------
# Cluster 5:
#   模块 M108_anno_smooth, Jaccard 相似度: 0.427 ****
#   模块 M093_anno_smooth, Jaccard 相似度: 0.266
#   模块 M090_anno_smooth, Jaccard 相似度: 0.185
# ----------------------------------------
# Cluster 6:
#   模块 M061_anno_smooth, Jaccard 相似度: 0.754 ****
#   模块 M088_anno_smooth, Jaccard 相似度: 0.584
#   模块 M036_anno_smooth, Jaccard 相似度: 0.335
# ----------------------------------------
# Cluster 7:
#   模块 M007_anno_smooth, Jaccard 相似度: 0.408 ****
#   模块 M025_anno_smooth, Jaccard 相似度: 0.331
#   模块 M073_anno_smooth, Jaccard 相似度: 0.261
# ----------------------------------------
# Cluster 8:
#   模块 M106_anno_smooth, Jaccard 相似度: 0.688 ****
#   模块 M074_anno_smooth, Jaccard 相似度: 0.313
#   模块 M104_anno_smooth, Jaccard 相似度: 0.287 ****
# ----------------------------------------
# Cluster 9:
#   模块 M110_anno_smooth, Jaccard 相似度: 0.583
#   模块 M058_anno_smooth, Jaccard 相似度: 0.530 ****
#   模块 M055_anno_smooth, Jaccard 相似度: 0.325
# ----------------------------------------
# Cluster 10:
#   模块 M032_anno_smooth, Jaccard 相似度: 0.444
#   模块 M010_anno_smooth, Jaccard 相似度: 0.302 ****
#   模块 M111_anno_smooth, Jaccard 相似度: 0.290
# ----------------------------------------
# Cluster 11:
#   模块 M074_anno_smooth, Jaccard 相似度: 0.398 ****
#   模块 M059_anno_smooth, Jaccard 相似度: 0.262
#   模块 M104_anno_smooth, Jaccard 相似度: 0.232
# ----------------------------------------
# Cluster 12:
#   模块 M013_anno_smooth, Jaccard 相似度: 0.772 ****
#   模块 M029_anno_smooth, Jaccard 相似度: 0.374
#   模块 M094_anno_smooth, Jaccard 相似度: 0.120
# ----------------------------------------
# Cluster 13:
#   模块 M041_anno_smooth, Jaccard 相似度: 0.505
#   模块 M031_anno_smooth, Jaccard 相似度: 0.321 ****
#   模块 M051_anno_smooth, Jaccard 相似度: 0.275
# ----------------------------------------
# Cluster 14:
#   模块 M059_anno_smooth, Jaccard 相似度: 0.114 ****
#   模块 M025_anno_smooth, Jaccard 相似度: 0.070
#   模块 M104_anno_smooth, Jaccard 相似度: 0.059
# ----------------------------------------

# %%
mapping_module_combined = ['M056','M089','M074','M061','M106','M108','M013','M027','M030','M039',
                           'M094','M073','M020','M031','M007','M094','M058','M005','M098',
                           'M059','M038','M010','M080'] # 其他备选的modules 'M069', 'M005', 'M059', 'M031', 'M058','M110', 'M038','M041'
# 其中'M038','M010','M080'因为其表达广泛，作为最底层模块出现，不列入keep_module_combined
keep_module_combined = ['M056','M089','M074','M061','M106','M108','M013','M027','M030','M039',
                        'M094','M073','M020','M031','M007','M094','M058','M005','M098']

# %%
smooth_annotations(adata_combined,
                    embedding_key='spatial',
                    k_neighbors=18,
                    min_annotated_neighbors=2
                    )
adata_combined.obs.drop(columns='M056_anno_smooth', inplace=True) # 删除表皮模块的smooth注释
integrate_annotations(adata_combined,
                      module_list = mapping_module_combined,
                      keep_modules = keep_module_combined,
                      result_anno='annotation_combined_mapping',
                      embedding_key='spatial',
                      k_neighbors=18,
                      use_smooth=True,
                      neighbor_majority_frac=0.90
                      )

# %%
# 2. Module 配色方案
adata_combined.obs['plot'] = adata_combined.obs['annotation_combined_mapping'].copy()
#adata_combined.obs.loc[adata_combined.obs['plot'] == 'None', 'plot'] = None

# 说明：对于与某个 Cluster 唯一映射的模块，直接采用对应 Cluster 的颜色；
# 对于同一 Cluster 存在多个模块（如 Cluster 2 的 M031 与 M038），
# 这里将 M031 用 Cluster 原色，M038 用一个对比色（#fdae61）。
module_colors = {
    "M056": "#a6cee3",  # 未映射模块
    "M089": "#1f78b4",  # 未映射模块
    "M074": "#bebada",  # 对应 Cluster 11
    "M061": "#e6ab02",  # 对应 Cluster 6
    "M106": "#666666",  # 对应 Cluster 8
    "M108": "#66a61e",  # 对应 Cluster 5
    "M013": "#fb8072",  # 对应 Cluster 12
    "M027": "#7570b3",  # 对应 Cluster 3
    "M030": "#b2df8a",  # 未映射模块
    "M039": "#33a02c",  # 未映射模块
    "M094": "#e7298a",  # 对应 Cluster 4
    "M073": "#e31a1c",  # 未映射模块
    "M020": "#fdbf6f",  # 未映射模块
    "M031": "#d95f02",  # 对应 Cluster 2（主映射）
    "M007": "#a6761d",  # 对应 Cluster 7
    "M058": "#8dd3c7",  # 对应 Cluster 9
    "M005": "#ff7f00",  # 未映射模块
    "M098": "#cab2d6",  # 未映射模块
    "M059": "#fdb462",  # 对应 Cluster 14
    "M038": "#fdae61",  # 对应 Cluster 2（对比色）
    "M010": "#ffffb3",  # 对应 Cluster 10
    "M080": "#1b9e77",  # 对应 Cluster 1
    "None": "grey"
}
sc.pl.spatial(adata_combined, alpha_img = 0.5, size = 1.6, title= "", frameon = False, 
              color="plot",  palette=module_colors,show=True,
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_combined_annotation_mapping_23_modules.pdf")


# %%
# 计算ARI和NMI
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari_combined = adjusted_rand_score(adata_combined.obs['graph_cluster'], adata_combined.obs['annotation_combined_mapping'])
nmi_combined = normalized_mutual_info_score(adata_combined.obs['graph_cluster'], adata_combined.obs['annotation_combined_mapping'])
print(f"ARI: {ari_combined:.3f}, NMI: {nmi_combined:.3f}")

ari_kemans = adjusted_rand_score(adata_combined.obs['kmeans_10_clusters'], adata_combined.obs['annotation_combined_mapping'])
nmi_kemans = normalized_mutual_info_score(adata_combined.obs['kmeans_10_clusters'], adata_combined.obs['annotation_combined_mapping'])
print(f"ARI: {ari_kemans:.3f}, NMI: {nmi_kemans:.3f}")




# %%
for module in adata_self.uns['module_stats']['module_id'].unique():
    print(module)
    sc.pl.spatial(adata_self, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"], color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True,
                  save=f"/{module}_of_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Self_Anno.pdf")

# %%
overlap_records = calculate_module_overlap(adata_self, 
                                           module_list = adata_self.uns['module_stats']['module_id'].unique())
