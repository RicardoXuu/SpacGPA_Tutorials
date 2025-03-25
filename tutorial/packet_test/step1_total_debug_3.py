
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

# # %%
# # 读取数据
# adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
#                        count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
# adata.var_names_make_unique()
# adata.var_names = adata.var['gene_ids']
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# print(adata.X.shape)

# # %%
# # 读取原始数据集提供的两种注释
# graph_cluster = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_graphclust/clusters.csv',
#                             header=0, sep=',', index_col=0)
# kmeans_10_clusters = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2/analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv',
#                                 header=0, sep=',', index_col=0)

# adata.obs['graph_cluster'] = graph_cluster.loc[adata.obs_names, 'Cluster']
# adata.obs['graph_cluster'] = adata.obs['graph_cluster'].astype('category')

# adata.obs['kmeans_10_clusters'] = kmeans_10_clusters.loc[adata.obs_names, 'Cluster']
# adata.obs['kmeans_10_clusters'] = adata.obs['kmeans_10_clusters'].astype('category')


# %%
# # 读取 ggm
# start_time = time.time()
ggm = sg.load_ggm("data/ggm_gpu_32.h5")
# print(f"Read ggm: {time.time() - start_time:.5f} s")
# # 读取联合分析的ggm
# ggm_mulit_intersection = sg.load_ggm("data/ggm_mulit_intersection.h5")
# print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
# ggm_mulit_union = sg.load_ggm("data/ggm_mulit_union.h5")
# print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
# print("=====================================")
# print(ggm)
# print("=====================================")
# print(ggm_mulit_intersection)
# print("=====================================")
# print(ggm_mulit_union)

# # %%
# adata

# # %%
# # 计算模块的加权表达值
# start_time = time.time()
# sg.calculate_module_expression(adata, 
#                                ggm_obj=ggm, 
#                                top_genes=30,
#                                weighted=True,
#                                calculate_moran=True,
#                                embedding_key='spatial',
#                                k_neighbors=6)  
# print(f"Time1: {time.time() - start_time:.5f} s")

# sg.calculate_module_expression(adata, 
#                                ggm_obj=ggm_mulit_intersection, 
#                                ggm_key='intersection',
#                                top_genes=30,
#                                weighted=True,
#                                calculate_moran=True,
#                                embedding_key='spatial',
#                                k_neighbors=6)  
# print(f"Time2: {time.time() - start_time:.5f} s")

# sg.calculate_module_expression(adata, 
#                                ggm_obj=ggm_mulit_union, 
#                                ggm_key='union',
#                                top_genes=30,
#                                weighted=True,
#                                calculate_moran=True,
#                                embedding_key='spatial',
#                                k_neighbors=6)  
# print(f"Time3 {time.time() - start_time:.5f} s")


# # %%
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='module_expression_scaled',n_pcs=40)
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1')

# # %%
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='intersection_module_expression_scaled',
#                 n_pcs=adata.obsm['intersection_module_expression_scaled'].shape[1])
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_intersection')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1_intersection')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_intersection')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1_intersection')

# # %%
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='union_module_expression_scaled',
#                 n_pcs=adata.obsm['union_module_expression_scaled'].shape[1])
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_union')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1_union')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_union')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1_union')


# # %%
# adata.write("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno_union_intersection.h5ad")
# del adata, ggm, ggm_mulit_intersection, ggm_mulit_union
# gc.collect()


# %%
print(ggm.go_enrichment)



# %%
adata = sc.read("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno_union_intersection.h5ad")
adata

# # %%
# sc.pl.spatial(adata, color='graph_cluster', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_graph_cluster.pdf")
# # %%
# sc.pl.spatial(adata, color='kmeans_10_clusters', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_kmeans_10_clusters.pdf")

# # %%
# sc.pl.spatial(adata, color='leiden_0.5', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_leiden_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='leiden_1', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_leiden_1.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_0.5', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_louvan_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_1', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_louvan_1.pdf")


# # %%
# sc.pl.spatial(adata, color='leiden_0.5_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_leiden_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='leiden_1_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_leiden_1.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_0.5_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_louvan_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_1_intersection', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_intersection_louvan_1.pdf")


# # %%
# sc.pl.spatial(adata, color='leiden_0.5_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_leiden_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='leiden_1_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_leiden_1.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_0.5_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_louvan_0.5.pdf")
# # %%
# sc.pl.spatial(adata, color='louvan_1_union', size=1.6, alpha_img=0.5, frameon=False, show=True,
#               save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_union_louvan_1.pdf")



# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            #modules_used=None,
                            #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                            #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                            #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M01_anno", show=True)

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                        ggm_key='ggm',
                        #modules_used=None,
                        #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                        #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                        #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        embedding_key='spatial',
                        k_neighbors=18,
                        min_annotated_neighbors=2
                        )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M01_anno_smooth", show=True)

# %%
for module in adata.uns['module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['module_info'][adata.uns['module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"], 
                  color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True,
                  save=f"/CytAssist_FreshFrozen_Mouse_Brain_Rep2_{module}_ggm_anno.pdf")


# %%
# 合并注释（考虑空间坐标和模块表达值）
start_time = time.time()
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_used = mod_assessment[mod_assessment['module_category'] == 'identity_module']['module_id'],
                        #modules_used=None,
                        #modules_used=adata.uns['module_stats'][adata.uns['module_stats']['module_moran_I'] > 0.7]['module_id'],
                        #modules_preferred=adata.uns['module_stats'][adata.uns['module_stats']['module_moran_I'] > 0.9]['module_id'],
                        #modules_used = adata.uns['module_info']['module_id'].unique()[0:20], 
                        #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                        #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                        #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        result_anno='annotation',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )
print(f"Time: {time.time() - start_time:.5f} s")


# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", 
              na_color="black", show=True)

# %%
# 合并注释（仅考虑模块注释的细胞数目）
start_time = time.time()
sg.integrate_annotations_old(adata,
                            ggm_key='ggm',
                            #modules_used=None,
                            #modules_used=adata.uns['module_stats']['module_id'].unique()[0:30], 
                            #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                            #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                            result_anno = "annotation_old",
                         use_smooth=True
                         )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_old", show=True)

# %%
# 计算模块重叠
start_time = time.time()
overlap_records = sg.calculate_module_overlap(adata, 
                                              #modules_used = adata.uns['module_stats']['module_id'].unique()
                                              modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10']
                                              )
print(f"Time: {time.time() - start_time:.5f} s")
print(overlap_records[overlap_records['module_a'] == 'M11'])

# %%

# %%
# 问题5，关于高斯混合分布，设计activity模块的排除标准。尽量不使用先验知识，
# 待定

# %%
# 问题6，关于高斯混合分布，阈值和主成分数目的关系优化。
# 待定

# %%
# 问题7，关于高斯混合分布，除了使用高斯混合分布，也考虑表达值的排序。
#       对于一个模块，只有那些表达水平大于模块最大表达水平（或者为了防止一些离散的点，可以考虑前20个或者30个细胞的平均值作为模块最大表达水平）的一定比例的细胞才被认为是注释为该模块的
# 待定

# %%
# 问题10，关于合并注释，尝试引入模块的整体莫兰指数，来评估模块的空间分布。如果一个模块的莫兰指数很高，则优先考虑该模块的细胞的可信度。
def assess_modules_for_annotation(adata, 
                                  ggm_key='ggm',
                                  mod_stats_key='module_stats',
                                  expr_key='module_expression',
                                  overlap_threshold=0.8,
                                  high_anno_fraction=0.8,
                                  low_positive_moran=0.2,
                                  identity_min_effect=0.5,
                                  identity_positive_moran=0.3):
    """
    Assess each module for its suitability for cell type annotation merging.
    
    基于以下指标：
      - anno_fraction: 模块正注释细胞比例 = anno_one / total_cells.
      - effect_size: (mean(expression in positive cells) - mean(expression in negative cells)) / std(expression of all nonzero cells.
      - module_moran: 全模块 Moran's I (已有).
      - positive_moran: 正组 Moran's I (已有).
      - positive_mean_distance: 平均正注释细胞之间空间距离 (已有).
      - 同时，利用 adata.obs 中的注释列（"{module_id}_anno"）计算模块之间的重叠度。
    
    规则示例：
      1. 如果 anno_fraction > high_anno_fraction，则可能属于 activity 模块，不适合。
      2. 如果 positive_moran < low_positive_moran，则说明空间自相关弱，不适合。
      3. 如果正组平均距离较大（大于所有模块正组距离的75%分位数），则可能为 cross_domain，不适合。
      4. 如果 effect_size 较高（>= identity_min_effect），且正组 Moran's I >= identity_positive_moran，
         则倾向于为 identity 模块，适合合并。
      5. 如果两个模块注释重叠度超过 overlap_threshold，且当前模块的 effect_size 较低，则标记为 redundant。
      6. 其他情况标记为 unsuitable.
    
    Parameters:
      adata: AnnData object.
      mod_stats_key: str, key in adata.uns 存储模块统计信息的 DataFrame.
      expr_key: str, key in adata.obsm 存储模块加权表达矩阵.
      overlap_threshold: float, 用于判断模块注释重叠的阈值.
      high_anno_fraction: float, 正注释比例较高的阈值.
      low_positive_moran: float, 正组 Moran's I 的低阈值.
      identity_min_effect: float, 认为模块表达差异显著的效应值下限.
      identity_positive_moran: float, 正组 Moran's I 的要求下限.
    
    Returns:
      assessment_df: DataFrame，每行记录模块的各项指标、分类结果和适合性（suitable: True/False）。
      同时，更新 adata.uns['module_assessment'].
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist

    # 1. 读取模块统计信息 DataFrame
    mod_df = adata.uns[mod_stats_key].copy()
    total_cells = adata.n_obs

    # 计算额外特征：正注释比例、效应值
    mod_df['anno_fraction'] = mod_df['anno_one'] / total_cells

    # 从模块表达矩阵中计算效应值
    # 假设 adata.obsm[expr_key] 是模块表达矩阵，列顺序与 mod_df 中 module_id 对应
    mod_expr = adata.obsm[expr_key]
    # 构造成 DataFrame（列名为 module_id）
    mod_expr_df = pd.DataFrame(mod_expr, index=adata.obs.index, columns=adata.uns['module_info']['module_id'].unique())
    
    effect_sizes = []
    for mid in mod_df['module_id']:
        expr_vals = mod_expr_df[mid].values
        # 提取非零表达
        non_zero = expr_vals != 0
        if np.sum(non_zero) == 0:
            effect_sizes.append(0)
        else:
            pos = expr_vals[non_zero][np.array([1 if x is not None else 0 for x in adata.obs[f"{mid}_anno"].values])] \
                  if f"{mid}_anno" in adata.obs.columns else expr_vals[non_zero]
            # 计算正组与负组均值及标准差（仅使用非零值）
            pos_mean = np.mean(expr_vals[non_zero][expr_vals[non_zero] > np.median(expr_vals[non_zero])])
            neg_mean = np.mean(expr_vals[non_zero][expr_vals[non_zero] <= np.median(expr_vals[non_zero])])
            std_all = np.std(expr_vals[non_zero])
            eff = (pos_mean - neg_mean) / std_all if std_all != 0 else 0
            effect_sizes.append(eff)
    mod_df['effect_size'] = effect_sizes

    # 2. 计算模块之间的注释重叠度
    module_ids = mod_df['module_id'].unique()
    overlap = pd.DataFrame(index=module_ids, columns=module_ids, dtype=float)
    for i in module_ids:
        for j in module_ids:
            # 提取对应的注释向量（转为二值数组，1表示细胞被该模块注释）
            col_i = adata.obs[f"{i}_anno"].cat.codes.values if pd.api.types.is_categorical_dtype(adata.obs[f"{i}_anno"]) else adata.obs[f"{i}_anno"].values
            col_j = adata.obs[f"{j}_anno"].cat.codes.values if pd.api.types.is_categorical_dtype(adata.obs[f"{j}_anno"]) else adata.obs[f"{j}_anno"].values
            overlap.loc[i, j] = np.sum((col_i==1) & (col_j==1)) / total_cells

    # 3. 计算各模块正组平均空间距离（提取模块统计中 positive_mean_distance，然后计算75%分位数）
    pos_mean_dists = mod_df['positive_mean_distance'].dropna().values
    if len(pos_mean_dists) > 0:
        pos_md_threshold = np.percentile(pos_mean_dists, 75)
    else:
        pos_md_threshold = np.nan

    # 4. 对每个模块基于以上特征进行分类
    categories = []
    suitability = []
    for idx, row in mod_df.iterrows():
        # 读取已有指标
        af = row['anno_fraction']
        eff = row['effect_size']
        mod_moran = row['module_moran_I']
        pos_moran = row['positive_moran_I']
        pos_md = row['positive_mean_distance']
        # 初步判断：我们希望 identity 模块满足：
        #   - 正注释比例适中：0.2 ~ 0.6
        #   - 效应值高（>= identity_min_effect）
        #   - 正组 Moran's I 较高 (>= identity_positive_moran)
        #   - 正注释细胞聚集，即平均空间距离低 (< pos_md_threshold)
        if af > high_anno_fraction:
            cat = "activity_module"  # 过多细胞被注释，说明是生命活动模块
            suit = False
        elif pos_moran < low_positive_moran:
            cat = "weak_spatial_autocorrelation"
            suit = False
        # elif pos_md > pos_md_threshold:
        #     cat = "cross_domain"
        #     suit = False
        elif eff >= identity_min_effect and pos_moran >= identity_positive_moran:
            cat = "identity_module"
            suit = True
        else:
            cat = "unsuitable"
            suit = False
        categories.append(cat)
        suitability.append(suit)
    
    mod_df['module_category'] = categories
    mod_df['merge_suitability'] = suitability
    
    # 5. 判断重叠模块：如果两个模块注释重叠度 > overlap_threshold，则将效应值较低者标记为 redundant
    for i in module_ids:
        for j in module_ids:
            if i == j:
                continue
            if overlap.loc[i, j] > overlap_threshold:
                # 如果效应值较低，则将 i 标记为 redundant
                if mod_df.loc[mod_df['module_id'] == i, 'effect_size'].values[0] < mod_df.loc[mod_df['module_id'] == j, 'effect_size'].values[0]:
                    mod_df.loc[mod_df['module_id'] == i, 'module_category'] = 'annotation_similar'
                    mod_df.loc[mod_df['module_id'] == i, 'merge_suitability'] = False
    
    # 6. 存储评估结果到 adata.uns['module_assessment']
    adata.uns['module_assessment'] = mod_df.copy()
    
    return mod_df


# %%
# 测试
mod_assessment = assess_modules_for_annotation(adata,
                                               ggm_key='ggm',
                                               mod_stats_key='module_stats',
                                               expr_key='module_expression',
                                               overlap_threshold=0.7,
                                               high_anno_fraction=0.5,
                                               low_positive_moran=0.2,
                                               identity_min_effect=0.5,
                                               identity_positive_moran=0.3)

# %%
mod_assessment[mod_assessment['module_category'] == 'identity_module']['module_id']

# %%
start_time = time.time()
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_used = mod_assessment[mod_assessment['module_category'] == 'identity_module']['module_id'],
                        #modules_used=None,
                        #modules_used=adata.uns['module_stats'][adata.uns['module_stats']['module_moran_I'] > 0.7]['module_id'],
                        #modules_preferred=adata.uns['module_stats'][adata.uns['module_stats']['module_moran_I'] > 0.9]['module_id'],
                        #modules_used = adata.uns['module_info']['module_id'].unique()[0:20], 
                        #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                        #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                        #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        result_anno='annotation',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
print(f"Time: {time.time() - start_time:.5f} s")


# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", 
              na_color="white", show=True)



# %%
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.metrics import jaccard_score

def assess_module_quality(
    adata,
    ggm_key='ggm',
    use_smooth=True,
    spatial_key='spatial',
    # 用户可以提供与生命活动相关的GO关键词列表
    go_activity_keywords=["metabolic", "cell cycle", "signal transduction", "translation"],
    # 阈值参数（可根据数据进行调整）
    fraction_threshold_activity=0.7,
    moran_threshold=0.3,
    flooding_fraction_threshold=0.9,
    skew_threshold=0.5,
    top1pct_threshold=0.1,
    spatial_dispersion_ratio_threshold=0.8,
    jaccard_overlap_threshold=0.8
):
    """
    评估各模块是否适合用于细胞身份注释。对每个模块计算以下特征：
      - 注释细胞数目及比例；
      - 空间分布离散程度（基于空间坐标的标准差相对于全局的比例）；
      - 莫兰指数（moran或positive_moran）；
      - 表达数据的偏度(skew)和top1pct_ratio；
      - GO富集（如果模块统计中包含相关信息）；
      - 利用加权表达数据计算模块的判别能力（注释细胞与非注释细胞的表达差异）。
    同时，对模块间注释细胞集合进行重叠计算，识别注释相似模块。  
      
    参数：
      adata: anndata.AnnData对象，要求包含以下信息：
             - adata.uns['module_info']: 每个模块包含的基因及其权重和莫兰指数。
             - adata.uns['module_stats']: 每个模块的统计指标（包含模块ID、moran、skew、top1pct_ratio、（可选）GO信息等）。
             - adata.obsm['module_expression']: 模块加权表达矩阵，列应与模块ID一致。
             - adata.obs中，各模块的注释信息，列名格式为 "M1_anno" 或 "M1_anno_smooth"。
      ggm_key: 指定的GGM对象的key，默认 'ggm'，用于在adata.uns['ggm_keys']中查找对应模块统计信息。
      use_smooth: 是否使用平滑后的注释结果，默认True。
      spatial_key: 存储空间坐标的键，默认 'spatial'。
      go_activity_keywords: 列表，包含生命活动相关的GO关键词，用于判断activity模块。
      fraction_threshold_activity: 如果模块注释比例高于该值，且GO富集匹配生命活动关键词，则标记为activity模块。
      moran_threshold: 如果模块的莫兰指数低于此阈值，则标记为弱空间自相关模块。
      flooding_fraction_threshold: 注释比例高于该值且表达差异较低，标记为空间泛滥模块。
      skew_threshold: 用于判断表达分布是否平坦（数值较低表示平坦）。
      top1pct_threshold: 用于判断表达在细胞间差异是否明显（值较低表示差异不大）。
      spatial_dispersion_ratio_threshold: 注释细胞在空间分布的标准差与全局标准差的比例阈值，超过该值视为跨结构域模块。
      jaccard_overlap_threshold: 两个模块注释细胞集合Jaccard重合度的阈值，超过该值则判断为注释相似模块。
      
    返回：
      一个DataFrame，每一行对应一个模块，包含以下列：
        - module_id: 模块ID。
        - annotated_cell_count: 注释该模块的细胞数。
        - fraction_annotated: 注释比例。
        - moran: 模块的莫兰指数。
        - skew: 表达分布偏度。
        - top1pct_ratio: 表达数据中top1%比率。
        - spatial_dispersion: 注释细胞空间分布的标准差比全局标准差的比例。
        - discriminative_power: 注释细胞与非注释细胞加权表达的中位数差值。
        - issues: 识别出的模块问题（可能为activity, cross_structural, weak_spatial_autocorrelation, spatial_flooding, annotation_similar）。
        - suitable_for_integration: 布尔值，若存在任一问题，则标记为False，否则为True。
    """
    # 检查必要信息
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    
    # 提取模块统计信息，假设其为DataFrame格式
    module_stats = adata.uns[mod_stats_key]
    
    total_cells = adata.n_obs
    spatial_coords = adata.obsm[spatial_key]
    overall_std = np.std(spatial_coords, axis=0).mean()  # 全局空间分布的平均标准差

    # 预先存储各模块的注释二值向量及其它指标，便于后续模块间对比
    module_info_list = []

    # 处理模块时优先按注释细胞数目从多到少排序
    # 这里假设 module_stats 中有 'module_id' 字段
    modules = module_stats['module_id'].tolist()
    # 收集每个模块的注释细胞数
    module_cell_counts = {}
    for mid in modules:
        anno_col = f"{mid}_anno_smooth" if use_smooth and f"{mid}_anno_smooth" in adata.obs.columns else f"{mid}_anno"
        if anno_col not in adata.obs.columns:
            raise ValueError(f"Annotation column for module {mid} not found in adata.obs.")
        # 假设细胞被标注为该模块时，adata.obs[anno_col]的值正好等于模块ID
        cell_bool = adata.obs[anno_col] == mid
        count = int(cell_bool.sum())
        module_cell_counts[mid] = count

    modules_sorted = sorted(modules, key=lambda x: module_cell_counts[x], reverse=True)

    # 先收集各模块的注释向量及判别能力（利用加权表达数据计算）
    # 假设 adata.obsm['module_expression'] 是一个 DataFrame 或二维array，其列名与模块ID一致
    if isinstance(adata.obsm['module_expression'], pd.DataFrame):
        module_expr_df = adata.obsm['module_expression']
    else:
        # 若为array，则需要提供列顺序信息
        raise ValueError("module_expression in adata.obsm should be a DataFrame with module IDs as columns.")

    module_anno_dict = {}  # 存储每个模块的注释二值向量（1表示该细胞被注释）
    module_disc_power = {}  # 存储模块的判别能力

    for mid in modules_sorted:
        anno_col = f"{mid}_anno_smooth" if use_smooth and f"{mid}_anno_smooth" in adata.obs.columns else f"{mid}_anno"
        cell_bool = (adata.obs[anno_col] == mid).values.astype(int)  # 0/1向量
        module_anno_dict[mid] = cell_bool
        
        # 计算判别能力：注释细胞与非注释细胞的加权表达中位数差
        expr = module_expr_df[mid].values
        if cell_bool.sum() > 0 and (len(cell_bool) - cell_bool.sum()) > 0:
            median_diff = np.median(expr[cell_bool==1]) - np.median(expr[cell_bool==0])
        else:
            median_diff = 0
        module_disc_power[mid] = median_diff

    # 初始化保存结果的列表
    results = []

    # 逐个模块进行评估
    for mid in modules_sorted:
        issues = []
        anno = module_anno_dict[mid]
        annotated_count = int(anno.sum())
        fraction_annotated = annotated_count / total_cells

        # 获取模块统计信息中的指标
        # 这里假设module_stats中包含'moran', 'skew', 'top1pct_ratio'字段
        row = module_stats[module_stats['module_id'] == mid].iloc[0]
        moran = row.get('moran', np.nan)
        skew = row.get('skew', np.nan)
        top1pct_ratio = row.get('top1pct_ratio', np.nan)
        
        # 计算空间分布离散程度
        # 计算注释细胞的空间坐标标准差（均值）
        if annotated_count > 1:
            coords_anno = spatial_coords[adata.obs.index[anno.astype(bool)], :]
            dispersion = np.std(coords_anno, axis=0).mean()
        else:
            dispersion = np.nan
        spatial_dispersion_ratio = dispersion / overall_std if overall_std > 0 else np.nan

        # 计算判别能力
        disc_power = module_disc_power[mid]

        # 1. Activity module判断
        # 如果注释比例过高，并且（假如提供了GO信息且GO富集中包含生命活动相关关键词）
        go_info = row.get('go_terms', "")  # 假定为字符串，可为空
        if fraction_annotated > fraction_threshold_activity:
            if any(kw.lower() in go_info.lower() for kw in go_activity_keywords):
                issues.append("activity")
        
        # 2. 跨结构域表达模块判断：如果注释细胞在空间上过于分散
        if not np.isnan(spatial_dispersion_ratio):
            if spatial_dispersion_ratio > spatial_dispersion_ratio_threshold:
                issues.append("cross_structural")
        
        # 3. 弱空间自相关模块判断：根据莫兰指数
        if not np.isnan(moran) and moran < moran_threshold:
            issues.append("weak_spatial_autocorrelation")
        
        # 4. 空间泛滥模块判断
        if (fraction_annotated > flooding_fraction_threshold and 
            not np.isnan(skew) and skew < skew_threshold and 
            not np.isnan(top1pct_ratio) and top1pct_ratio < top1pct_threshold):
            issues.append("spatial_flooding")
        
        # 5. 注释相似模块判断（在后续进行模块两两比较后补充，目前先初始化为空）
        # 这里我们暂时不直接判断，而是存储每个模块的注释向量和判别能力，
        # 后续在所有模块之间进行pairwise比较，若有高重合则判断判别能力较弱的那个
        results.append({
            "module_id": mid,
            "annotated_cell_count": annotated_count,
            "fraction_annotated": fraction_annotated,
            "moran": moran,
            "skew": skew,
            "top1pct_ratio": top1pct_ratio,
            "spatial_dispersion": spatial_dispersion_ratio,
            "discriminative_power": disc_power,
            "issues": issues  # 目前还未加入注释相似模块
        })

    result_df = pd.DataFrame(results)
    
    # 5. 注释相似模块判断（基于Jaccard相似度）
    # 对于每对模块，如果Jaccard相似度超过阈值，则将判别能力较弱的模块标记为"annotation_similar"
    mod_ids = result_df['module_id'].tolist()
    for i in range(len(mod_ids)):
        for j in range(i+1, len(mod_ids)):
            mid1 = mod_ids[i]
            mid2 = mod_ids[j]
            vec1 = module_anno_dict[mid1]
            vec2 = module_anno_dict[mid2]
            # 计算Jaccard指数
            union = np.logical_or(vec1, vec2).sum()
            intersection = np.logical_and(vec1, vec2).sum()
            jaccard = intersection / union if union > 0 else 0
            if jaccard > jaccard_overlap_threshold:
                # 比较两个模块的判别能力，判别能力较低者标记为 annotation_similar
                if module_disc_power[mid1] < module_disc_power[mid2]:
                    # mid1判别能力较弱
                    result_df.loc[result_df['module_id'] == mid1, 'issues'] = result_df.loc[result_df['module_id'] == mid1, 'issues'].apply(lambda x: list(set(x + ["annotation_similar"])))
                else:
                    result_df.loc[result_df['module_id'] == mid2, 'issues'] = result_df.loc[result_df['module_id'] == mid2, 'issues'].apply(lambda x: list(set(x + ["annotation_similar"])))
    
    # 设定最终是否适合整合注释：只要存在上述任一问题，则认为该模块不适合用于细胞身份注释（identity module）
    result_df['suitable_for_integration'] = result_df['issues'].apply(lambda x: False if len(x) > 0 else True)
    
    return result_df

# %%
# 测试
quality_df = assess_module_quality(adata,
                                   ggm_key='ggm',
                                   use_smooth=True,
                                   spatial_key='spatial',
                                   go_activity_keywords=["metabolic", "cell"],
                                   fraction_threshold_activity=0.5)


# %%
print(adata.uns['module_stats'].head())


# %%
print(adata.obs['M01_anno_smooth'].value_counts())


# %%
adata.obsm['module_expression'].shape


# %%
# 问题11，关于合并注释，尝试结合louvain或者leiden的聚类结果，在每个聚类之内使用模块来精准注释。

# %%
# 问题13，关于合并注释，在adata的uns中添加一个配色方案，为每个模块指定配色，特别是模块过多的时候。



















# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                             ggm_key='intersection',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         max_iter=200,
                         prob_threshold=0.99,
                         min_samples=10,
                         n_components=3,
                         enable_fallback=True,
                         random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                      ggm_key='intersection',
                    #modules_used=None,
                    #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                    #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                    #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                    embedding_key='spatial',
                    k_neighbors=18,
                    min_annotated_neighbors=2
                    )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
# 合并注释（考虑空间坐标和模块表达值）
start_time = time.time()
sg.integrate_annotations(adata,
                         ggm_key='ggm',
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
sg.integrate_annotations(adata,
                         ggm_key='union',
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='union_annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
sg.integrate_annotations(adata,
                         ggm_key='intersection',
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='intersection_annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.integrate_annotations(adata,
                         ggm_key='intersection',
                         cross_ggm=True,
                         modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
                                      'intersection_M001', 'intersection_M002', 'intersection_M003', 'intersection_M004', 'intersection_M005',
                                      'intersection_M006', 'intersection_M007', 'intersection_M008', 'intersection_M009', 'intersection_M010',
                                      'union_M001', 'union_M002', 'union_M003', 'union_M004', 'union_M005',
                                      'union_M006', 'union_M007', 'union_M008', 'union_M009', 'union_M010'],
                  #modules_used=None,
                  #modules_used = adata.uns['module_info']['module_id'].unique(), 
                  #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='merge_annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_similarity_ratio=1
                  )
# %%
# %%
# 合并注释（仅考虑模块注释的细胞数目）
start_time = time.time()
sg.integrate_annotations_old(adata,
                             ggm_key='ggm',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "annotation_old",
                         use_smooth=True
                         )
sg.integrate_annotations_old(adata,
                             ggm_key='intersection',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "intersection_annotation_old",
                         use_smooth=True
                         )
sg.integrate_annotations_old(adata,
                             ggm_key='union',
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "union_annotation_old",
                         use_smooth=True
                         )
sg.integrate_annotations_old(adata,
                             ggm_key='ggm',
                             cross_ggm=True,
                             modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
                                      'intersection_M001', 'intersection_M002', 'intersection_M003', 'intersection_M004', 'intersection_M005',
                                      'intersection_M006', 'intersection_M007', 'intersection_M008', 'intersection_M009', 'intersection_M010',
                                      'union_M001', 'union_M002', 'union_M003', 'union_M004', 'union_M005',
                                      'union_M006', 'union_M007', 'union_M008', 'union_M009', 'union_M010'],
                         #modules_used=None,
                         #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "merge_annotation_old",
                         use_smooth=True
                         )
print(f"Time: {time.time() - start_time:.5f} s")




# %%
for module in adata.uns['intersection_module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['intersection_module_info'][adata.uns['intersection_module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"],
                    color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True)
    

# %%
for module in adata.uns['union_module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['union_module_info'][adata.uns['union_module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"],
                    color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True)




# %%
# 计算模块重叠
start_time = time.time()
overlap_records = sg.calculate_module_overlap(adata, 
                                           modules_used = adata.uns['module_stats']['module_id'].unique())
print(f"Time: {time.time() - start_time:.5f} s")
print(overlap_records[overlap_records['module_a'] == 'M01'])

# %%
# 注释结果可视化
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="union_annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="intersection_annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="merge_annotation", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="union_annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="intersection_annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="merge_annotation_old", show=True)

# %%
# 保存adata
adata.write("data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno.h5ad")


# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id'].unique()
# 1. 原始注释绘图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Raw.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()    
        sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_anno", frameon = False, color="plot_anno",show=False)
        raw_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Raw.png"
        plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# 2. 平滑注释绘图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Smooth.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno_smooth"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()
        sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_anno_smooth", frameon = False, color="plot_anno",show=False)
        smooth_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Smooth.png"
        plt.savefig(smooth_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(smooth_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# 3. 模块加权表达图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Exp.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_exp", frameon = False, color=f"{module}_exp", color_map="Reds", show=False)
    raw_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Exp.png"
    plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# %%






# %%
# 细胞注释相关的问题
# 问题1，关于计算平均表达值。当使用新的ggm结果注释已经存在module expression的adata时，会报错
# 添加ggm_key 参数，为GGM指定一个key，用来区分不同的ggm结果。
# 解决

# %%
# 问题2，关于计算平均表达值。添加可选参数，计算模块内每个基因的莫兰指数以及模块整体的莫兰指数。
# 解决

# %%
# 问题3，关于模块注释的全部函数，添加反选参数，用来反向排除模块。
# 解决

# %%
# 问题4，关于模块注释的全部函数, 细胞按模块的注释结果改为category类型。而不是现在的0，1，int类型。并注意，之后在涉及到使用这些数据的时候还要换回int类型。
# 解决

# %%
# 问题8，关于平滑处理，在使用的时候，无法仅处理部分模块。
# 解决

# %%
# 问题9，关于合并注释，优化keep_modules的参数。
# 解决

# %%
# 问题12，关于合并注释，注释结果中，字符串None改为空值的None。
# 解决

# %%
# 问题14，关于合并注释，neighbor_similarity_ratio参数似乎会导致activity模块的权重过高。考虑将其设置为0或者1来避免考虑neighbor_similarity_ratio
# 解决
