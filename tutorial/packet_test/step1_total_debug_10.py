
# %%
# 开发新的平滑函数
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
ggm.adjust_cutoff(pcor_threshold=0.059)
#best_inf, _ = sg.find_best_inflation(ggm, min_inflation=1.1, phase=3, show_plot=True)
ggm.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=1.38, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm.modules_summary.shape)


# %%
sg.annotate_with_ggm(adata, ggm,
                     ggm_key='ggm')


# %%
sg.smooth_annotations(adata,
                      ggm_key='ggm',
                      k_neighbors=24)


# %%
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import random

def smooth_annotations_optimized(adata,
                                 ggm_key='ggm',
                                 modules_used=None,
                                 modules_excluded=None,
                                 embedding_key='spatial',
                                 k_neighbors=24,
                                 min_neighbors_keep=1,
                                 min_neighbors_expand=2,
                                 expr_threshold_frac=0.5,
                                 relative_neighborhood_expr=True):
    """
    优化的平滑模块注释函数：
    - 基于邻居支持扩展原未标记细胞的模块标记
    - 动态调整边缘细胞的保留阈值
    - 综合表达强度和邻居情况决定保留或移除标记
    """
    # 检查ggm_key和embedding是否存在
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm")
    # 确定要处理的模块列表
    if modules_used is None:
        modules_used = list(adata.uns[mod_stats_key]['module_id'].unique())
    if modules_excluded is not None:
        modules_used = [m for m in modules_used if m not in modules_excluded]
    # 移除已有的平滑结果列
    existing_cols = [f"{m}_anno_smooth_optimized" for m in modules_used if f"{m}_anno_smooth_optimized" in adata.obs]
    if existing_cols:
        adata.obs.drop(columns=existing_cols, inplace=True)
    # 提取空间坐标和初始注释矩阵
    coords = adata.obsm[embedding_key]
    # 计算k近邻索引
    nbrs_model = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean').fit(coords)
    distances, knn_indices = nbrs_model.kneighbors(coords)
    # 准备初始注释和表达数据
    anno_cols = [f"{m}_anno" for m in modules_used]
    exp_cols = [f"{m}_exp" for m in modules_used]
    exp_trim_cols = [f"{m}_exp_trim" for m in modules_used]
    module_annotations = adata.obs[anno_cols].copy()
    module_expressions = adata.obs[exp_cols].copy() if set(exp_cols) <= set(adata.obs.columns) else None
    module_expressions_trim = adata.obs[exp_trim_cols].copy() if set(exp_trim_cols) <= set(adata.obs.columns) else None
    # 将anno列转换为0/1整数
    for m in modules_used:
        col = f"{m}_anno"
        module_annotations[col] = (adata.obs[col] == m).astype(int)
    n_cells = module_annotations.shape[0]
    # 创建矩阵存储新注释结果（初始为原注释拷贝，可原地修改）
    new_annotations = module_annotations.values.copy()
    # 扩展阶段：为每个模块尝试将符合条件的细胞从0设为1
    for j, m in enumerate(modules_used):
        # 获取第j个模块的初始注释和表达向量
        anno_vec = module_annotations.iloc[:, j].values
        exp_vec = module_expressions.iloc[:, j].values if module_expressions is not None else None
        # 原GMM阈值：可从module_stats中提取 threshold
        thr = None
        if mod_stats_key in adata.uns:
            stats_df = adata.uns[mod_stats_key]
            if 'threshold' in stats_df.columns:
                row = stats_df[stats_df['module_id'] == m]
                if len(row) > 0:
                    thr = float(row['threshold'].iloc[0])
        # 扩展条件
        for i in range(n_cells):
            if anno_vec[i] == 1:
                continue  # 原本已标记则跳过
            # 计算邻居阳性数
            neighbors = knn_indices[i, 1:]  # 排除自身
            neigh_anno_count = np.sum(anno_vec[neighbors])
            if neigh_anno_count < min_neighbors_expand:
                continue  # 邻居阳性不足，不扩展
            # 检查表达阈值条件
            if exp_vec is not None:
                # 若未提供threshold则使用阳性邻居平均*frac
                if thr is None:
                    if relative_neighborhood_expr:
                        # 邻居阳性表达均值
                        pos_neighbors = neighbors[anno_vec[neighbors] == 1]
                        if len(pos_neighbors) > 0:
                            thr_local = exp_vec[pos_neighbors].mean() * expr_threshold_frac
                        else:
                            thr_local = 0
                    else:
                        thr_local = np.quantile(exp_vec, 0.9) * expr_threshold_frac
                else:
                    thr_local = thr * expr_threshold_frac
                if exp_vec[i] < thr_local:
                    continue  # 自身表达不足，不扩展
            # 满足条件，标记扩展
            new_annotations[i, j] = 1
    # 更新注释矩阵（包括扩展新增的标记）
    # 收缩阶段：移除不满足保留条件的标记
    for j, m in enumerate(modules_used):
        anno_vec = new_annotations[:, j]
        exp_vec = module_expressions.iloc[:, j].values if module_expressions is not None else None
        for i in range(n_cells):
            if anno_vec[i] == 0:
                continue
            neighbors = knn_indices[i, 1:]
            neigh_anno_count = np.sum(anno_vec[neighbors])
            # 若阳性邻居少于保留阈值，则考虑移除
            if neigh_anno_count < min_neighbors_keep:
                # # 但若自身表达非常高则仍保留
                # if exp_vec is not None:
                #     if module_expressions_trim is not None:
                #         # 以阳性邻居表达均值为参考
                #         pos_neighbors = neighbors[anno_vec[neighbors] == 1]
                #         if len(pos_neighbors) > 0:
                #             neigh_expr_mean = exp_vec[pos_neighbors].mean()
                #         else:
                #             neigh_expr_mean = 0
                #         #if exp_vec[i] < neigh_expr_mean * 0.5:
                #         if exp_vec[i] < neigh_expr_mean:
                #             new_annotations[i, j] = 0
                #     else:
                #         # 没有trim信息则用阳性总体均值或阈值
                #         if thr is None:
                #             thr_value = np.quantile(exp_vec, 0.9)
                #         else:
                #             thr_value = thr
                #         if exp_vec[i] < thr_value:
                #             new_annotations[i, j] = 0
                # else:
                #     # 无表达信息则严格按邻居规则
                #     new_annotations[i, j] = 0
                new_annotations[i, j] = 0
    # 构建结果DataFrame
    smooth_df = pd.DataFrame(new_annotations, index=adata.obs.index, columns=[f"{m}_anno_smooth_optimized" for m in modules_used])
    # 将0/1转换为类别：1->模块ID, 0->None
    for m in modules_used:
        col = f"{m}_anno_smooth_optimized"
        smooth_df[col] = np.where(smooth_df[col] == 1, m, None)
        smooth_df[col] = pd.Categorical(smooth_df[col])
    # 合并结果到adata.obs
    adata.obs = pd.concat([adata.obs, smooth_df], axis=1)
    print("Optimized annotation smoothing completed.")


# %%
# 测试
smooth_annotations_optimized(adata, ggm_key='ggm',
                             embedding_key='spatial',
                             k_neighbors=24,
                             min_neighbors_keep=2,
                             min_neighbors_expand=12,
                             expr_threshold_frac=0.8,
                             relative_neighborhood_expr=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M10_anno_smooth", show=True)


# %%
for module in adata.uns['module_stats']['module_id']:
    sc.pl.spatial(adata, size=1.6, alpha_img=0.5, frameon = False, color_map="Reds", 
                  color=[f"{module}_exp",f"{module}_anno",
                         f"{module}_anno_smooth",f"{module}_anno_smooth_optimized"],show=True)
# %%
