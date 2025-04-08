
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
import numpy as np
import scipy.spatial

def smooth_annotations(
    adata, 
    label_key='initial_label', 
    output_key='smoothed_label', 
    radius=None, k=6, 
    lambda_weight=1.0, 
    max_iter=10, 
    min_improve=0.001
):
    """
    对AnnData对象的空间注释标签进行平滑优化。
    
    参数：
    - adata: AnnData，必须包含obs[label_key]初始标签和obsm['spatial']空间坐标。
    - label_key: 初始标签在adata.obs中的列名。
    - output_key: 平滑优化后的标签将保存的列名。
    - radius: 定义邻居的半径距离（如果提供，则使用距离阈值邻居）。
    - k: 定义邻居数量（如果radius未提供，则使用每个点最近的k个邻居）。
    - lambda_weight: 空间平滑项权重参数λ。
    - max_iter: 最大迭代次数。
    - min_improve: 最小改进阈值，低于该阈值则提早停止迭代。
    """
    # 1. 初始化数据结构
    coords = adata.obsm['spatial']
    labels = adata.obs[label_key].astype(str).values  # 初始标签数组（字符串）
    n = labels.shape[0]
    # 邻接表：根据radius或k构建
    if radius is not None:
        # 基于半径的邻居：使用KDTree高效查询
        tree = scipy.spatial.cKDTree(coords)
        neighbors = [tree.query_ball_point(coords[i], r=radius) for i in range(n)]
    else:
        # 基于k近邻
        tree = scipy.spatial.cKDTree(coords)
        neighbors = [list(tree.query(coords[i], k+1)[1][1:]) for i in range(n)]  # 排除自己
    neighbors = [set(neigh_list) - {i} for i, neigh_list in enumerate(neighbors)]  # 确保不包含自身且为集合
    
    labels_smoothed = labels.copy()  # 将要优化的标签
    adata.obs[output_key] = labels_smoothed  # 先写入，以防中途需要监测
    
    # 预计算模块平均表达或标志基因（为简单，此处用X的列均值作为示例）
    X = adata.X  # 假定已经标准化/对数处理过
    # 计算每个标签的均值表达向量（在dense情况下）
    unique_labels = np.unique(labels_smoothed)
    label_indices = {lab: np.where(labels_smoothed == lab)[0] for lab in unique_labels}
    mean_expr = {lab: np.array(X[label_indices[lab]].mean(axis=0)).ravel() for lab in unique_labels}
    
    # 函数：计算一个点与某模块的距离（用欧氏距离，也可以换成相关距离或KL散度等）
    def expr_distance(i, lab):
        if lab not in mean_expr:
            return np.inf
        diff = X[i] - mean_expr[lab]
        return np.linalg.norm(diff)
    
    # 函数：计算点i的能量增量（若将i的标签从old_lab换成new_lab，对能量的影响）
    def delta_energy(i, old_lab, new_lab):
        # 数据项差值: D(i,new) - D(i,old)
        data_diff = expr_distance(i, new_lab) - expr_distance(i, old_lab)
        # 平滑项差值: 计算邻居中有多少不一样标签
        smooth_diff = 0.0
        for j in neighbors[i]:
            # 邻居标签
            lab_j = labels_smoothed[j]
            # 原本i与邻居j不同标签的情况
            orig_diff = 1.0 if (lab_j != old_lab) else 0.0
            # 如果变为new_lab，不同标签的情况
            new_diff = 1.0 if (lab_j != new_lab) else 0.0
            smooth_diff += (new_diff - orig_diff)
        smooth_diff *= lambda_weight
        return data_diff + smooth_diff
    
    # 主循环迭代
    for it in range(max_iter):
        changes = 0
        # 计算本轮要检查的候选点集合（边缘点）
        # 这里简单用邻居不同标签数量来判断
        edge_points = []
        for i in range(n):
            neigh_labels = [labels_smoothed[j] for j in neighbors[i]]
            # 若邻居中不同于自身标签的比例超过30%，则视为边缘点
            if neigh_labels and (np.mean([1 if lab != labels_smoothed[i] else 0 for lab in neigh_labels]) > 0.3):
                edge_points.append(i)
        # 遍历候选点尝试更新
        for i in edge_points:
            current_lab = labels_smoothed[i]
            # 计算与邻居最常见标签之间的差异，仅考虑邻居出现的标签
            neighbor_labels = [labels_smoothed[j] for j in neighbors[i]]
            candidate_labs = set(neighbor_labels)
            # 排除当前标签自身，如果没有别的候选则跳过
            if current_lab in candidate_labs:
                candidate_labs.remove(current_lab)
            if len(candidate_labs) == 0:
                continue
            # 选取使能量下降最多的候选标签
            best_lab = current_lab
            best_delta = 0.0
            for lab in candidate_labs:
                dE = delta_energy(i, current_lab, lab)
                if dE < best_delta:  # 能量降低
                    best_delta = dE
                    best_lab = lab
            if best_lab != current_lab:
                # 更新标签
                labels_smoothed[i] = best_lab
                changes += 1
                # 更新均值表达（增量更新也可，这里简化为重新计算该两个类的均值）
                for lab in [current_lab, best_lab]:
                    idx = np.where(labels_smoothed == lab)[0]
                    if len(idx) > 0:
                        mean_expr[lab] = np.array(X[idx].mean(axis=0)).ravel()
        adata.obs[output_key] = labels_smoothed  # 更新结果到AnnData
        # 检查本轮更改数占总点数比例
        if changes / n < min_improve:
            # 改变很少，提早停止
            break
    # 循环结束，返回AnnData（已经在obs写入结果列）
    return adata

# %%