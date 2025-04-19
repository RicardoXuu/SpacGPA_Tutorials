
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
# ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
# print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
# ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
# print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
print("=====================================")
print(ggm)
# print("=====================================")
# print(ggm_mulit_intersection)
# print("=====================================")
# print(ggm_mulit_union)


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
# 方案1
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
                                 min_neighbors_keep=2,
                                 min_neighbors_expand=12,
                                 expr_threshold_frac=0.8,
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
# 方案2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def smooth_annotations_advanced(adata,
                                ggm_key='ggm', 
                                modules_used=None,
                                modules_excluded=None, 
                                embedding_key='spatial',
                                k_neighbors=24,
                                min_votes=0.6,
                                alpha=0.5,
                                lambda_coef=0.5,
                                beta=0.3,
                                max_iter=10,
                                tolerance=1e-3):
    """
    高级平滑处理函数：基于细胞邻域信息与加权表达水平，利用局部统计和形态学思想对模块注释进行自适应平滑处理。
    
    参数说明：
    -----------
    adata: anndata.AnnData
        存储空间转录组数据的AnnData对象，必须包含空间坐标数据和各模块的初始注释、加权表达数据。
    ggm_key: str
        存储GGM模型信息的键，用于从 adata.uns 中提取相关信息。
    modules_used: list or None
        指定需要平滑处理的模块列表，如果为None，则默认处理所有模块。
    modules_excluded: list or None
        指定不参与平滑处理的模块列表，优先级高于modules_used。
    embedding_key: str
        存储空间坐标数据的键，默认为'spatial'。
    k_neighbors: int
        最近邻数量，用于构建局部邻域，一般建议依据细胞密度调节（默认24）。
    min_votes: float
        邻域投票阈值（0~1之间），当邻域中超过此比例细胞满足条件时，可对中心细胞补充正注释。
    alpha: float
        加权表达与0/1注释融合时表达占比（0~1之间），alpha越大更依赖原始表达数据。
    lambda_coef: float
        调控局部阈值收缩程度的系数。
    beta: float
        用于局部密度修正的权重修正参数。
    max_iter: int
        最大迭代次数，防止无限循环。
    tolerance: float
        收敛判据，若两次迭代间注释变化比例低于该值，则认为收敛。
        
    返回：
    -----------
    更新后的adata.obs中增加每个模块的平滑注释结果（字段名格式为'{module}_anno_smooth'）
    并输出详细的调试信息，便于后续结果评估。
    
    算法流程：
    -----------
    1. 检查输入数据与必要字段是否存在，预处理需要平滑的模块列表。
    2. 清除已有的平滑注释字段，提取空间坐标与各模块的初始注释及加权表达（_exp, _exp_trim）。
    3. 对每个模块，依据空间坐标构建KNN邻域，计算局部表达权重，并依据局部均值、标准差及密度修正构建自适应阈值。
    4. 利用局部投票机制对每个细胞更新注释状态；同时对边界与特殊区域采用形态学修正方法进行二次校正。
    5. 迭代更新直至注释状态收敛或达到最大迭代次数。
    
    详细步骤：
    -----------
    """
    # 1. 检查输入和预处理
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} 不存在于 adata.uns['ggm_keys'] 中。")
    
    module_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} 不存在于 adata.obsm 中，请确保空间坐标信息可用。")
    
    # 若 modules_used 未提供则获取所有模块ID
    if modules_used is None:
        modules_used = list(adata.uns[module_stats_key]['module_id'].unique())
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
    
    # 移除已存在的平滑注释列
    existing_cols = [f"{mid}_anno_smooth_advanced" for mid in modules_used if f"{mid}_anno_smooth_advanced" in adata.obs.columns]
    if existing_cols:
        print(f"移除已有的平滑注释列：{existing_cols}")
        adata.obs.drop(columns=existing_cols, inplace=True)
    
    # 提取空间坐标和模块初始注释、加权表达数据
    embedding_coords = adata.obsm[embedding_key]
    # 假设每个模块有字段：{module}_anno（0/1注释），{module}_exp（原始加权表达），{module}_exp_trim（修剪后的加权表达）
    module_info = {}
    for mid in modules_used:
        anno_col = f"{mid}_anno"
        exp_col = f"{mid}_exp"     # 原始加权表达
        exp_trim_col = f"{mid}_exp_trim"  # 去除低表达值后的数据
        if anno_col not in adata.obs.columns:
            raise ValueError(f"缺失字段 {anno_col}")
        if exp_col not in adata.obs.columns or exp_trim_col not in adata.obs.columns:
            raise ValueError(f"缺失加权表达信息，请检查 {exp_col} 和 {exp_trim_col}")
        module_info[mid] = {'anno': adata.obs[anno_col].copy(),
                            'exp': adata.obs[exp_col].copy(),
                            'exp_trim': adata.obs[exp_trim_col].copy()}
    
    n_cells = adata.n_obs
    # 为每个模块构建平滑结果字典
    smooth_result = {f"{mid}_anno_smooth_advanced": pd.Series(np.zeros(n_cells, dtype=int), index=adata.obs.index) for mid in modules_used}
    
    # 构建 KNN 邻域（包含自身, 故取 k_neighbors+1）
    k = k_neighbors + 1
    print(f"\n基于 {embedding_key} 坐标计算每个细胞的 {k_neighbors} 个最近邻...")
    nbrs_model = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embedding_coords)
    distances, indices = nbrs_model.kneighbors(embedding_coords)  # distances与indices均为 (n_cells, k) 矩阵
    
    # 预先计算细胞间距离用于密度估计
    # 注意：这里为了效率，我们只使用 KNN邻域的距离
    # 构造初始平滑注释字典，用于后续迭代更新
    for mid in modules_used:
        # 初始注释转换为0/1（确保 1 表示模块被注释为正）
        module_info[mid]['anno'] = (module_info[mid]['anno'] == mid).astype(int)

    # 对于每个模块分别更新注释
    for mid in modules_used:
        print(f"\n开始处理模块 {mid} ......")
        # 提取单模块的数据
        anno_vals = module_info[mid]['anno'].values.astype(float)
        # 结合原始加权表达与修剪后的表达，取二者平均或采用其他组合策略，这里采用平均值作为参考
        exp_vals = module_info[mid]['exp'].values.astype(float)
        exp_trim_vals = module_info[mid]['exp_trim'].values.astype(float)
        # 这里采用简单平均，可根据需要采用更复杂的组合策略
        orig_exp = (exp_vals + 2 * exp_trim_vals) / 3.0
        
        # 对表达水平归一化
        exp_min, exp_max = np.min(orig_exp), np.max(orig_exp)
        norm_exp = (orig_exp - exp_min) / (exp_max - exp_min + 1e-8)
        
        # 计算混合权重 w = alpha * norm_exp + (1 - alpha) * anno
        w = alpha * norm_exp + (1 - alpha) * anno_vals
        print(w.max())
        # 初始化当前模块注释更新向量（初始值取原始注释）
        updated_anno = anno_vals.copy()
        
        # 迭代更新平滑注释，直至收敛或达到最大迭代次数
        for it in range(max_iter):
            updated_anno_prev = updated_anno.copy()
            
            # 根据当前注释状态，找出被标记为1的细胞
            candidate_cells = np.where(updated_anno_prev == 1)[0]
            
            # 根据 indices 矩阵，找出所有邻域中包含 candidate_cells 的细胞。
            candidate_pool = set(candidate_cells)
            for i in range(n_cells):
                # 如果当前细胞 i 的邻域中有任一细胞属于 candidate_cells，则把 i 加入候选集
                if np.any(np.isin(indices[i, 1:], candidate_cells)):
                    candidate_pool.add(i)
            candidate_pool = np.array(list(candidate_pool)) 
            
            # 针对每个细胞计算其局部统计量
            #for i in range(n_cells):
            for i in candidate_pool:
                # 获取细胞 i 的邻域（去除自身）
                neighbor_idx = indices[i, 1:]
                neighbor_w = w[neighbor_idx]
                
                # 局部统计
                mu_local = np.mean(neighbor_w)
                sigma_local = np.std(neighbor_w)
                # 构建局部密度：利用邻域内距离的加权和
                neighbor_dists = distances[i, 1:]
                # 设 sigma_d 为邻域距离的标准差，或用户预设的值
                sigma_d = np.std(neighbor_dists) + 1e-8
                rho_local = np.mean(np.exp(- (neighbor_dists**2) / (sigma_d**2)))
                
                # 自适应阈值 T = mu + lambda * sigma，再加入局部密度修正
                T_local = (mu_local + lambda_coef * sigma_local) * (1 + beta * np.log(1 + rho_local))
                
                # 邻域投票机制：计算符合条件的邻居比例
                vote_ratio = np.mean(neighbor_w >= T_local)
                
                # 更新规则：
                # 如果细胞 i 的 w 值高于阈值，或者邻域中投票比例高于 min_votes，则置1；否则置0。
                if w[i] >= T_local or vote_ratio >= min_votes:
                    updated_anno[i] = 1
                elif updated_anno[i] == 1 :    
                    # 如果当前细胞原本被标记为1，但邻域投票比例低于阈值，则置0
                    updated_anno[i] = 1
                else:
                    updated_anno[i] = 0
            
            # 更新混合权重：重新融合新的注释信息
            w = alpha * norm_exp + (1 - alpha) * updated_anno
            
            # 检查收敛性
            delta = np.sum(np.abs(updated_anno - updated_anno_prev)) / n_cells
            print(f"模块 {mid} 第 {it+1} 次迭代，更新变化率: {delta:.4f}")
            if delta < tolerance:
                print(f"模块 {mid} 收敛于第 {it+1} 次迭代。")
                break
        
        # 形态学后处理：对更新后的注释进行简单的形态学补正
        # 这里以二维空间为例，构造一个伪图像，每个像素对应一个细胞
        # 对于复杂情况，可引入 skimage.morphology 中的开闭操作，这里给出简单模拟
        # 将1的小区域孤立点剔除；对缺口区域进行填补（本处实现较为简单，实际可扩展）
        # 由于空间位置不一定满足图像规则，需依据真实坐标投影到离散网格，下面代码为伪代码示例：
        # smooth_updated = morphology_correction(updated_anno, embedding_coords)
        # 在此示例中，我们暂不实现形态学部分，可视需要进一步拓展
        
        # 将最终注释结果存入结果字典，转换为类别类型，1对应模块名，0对应None
        smooth_series = pd.Series(updated_anno, index=adata.obs.index)
        smooth_series = smooth_series.replace({1: mid, 0: None})
        smooth_result[f"{mid}_anno_smooth_advanced"] = pd.Categorical(smooth_series)
        print(f"模块 {mid} 处理完成，初始正注释数: {int(np.sum(module_info[mid]['anno']))}, 收敛后正注释数: {int(np.sum(updated_anno))}")

    # 将所有模块的平滑注释合并到 adata.obs 中
    smooth_df = pd.DataFrame(smooth_result, index=adata.obs_names)
    adata.obs = pd.concat([adata.obs, smooth_df], axis=1)
    print("\n所有模块平滑处理完成，结果存储于 adata.obs 相应列中。")


# %%
# 测试
smooth_annotations_advanced(adata,
                            ggm_key='ggm',
                            embedding_key='spatial',
                            k_neighbors=24,
                            min_votes=0.5,
                            alpha=0.5,
                            lambda_coef=0.7,
                            beta=0.3,
                            max_iter=10,
                            tolerance=1e-3)


# %%
# 方案3
def smooth_annotations_enhanced(adata, ggm_key='ggm', modules_used=None, modules_excluded=None,
                                embedding_key='spatial', k_neighbors=24,
                                min_annotated_neighbors=1, min_neighbors_to_add=2):
    """
    平滑空间注释（增强版）。
    对每个模块的注释执行邻域支持过滤和补全，输出平滑后的注释结果。
    
    参数：
        adata : AnnData对象，包含空间转录组数据。
        ggm_key : str，adata.uns['ggm_keys']中的键，用于找到module信息。
        modules_used : list，可选，仅平滑指定的模块；默认None表示平滑所有模块。
        modules_excluded : list，可选，不平滑这些模块。
        embedding_key : str，空间坐标在adata.obsm中的键名，默认'spatial'。
        k_neighbors : int，KNN邻居数，默认24。
        min_annotated_neighbors : int，保留阳性标注所需的最少邻居数或等效权重，默认1。
        min_neighbors_to_add : int，新增标注所需的最少邻居数或等效权重，默认2。
    返回：
        None（直接在adata.obs中添加/更新“模块_anno_smooth”列）。
    """
    # 检查ggm_key合法
    if ggm_key not in adata.uns.get('ggm_keys', {}):
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_info = adata.uns['ggm_keys'][ggm_key]
    module_stats_key = mod_info.get('module_stats')
    
    # 确认空间坐标存在
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm")
    coords = adata.obsm[embedding_key]
    N = coords.shape[0]
    
    # 决定要处理的模块列表
    all_modules = adata.uns[module_stats_key]['module_id'].unique()
    if modules_used is None:
        modules_list = list(all_modules)
    else:
        modules_list = [m for m in modules_used if m in all_modules]
    if modules_excluded:
        modules_list = [m for m in modules_list if m not in modules_excluded]
    
    # 移除已有的平滑结果列，避免重复
    for m in modules_list:
        col_name = f"{m}_anno_smooth_enhanced"
        if col_name in adata.obs:
            adata.obs.drop(columns=[col_name], inplace=True)
    
    # 预计算KNN邻居索引（每个细胞的最近k_neighbors个邻居索引）
    k = k_neighbors + 1  # 包含自身
    nbrs_model = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(coords)
    _, knn_indices = nbrs_model.kneighbors(coords)
    # knn_indices[i,0]是自身，因此实际邻居从索引1开始
    knn_indices = knn_indices[:, 1:]
    
    # 初始化结果存储
    smooth_results = {}
    
    # 遍历每个模块执行平滑
    for mod in modules_list:
        # 1. 获取模块初始注释（0/1）和模块表达量
        anno_col = f"{mod}_anno"
        exp_col = f"{mod}_exp"
        if anno_col not in adata.obs or exp_col not in adata.obs:
            # 若缺少该模块的注释或表达信息，则跳过
            continue
        # 将类别型anno列转换为0/1（1表示该模块）
        a = (adata.obs[anno_col] == mod).astype(int).values  # 初始标注数组
        E = adata.obs[exp_col].values                        # 模块表达值数组
        # 从模块统计中获取阈值，如不存在则以经验值代替
        try:
            # threshold为调用阳性的表达阈值
            threshold_val = adata.uns[module_stats_key].set_index('module_id').loc[mod, 'threshold']
        except Exception as e:
            threshold_val = np.percentile(E, 90)  # 取高表达的一个分位数作为近似阈值
        # 准备数组保存阶段1和阶段2结果
        b_temp = np.zeros(N, dtype=int)
        b_final = np.zeros(N, dtype=int)
        
        # 2. 阶段1：根据邻居支持决定保留的标注
        for i in range(N):
            if a[i] == 1:
                # 计算支持分数S_i
                # 邻居权重求和，权重使用表达量相对threshold_val截断到1
                neighbors = knn_indices[i]
                # 只累加那些邻居j初始也为1的情况
                # 将所有邻居的 a[j]*min(E[j]/threshold_val, 1) 相加
                S_i = 0.0
                for j in neighbors:
                    if a[j] == 1:
                        # 标注邻居贡献权重，未标注的不贡献
                        w = E[j] / threshold_val
                        if w > 1:
                            w = 1.0
                        S_i += w
                # 判定是否保留，根据边缘与否采用不同阈值
                # 简单边缘判定：邻居标注数< min_annotated_neighbors则认为是边缘
                num_annotated_neighbors = (a[neighbors] == 1).sum()
                if num_annotated_neighbors < min_annotated_neighbors:
                    # 边缘细胞：降低要求（至少有一个标注邻居即可）
                    required = 1.0
                else:
                    required = float(min_annotated_neighbors)
                if S_i >= required:
                    b_temp[i] = 1
                else:
                    b_temp[i] = 0
            else:
                b_temp[i] = 0  # 原本未标注的直接0（阶段1不会新增）
        
        # 3. 阶段2：在阶段1基础上补全遗漏标注
        b_final[:] = b_temp  # 先拷贝阶段1的结果
        for i in range(N):
            if b_temp[i] == 0:
                # 仅考虑原本和阶段1都未标注的细胞是否可以转为1
                neighbors = knn_indices[i]
                # 计算新的支持分数 S'_i （基于阶段1结果b_temp）
                S_prime = 0.0
                for j in neighbors:
                    if b_temp[j] == 1:
                        w = E[j] / threshold_val
                        if w > 1:
                            w = 1.0
                        S_prime += w
                # 自身表达条件
                self_ok = (E[i] >= 0.5 * threshold_val)
                # 邻居数量判断用于边缘检测
                num_final_neighbors = (b_temp[neighbors] == 1).sum()
                # 若周围几乎都是该模块但自己未标注，且自身表达过半阈值，则补全
                if S_prime >= min_neighbors_to_add and self_ok:
                    # 另外防止跨模块：如果该细胞原本属于其他模块（初始a_other=1），这里不补
                    # 简化实现：若它在任何其他模块初始标注=1，则跳过
                    # （假设integrate_annotations稍后处理，此处不细查）
                    b_final[i] = 1
                # （我们暂不在代码中细分边缘内部，这里统一用min_neighbors_to_add判断）
        # 将结果保存（先存为0/1，之后再赋模块名称或None）
        smooth_results[mod] = b_final
        # 打印模块处理信息
        removed = (a.sum() - b_final.sum())
        kept = b_final.sum()
        print(f"{mod}: 移除了 {removed} 个细胞标注，保留/新增 {kept} 个细胞标注")
    
    # 4. 整理输出，将每个模块的平滑结果写入adata.obs
    for mod, b_arr in smooth_results.items():
        col_name = f"{mod}_anno_smooth_enhanced"
        # 创建分类列：1用模块名表示，0用None表示
        smooth_labels = np.where(b_arr == 1, mod, None)
        adata.obs[col_name] = pd.Categorical(smooth_labels)

    print("平滑处理完成，结果存储在adata.obs中。")


# %%
# 测试
smooth_annotations_enhanced(adata, 
                            ggm_key='ggm',
                            embedding_key='spatial',
                            k_neighbors=24,
                            min_annotated_neighbors=2,
                            min_neighbors_to_add=12)    


# %%
# 方案4
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def _calc_slice_edge_flags(coords, k, iqr_factor=1.5):
    """
    判定物理切片边缘细胞：计算每个细胞到其第 k 个邻居的平均距离，
    若该距离 > Q3 + iqr_factor*(Q3 - Q1)，则认为是切片边缘。
    返回长度 N 的布尔数组。
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_k = dists[:,1:].mean(axis=1)
    q1, q3 = np.percentile(mean_k, [25,75])
    thr = q3 + iqr_factor * (q3 - q1)
    return mean_k > thr

def _detect_module_edges(a, knn_idx, dir_vecs, min_neighbors):
    """
    检测模块边缘：对 a[i]==1 且正邻居数 < min_neighbors 的细胞，
    若这些阳性邻居集中在同一方向（比例 ≥ 0.8），则标为模块边缘。
    """
    N = len(a)
    is_edge = np.zeros(N, dtype=bool)
    bins = 8
    for i in range(N):
        if a[i] != 1:
            continue
        neigh = knn_idx[i]
        mask = (a[neigh] == 1)
        cnt = mask.sum()
        if 0 < cnt < min_neighbors:
            vecs = dir_vecs[i][mask]
            angles = (np.arctan2(vecs[:,1], vecs[:,0]) + 2*np.pi) % (2*np.pi)
            idxs = (angles / (2*np.pi/bins)).astype(int)
            counts = np.bincount(idxs, minlength=bins)
            if counts.max()/cnt >= 0.8:
                is_edge[i] = True
    return is_edge

def smooth_annotations_enhanced_2(
        adata,
        ggm_key='ggm',
        modules_used=None,
        modules_excluded=None,
        embedding_key='spatial',
        k_neighbors=24,
        min_annotated_neighbors=1,
        min_neighbors_to_add=2,
        weight_clip=1.5,
        verbose=True):
    """
    平滑空间注释（保留边缘不易删除，但正常补全所有细胞）。
    """
    # 1. 获取 module_stats
    if ggm_key not in adata.uns.get('ggm_keys', {}):
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    stats_df  = adata.uns[stats_key]

    # 2. 空间坐标与 KNN
    coords = adata.obsm.get(embedding_key)
    if coords is None:
        raise ValueError(f"{embedding_key} not in adata.obsm")
    N = coords.shape[0]
    k = k_neighbors + 1
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:,1:]
    dir_vecs = coords[:,None,:] - coords[knn_idx]
    slice_edge = _calc_slice_edge_flags(coords, k_neighbors)
    print(f"slice_edge: {slice_edge.sum()} cells are slice edges")

    # 3. 确定模块列表
    all_mods = stats_df['module_id'].values
    mods = list(all_mods) if modules_used is None else [m for m in modules_used if m in all_mods]
    if modules_excluded:
        mods = [m for m in mods if m not in modules_excluded]

    # 4. 对每个模块执行平滑
    for mod in mods:
        anno_col = f"{mod}_anno"
        exp_col  = f"{mod}_exp"
        if anno_col not in adata.obs or exp_col not in adata.obs:
            if verbose:
                print(f"skip {mod}: missing columns")
            continue

        # 4.1 初始注释与表达
        a = (adata.obs[anno_col] == mod).astype(int).values
        E = adata.obs[exp_col].values

        # 4.2 计算阈值 & 权重截断
        try:
            thr = stats_df.set_index('module_id').loc[mod, 'threshold']
            print(f"{mod}: threshold = {thr}")
        except KeyError:
            thr = np.percentile(E, 90)
        w = E / thr
        w[w > weight_clip] = weight_clip
        print(w.mean(), w.max(), thr)

        # 4.3 模块边缘检测
        mod_edge = _detect_module_edges(a, knn_idx, dir_vecs, min_annotated_neighbors)
        if verbose:
            print(f"{mod}: {mod_edge.sum()} cells are module edges")

        # 4.4 阶段1：保留（边缘细胞 required=0）
        b1 = np.zeros(N, dtype=int)
        for i in range(N):
            if a[i] != 1:
                continue
            neigh = knn_idx[i]
            S = np.sum(a[neigh] * w[neigh])
            if slice_edge[i] or mod_edge[i]:
                req = 0.0
            else:
                req = float(min_annotated_neighbors)
            b1[i] = 1 if S >= req else 0

        # 4.5 阶段2：补全（边缘也可补全）
        b2 = b1.copy()
        for i in range(N):
            if b1[i] == 1:
                continue
            cnt = np.sum(b1[knn_idx[i]])
            if cnt >= min_neighbors_to_add:
                b2[i] = 1

        # 4.6 写回结果
        out_col = f"{mod}_anno_smooth_enhanced_2"
        adata.obs[out_col] = pd.Categorical(np.where(b2, mod, None))

        if verbose:
            kept    = b1.sum()
            added   = np.sum((b2==1) & (b1==0))
            removed = np.sum((a==1) & (b2==0))
            print(f"{mod}: kept {kept}, added {added}, removed {removed}")

    if verbose:
        print("smoothing complete")



# %%
# 测试
smooth_annotations_enhanced_2(
    adata,
    ggm_key='ggm',
    modules_used=None,     
    modules_excluded=None,
    embedding_key='spatial',
    k_neighbors=18,               # 默认24邻居
    min_annotated_neighbors=2,    # 保留时至少累积权重和≥1
    min_neighbors_to_add=9,       # 补全时至少累积权重和≥3
    verbose=True
)


# %%
# 方案5
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def _calc_border_flags(coords, k, iqr_factor=1.5):
    """
    判定切片边界细胞：第 k 近邻平均距离 > Q3 + iqr_factor*(Q3-Q1) 为边界细胞。
    返回 (border_flags, knn_dists)。
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_k = dists[:,1:].mean(axis=1)
    q1, q3 = np.percentile(mean_k, [25,75])
    thr = q3 + iqr_factor * (q3 - q1)
    return mean_k > thr, dists[:,1:]

def smooth_annotations_enhanced_3(
        adata,
        ggm_key='ggm',
        modules_used=None,
        modules_excluded=None,
        embedding_key='spatial',
        k_neighbors=24,
        min_drop_neighbors=1,
        min_add_neighbors='half',
        max_weight_ratio=1.5,
        border_iqr_factor=1.5,
        border_protect_fraction=0.3,
        verbose=True):
    """
    平滑空间注释（考虑切片边界保护，按邻居数补全）。

    参数:
      adata: AnnData，需含 `<mod>_anno` 和 `<mod>_exp` 列
      ggm_key: str, adata.uns['ggm_keys'] 中的键
      modules_used: list 或 None, 要平滑的模块
      modules_excluded: list 或 None, 排除平滑的模块
      embedding_key: str, 空间坐标字段
      k_neighbors: int, KNN 邻居数（不含自身）
      min_keep_neighbors: int, 阳性细胞保留所需最少同模块邻居数
      min_add_neighbors: 'half' | 'none' | int, 补全所需邻居数（'half' = k_neighbors/2, 'none' 禁用）
      max_weight_ratio: float, 表达/阈值 最大截断倍数
      border_iqr_factor: float, 边界判定 IQR 因子
      border_protect_fraction: float, 若模块原注释中边界细胞比例 > 此值，则保护边界细胞
      verbose: bool, 是否打印进度
    """
    # 1. module_stats
    if ggm_key not in adata.uns.get('ggm_keys', {}):
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    stats_df = adata.uns[stats_key]

    # 2. coords + KNN + border flags
    coords = adata.obsm.get(embedding_key)
    if coords is None:
        raise ValueError(f"{embedding_key} not found in adata.obsm")
    N = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean').fit(coords)
    knn_dists_all, knn_idx_all = nbrs.kneighbors(coords)
    knn_idx = knn_idx_all[:,1:]
    knn_dists = knn_dists_all[:,1:]
    border_mask, _ = _calc_border_flags(coords, k_neighbors, border_iqr_factor)

    # 3. modules to process
    all_mods = stats_df['module_id'].values
    mods = list(all_mods) if modules_used is None else [m for m in modules_used if m in all_mods]
    if modules_excluded:
        mods = [m for m in mods if m not in modules_excluded]

    # 4. parse min_add_neighbors
    if isinstance(min_add_neighbors, str):
        if min_add_neighbors == 'half':
            add_thresh = max(1, k_neighbors // 2)
        elif min_add_neighbors == 'none':
            add_thresh = None
        else:
            raise ValueError("min_add_neighbors must be 'half', 'none', or a positive integer")
    else:
        add_thresh = int(min_add_neighbors)
        if add_thresh < 1:
            raise ValueError("min_add_neighbors must be >= 1 when integer")

    # 5. per-module smoothing
    for mod in mods:
        anno_col = f"{mod}_anno"
        exp_col = f"{mod}_exp"
        if anno_col not in adata.obs or exp_col not in adata.obs:
            if verbose:
                print(f"skip {mod}: missing columns")
            continue

        # original labels & expression
        a = (adata.obs[anno_col] == mod).astype(int).values
        E = adata.obs[exp_col].values

        # threshold & weight clipping
        try:
            thr = stats_df.set_index('module_id').loc[mod, 'threshold']
        except KeyError:
            thr = np.percentile(E, 90)
        w = E / thr
        w[w > max_weight_ratio] = max_weight_ratio

        # decide whether protect border cells
        total_pos = a.sum()
        frac_border = (border_mask & (a == 1)).sum() / total_pos if total_pos > 0 else 0
        protect_border = frac_border > border_protect_fraction

        # stage1: keep
        b1 = np.zeros(N, dtype=int)
        for i in range(N):
            if a[i] != 1:
                continue
            neigh = knn_idx[i]
            S = np.sum(a[neigh] * w[neigh])
            req = 0.0 if (border_mask[i] and protect_border) else float(min_drop_neighbors)
            b1[i] = 1 if S >= req else 0

        # stage2: add
        b2 = b1.copy()
        if add_thresh is not None:
            for i in range(N):
                if b1[i] == 1:
                    continue
                cnt = np.sum(b1[knn_idx[i]])
                if cnt >= add_thresh:
                    b2[i] = 1

        # write back
        out_col = f"{mod}_anno_smooth_enhanced_3"
        adata.obs[out_col] = pd.Categorical(np.where(b2, mod, None))

        if verbose:
            kept = b1.sum()
            added = np.sum((b2 == 1) & (b1 == 0))
            removed = np.sum((a == 1) & (b2 == 0))
            print(f"{mod}: kept {kept}, added {added}, removed {removed}")

    if verbose:
        print("smoothing complete")


# %%
# 测试
smooth_annotations_enhanced_3(
    adata,
    ggm_key='ggm',
    modules_used=None,     
    modules_excluded=None,
    embedding_key='spatial',
    k_neighbors=18,               # 默认24邻居
    min_drop_neighbors=1,         
    min_add_neighbors='half',     
    border_iqr_factor=1.5,      # 边界判定 IQR 因子
    border_protect_fraction=0.3, # 边界保护比例
    verbose=True
)




# %%
# 可视化
for module in adata.uns['module_stats']['module_id']:
    sc.pl.spatial(adata, size=1.6, alpha_img=0.5, frameon = False, color_map="Reds",ncols=4, 
                  color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth",
                         f"{module}_anno_smooth_advanced",f"{module}_anno_smooth_optimized", 
                         f"{module}_anno_smooth_enhanced",f"{module}_anno_smooth_enhanced_3"],
                  show=True)




# %%
# 最终方案
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
def calc_border_flags(coords, k, iqr_factor=1.5):
    """
    Identify border cells based on neighbor distances.

    Parameters:
        coords (ndarray): Spatial coordinates, shape (N, D).
        k (int): Number of neighbors to consider.
        iqr_factor (float): Multiplier for IQR when setting distance threshold.

    Returns:
        border_mask (ndarray of bool): True for border cells.
        knn_dists (ndarray): Distances to k nearest neighbors for each cell.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_d = dists[:, 1:].mean(axis=1)
    q1, q3 = np.percentile(mean_d, [25, 75])
    threshold = q3 + iqr_factor * (q3 - q1)
    border_mask = mean_d > threshold
    return border_mask, dists[:, 1:]

def smooth_annotations_enhanced(
    adata,
    ggm_key='ggm',
    modules_used=None,
    modules_excluded=None,
    embedding_key='spatial',
    k_neighbors=24,
    min_drop_neighbors=1,
    min_add_neighbors='half',
    max_weight_ratio=1.5,
    border_iqr_factor=1.5,
    border_protect_fraction=0.3,
    verbose=True
):
    """
    Smooths module annotations by dropping isolated positives and adding supported negatives.

    Parameters:
        adata (AnnData): Must contain '<module>_anno' and '<module>_exp' in .obs.
        ggm_key (str): Key under adata.uns['ggm_keys'] containing module_stats.
        modules_used (list or None): Modules to process; None for all.
        modules_excluded (list or None): Modules to skip.
        embedding_key (str): Key in adata.obsm for coordinates.
        k_neighbors (int): Number of neighbors (excluding self).
        min_drop_neighbors (int): Minimum positive neighbors to retain a positive cell.
        min_add_neighbors ('half'|'none'|int): Neighbors needed to add negatives.
        max_weight_ratio (float): Cap for exp/threshold ratio.
        border_iqr_factor (float): IQR factor for border detection.
        border_protect_fraction (float): If fraction of positives on border exceeds this,
            border positives are protected from dropping.
        verbose (bool): Print before/after counts per module.
    """
    # Load module stats
    if ggm_key not in adata.uns.get('ggm_keys', {}):
        raise KeyError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    stats_df = adata.uns[stats_key]

    # Build KNN and detect border cells
    coords = adata.obsm.get(embedding_key)
    if coords is None:
        raise KeyError(f"{embedding_key} not found in adata.obsm")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean').fit(coords)
    knn_dists_all, knn_idx_all = nbrs.kneighbors(coords)
    knn_idx = knn_idx_all[:, 1:]
    knn_dists = knn_dists_all[:, 1:]
    border_mask, _ = calc_border_flags(coords, k_neighbors, border_iqr_factor)

    # Determine modules to process
    all_mods = stats_df['module_id'].values
    mods = list(all_mods) if modules_used is None else [m for m in modules_used if m in all_mods]
    if modules_excluded:
        mods = [m for m in mods if m not in modules_excluded]
    
    # Remove existing smoothed annotation columns if they exist
    existing_columns = [f"{mid}_anno_smooth" for mid in mods if f"{mid}_anno_smooth" in adata.obs]
    if existing_columns:
        print(f"Removing existing smooth annotation columns: {existing_columns}")
        adata.obs.drop(columns=existing_columns, inplace=True)
    
    # Parse add threshold
    if isinstance(min_add_neighbors, str):
        if min_add_neighbors == 'half':
            add_thresh = max(1, k_neighbors // 2)
        elif min_add_neighbors == 'none':
            add_thresh = None
        else:
            raise ValueError("min_add_neighbors must be 'half', 'none', or int")
    else:
        add_thresh = int(min_add_neighbors)
        if add_thresh < 1:
            raise ValueError("min_add_neighbors must be >=1 when integer")

    # Process each module
    for mod in mods:
        anno_col = f"{mod}_anno"
        exp_col = f"{mod}_exp"
        if anno_col not in adata.obs or exp_col not in adata.obs:
            if verbose:
                print(f"skip {mod}: missing anno/exp")
            continue

        a = (adata.obs[anno_col] == mod).astype(int).values
        E = adata.obs[exp_col].values

        # Compute weight and cap
        try:
            thr = stats_df.set_index('module_id').loc[mod, 'threshold']
        except KeyError:
            thr = np.percentile(E, 90)
        w = E / thr
        w = np.minimum(w, max_weight_ratio)

        # Determine border protection
        total_pos = a.sum()
        frac_border = (border_mask & (a == 1)).sum() / total_pos if total_pos > 0 else 0
        protect_border = frac_border > border_protect_fraction

        # Stage 1: drop isolated positives
        b1 = np.zeros_like(a)
        for i in range(len(a)):
            if a[i] != 1:
                continue
            neigh = knn_idx[i]
            support = np.sum(a[neigh] * w[neigh])
            required = 0.0 if (border_mask[i] and protect_border) else float(min_drop_neighbors)
            b1[i] = 1 if support >= required else 0

        # Stage 2: add supported negatives
        b2 = b1.copy()
        if add_thresh is not None:
            for i in range(len(a)):
                if b1[i] == 1:
                    continue
                cnt = np.sum(b1[knn_idx[i]])
                if cnt >= add_thresh:
                    b2[i] = 1

        # Write smoothed annotation
        out_col = f"{mod}_anno_smooth"
        adata.obs[out_col] = pd.Categorical(np.where(b2, mod, None))

        if verbose:
            after = b2.sum()
            print(f"{mod} processed. remain cells: {after}")


    if verbose:
        print("\nAnnotation smoothing completed. Results stored in adata.obs.\n")


# %%
# 测试
smooth_annotations_enhanced(
    adata,
    ggm_key='ggm',
    modules_used=None,     
    modules_excluded=None,
    embedding_key='spatial',
    k_neighbors=18,               # 默认24邻居
    min_drop_neighbors=1,         
    min_add_neighbors='half',     
    border_iqr_factor=1.5,      # 边界判定 IQR 因子
    border_protect_fraction=0.3, # 边界保护比例
    verbose=True
)


# %%
# 可视化
for module in adata.uns['module_stats']['module_id']:
    sc.pl.spatial(adata, size=1.6, alpha_img=0.5, frameon = False, color_map="Reds",ncols=4, 
                  color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth",
                         f"{module}_anno_smooth_advanced",f"{module}_anno_smooth_optimized", 
                         f"{module}_anno_smooth_enhanced",f"{module}_anno_smooth_enhanced_3"],
                  show=True)







# %%
pdf_file = "figures/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_smooth_methods.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in adata.uns['module_stats']['module_id']:
    plt.figure()    
    sc.pl.spatial(adata, size=1.6, alpha_img=0.5, frameon = False, color_map="Reds", ncols=3, 
                  color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth",
                         f"{module}_anno_smooth_advanced",f"{module}_anno_smooth_optimized", f"{module}_anno_smooth_enhanced"],
                  show=False)
    show_png_file = f"figures/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_smooth_methods_{module}.png"
    plt.savefig(show_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(show_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()

c.save()
# %%
