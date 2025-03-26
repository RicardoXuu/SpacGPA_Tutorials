
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
ggm = sg.load_ggm("data/ggm_gpu_32.h5")
print(f"Read ggm: {time.time() - start_time:.5f} s")
# 读取联合分析的ggm
ggm_mulit_intersection = sg.load_ggm("data/ggm_mulit_intersection.h5")
print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
ggm_mulit_union = sg.load_ggm("data/ggm_mulit_union.h5")
print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
print("=====================================")
print(ggm)
print("=====================================")
print(ggm_mulit_intersection)
print("=====================================")
print(ggm_mulit_union)

# %%
adata = sc.read("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno_union_intersection.h5ad")
adata

# %%
# 开发模块质量评估函数
import scanpy as sc
import pandas as pd
import numpy as np

def classify_modules(adata, cluster_method='leiden', resolution=1.0, jaccard_threshold=0.6):
    """
    根据模块统计和注释信息，对空间转录组模块进行分类筛选，标记哪些模块可作为身份模块用于细胞注释整合。
    
    参数：
    - adata : AnnData对象，包含空间转录组数据以及模块分析结果（adata.uns['module_stats'], adata.obsm['module_expression'], adata.obs 中的模块注释列，ggm.go_enrichment等）。
    - cluster_method : {'leiden', 'louvain'} 或者 str，聚类算法名称（默认 'leiden'）。如果提供其他字符串且该字符串存在于 adata.obs，则直接使用该列作为聚类标签。
    - resolution : float，聚类分辨率参数（默认1.0），仅当使用 leiden 或 louvain 算法时有效。
    - jaccard_threshold : float，判定“注释相似模块”的 Jaccard 重叠阈值（默认0.6）。
    
    此函数执行后，会在 adata.uns['module_filtering'] 保存结果DataFrame，
    每行包含 module_id, is_identity, type_tag, reason 四个字段，
    同时在 adata.uns['module_stats'] 对应模块增加 is_identity 和 type_tag 字段。
    """
    # 提取模块统计信息表
    module_stats = adata.uns.get('module_stats')
    if module_stats is None:
        raise ValueError("adata.uns['module_stats'] 未找到模块统计信息")
    # 将 module_stats 转为 DataFrame（如果不是的话）
    if isinstance(module_stats, pd.DataFrame):
        mod_stats_df = module_stats.copy()
    else:
        mod_stats_df = pd.DataFrame(module_stats)
    # 检查模块ID列表
    if 'module_id' in mod_stats_df.columns:
        module_ids = mod_stats_df['module_id'].tolist()
    else:
        # 如果没有显式module_id列，尝试使用索引作为模块ID
        module_ids = list(mod_stats_df.index)
        # 若索引是数字，则格式化为字符串如 "M01"
        if isinstance(module_ids[0], int) or module_ids[0].isdigit():
            module_ids = [f"M{int(i):02d}" for i in module_ids]
            mod_stats_df['module_id'] = module_ids
    
    # 1. 生成/获取空间聚类标签
    cluster_key = None
    if cluster_method in adata.obs.columns:
        # 如果指定的cluster_method是现成的obs列
        cluster_key = cluster_method
    else:
        # 否则使用leiden或louvain算法进行聚类
        if cluster_method.lower() == 'leiden':
            cluster_key = 'tmp_leiden'
            sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
        elif cluster_method.lower() == 'louvain':
            cluster_key = 'tmp_louvain'
            sc.tl.louvain(adata, resolution=resolution, key_added=cluster_key)
        else:
            # 既不是已有列也不是指定算法
            cluster_key = None
    
    # 获取每个模块对应的细胞集合（索引集合）以及细胞计数
    module_cells = {}  # 模块 -> 细胞索引集合
    module_cell_counts = {}  # 模块 -> 细胞数
    for module_id in module_ids:
        col = f"{module_id}_anno"
        if col not in adata.obs.columns:
            # 如果对应的注释列不存在，则跳过或记录空
            module_cells[module_id] = set()
            module_cell_counts[module_id] = 0
        else:
            # 选择该列非空的细胞索引作为模块细胞集合
            cells = adata.obs[~adata.obs[col].isna()].index
            module_cells[module_id] = set(cells)
            module_cell_counts[module_id] = len(cells)
    # 按细胞数目降序排序模块列表
    modules_sorted = sorted(module_ids, key=lambda m: module_cell_counts.get(m, 0), reverse=True)
    
    # 定义Activity模块判定的GO术语关键词集合
    activity_keywords = ["proliferation", "cell cycle", "cell division",
                         "DNA replication", "RNA processing", "translation",
                         "metabolic process", "biosynthetic process", "ribosome",
                         "chromosome segregation", "spindle", "mitotic", "cell growth"]
    activity_keywords = [kw.lower() for kw in activity_keywords]
    
    # 初始化结果列表
    results = []  # 每个元素是字典 {module_id, is_identity, type_tag, reason}
    # 用字典暂存分类标记，便于后续查找和更新
    is_identity_dict = {}
    type_tag_dict = {}
    reason_dict = {}
    
    # 2. 第一遍遍历：应用判定规则1-4
    for mod in modules_sorted:
        # 获取模块的统计指标
        # 注意：偏度、top1pct_ratio、Moran指数等可能在module_stats DataFrame中
        if 'skewness' in mod_stats_df.columns:
            skewness = float(mod_stats_df.loc[mod_stats_df['module_id']==mod, 'skewness']) \
                       if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'skewness'])
        else:
            skewness = None
        if 'top1pct_ratio' in mod_stats_df.columns:
            top1pct = float(mod_stats_df.loc[mod_stats_df['module_id']==mod, 'top1pct_ratio']) \
                      if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'top1pct_ratio'])
        else:
            top1pct = None
        if 'MoranI' in mod_stats_df.columns:
            moranI = float(mod_stats_df.loc[mod_stats_df['module_id']==mod, 'MoranI']) \
                     if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'MoranI'])
        elif 'Moran' in mod_stats_df.columns:
            moranI = float(mod_stats_df.loc[mod_stats_df['module_id']==mod, 'Moran']) \
                     if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'Moran'])
        else:
            moranI = None
        
        # 先假定模块为身份模块，后面根据规则修改
        is_identity = True
        type_tag = 'identity_module'
        reason = ''
        
        # (1) Activity模块判定
        # 检查GO富集是否含有活动相关关键词
        is_activity = False
        if hasattr(adata, 'uns') and 'go_enrichment' in getattr(adata, 'uns', {}):
            go_info = adata.uns['go_enrichment']
        elif 'ggm' in adata.uns and hasattr(adata.uns['ggm'], 'go_enrichment'):
            go_info = adata.uns['ggm'].go_enrichment
        else:
            go_info = None
        if go_info is not None:
            # 获取该模块的GO术语列表（根据模块ID作为键）
            terms = None
            if isinstance(go_info, dict):
                terms = go_info.get(mod)
            elif isinstance(go_info, pd.DataFrame):
                # 若是DataFrame，筛选出该模块的条目
                if 'module_id' in go_info.columns:
                    terms = go_info[go_info['module_id'] == mod]
                    # 提取术语名称列表
                    terms = [str(t).lower() for t in terms['term']] if 'term' in go_info.columns else None
            # 检查关键词
            if terms:
                for term in terms:
                    term_lower = term.lower()
                    if any(kw in term_lower for kw in activity_keywords):
                        is_activity = True
                        break
        if is_activity:
            is_identity = False
            type_tag = 'activity_module'
            reason = '模块功能富集显示与细胞活动相关（增殖/代谢等），非特定细胞类型标志'
            # 记录结果并跳过后续判断
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'reason': reason
            })
            continue  # 跳到下一个模块
        
        # (2) 空间泛滥模块判定
        if skewness is not None and top1pct is not None:
            if skewness < 2 and top1pct < 2:
                is_identity = False
                type_tag = 'spatially_pervasive_module'
                reason = '模块表达广泛分布于多数细胞（偏度和top1%比例低）'
        # 如果已经标记为非身份，则跳过后续判断
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'reason': reason
            })
            continue
        
        # (3) 弱空间自相关模块判定
        if moranI is not None:
            if moranI < 0.3 and moranI >= 0:
                is_identity = False
                type_tag = 'weak_spatial_module'
                reason = '模块 Moran\u2019s I 空间自相关指数低，空间聚集性弱'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'reason': reason
            })
            continue
        
        # (4) 跨结构域表达模块判定
        if cluster_key is not None and module_cell_counts.get(mod, 0) > 0:
            cells = module_cells.get(mod, set())
            clusters = adata.obs.loc[cells, cluster_key] if cluster_key in adata.obs.columns else None
            if clusters is not None:
                # 统计这些细胞所属的聚类类别
                cluster_counts = clusters.value_counts()
                if len(cluster_counts) > 1:
                    total = cluster_counts.sum()
                    max_frac = cluster_counts.max() / total
                    # 若不止一个聚类，并且最大聚类占比小于阈值（如0.8），则认为跨域
                    if max_frac < 0.8:
                        is_identity = False
                        type_tag = 'cross_structure_module'
                        reason = '模块细胞分布于多个空间聚类，未集中在单一结构域'
        # 记录跨域判定结果
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'reason': reason
            })
            continue
        
        # 如果通过所有筛选，则标记为身份模块
        is_identity = True
        type_tag = 'identity_module'
        reason = '通过所有筛选条件，作为细胞类型标志模块'
        is_identity_dict[mod] = is_identity
        type_tag_dict[mod] = type_tag
        reason_dict[mod] = reason
        results.append({
            'module_id': mod,
            'is_identity': is_identity,
            'type_tag': type_tag,
            'reason': reason
        })
    
    # 3. 第二遍遍历：注释相似模块判定
    # 获取仍标记为身份模块的列表
    identity_modules = [mod for mod in modules_sorted if is_identity_dict.get(mod, False)]
    # 为避免重复比较，对列表排序（例如按模块ID）确保结果可预期
    identity_modules = sorted(identity_modules)
    # 计算模块两两之间的Jaccard相似度并标记冗余模块
    marked_as_similar = set()
    for i, modA in enumerate(identity_modules):
        for modB in identity_modules[i+1:]:
            if modA in marked_as_similar or modB in marked_as_similar:
                continue  # 已经被标记为冗余的模块跳过
            cellsA = module_cells.get(modA, set())
            cellsB = module_cells.get(modB, set())
            if not cellsA or not cellsB:
                continue
            # 计算Jaccard指数
            inter = len(cellsA & cellsB)
            union = len(cellsA | cellsB)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= jaccard_threshold:
                # 高度重叠，决定保留哪个模块
                # 这里通过比较两个模块在各自细胞集上的平均表达强度或分布指标
                keep_mod = modA
                drop_mod = modB
                # 比较平均模块表达（使用module_expression加权表达矩阵）
                if 'module_expression' in adata.obsm:
                    mod_expr = adata.obsm['module_expression']
                    if isinstance(mod_expr, (pd.DataFrame, pd.Series)):
                        # 如果module_expression是DataFrame，可能列名或索引包含模块
                        if modA in mod_expr.columns and modB in mod_expr.columns:
                            meanA = np.nanmean(mod_expr.loc[cellsA, modA]) if len(cellsA) > 0 else 0
                            meanB = np.nanmean(mod_expr.loc[cellsB, modB]) if len(cellsB) > 0 else 0
                        else:
                            # 如果module_expression按顺序对应模块，可以利用索引
                            # 这里假设模块顺序与module_ids相同
                            try:
                                idxA = module_ids.index(modA)
                                idxB = module_ids.index(modB)
                                meanA = np.nanmean(mod_expr[:, idxA][list(cellsA)])
                                meanB = np.nanmean(mod_expr[:, idxB][list(cellsB)])
                            except Exception:
                                meanA = meanB = None
                    else:
                        # 假定是numpy数组，列序与module_ids对应
                        try:
                            idxA = module_ids.index(modA)
                            idxB = module_ids.index(modB)
                            meanA = np.nanmean(adata.obsm['module_expression'][list(cellsA), idxA]) if len(cellsA)>0 else 0
                            meanB = np.nanmean(adata.obsm['module_expression'][list(cellsB), idxB]) if len(cellsB)>0 else 0
                        except Exception:
                            meanA = meanB = None
                else:
                    meanA = meanB = None
                if meanA is not None and meanB is not None:
                    # 如果都有平均表达值，选择更高者保留
                    if meanB > meanA:
                        keep_mod, drop_mod = modB, modA
                else:
                    # 若无法比较平均表达，则根据细胞数和分布指标简单选择
                    # 可选择偏度+top1pct作为评分
                    scoreA = 0
                    scoreB = 0
                    if modA in type_tag_dict and modB in type_tag_dict:
                        # 直接用偏度和top1pct分数（之前已提取）
                        scoreA = (float(mod_stats_df.loc[mod_stats_df['module_id']==modA, 'skewness']) if 'skewness' in mod_stats_df.columns else 0) \
                               + (float(mod_stats_df.loc[mod_stats_df['module_id']==modA, 'top1pct_ratio']) if 'top1pct_ratio' in mod_stats_df.columns else 0)
                        scoreB = (float(mod_stats_df.loc[mod_stats_df['module_id']==modB, 'skewness']) if 'skewness' in mod_stats_df.columns else 0) \
                               + (float(mod_stats_df.loc[mod_stats_df['module_id']==modB, 'top1pct_ratio']) if 'top1pct_ratio' in mod_stats_df.columns else 0)
                    if scoreB > scoreA:
                        keep_mod, drop_mod = modB, modA
                # 将drop_mod标记为非身份模块，类型为annotation_similar
                is_identity_dict[drop_mod] = False
                type_tag_dict[drop_mod] = 'annotation_similar_module'
                reason_dict[drop_mod] = f'与模块{keep_mod}注释重叠，高度相似'
                marked_as_similar.add(drop_mod)
                # 更新结果列表中对应drop_mod的记录
                for rec in results:
                    if rec['module_id'] == drop_mod:
                        rec['is_identity'] = False
                        rec['type_tag'] = 'annotation_similar_module'
                        rec['reason'] = reason_dict[drop_mod]
                        break
                else:
                    # 如果不在结果列表（可能因为之前标为identity加入了），则添加新记录
                    results.append({
                        'module_id': drop_mod,
                        'is_identity': False,
                        'type_tag': 'annotation_similar_module',
                        'reason': reason_dict[drop_mod]
                    })
    # end of pairwise similarity loop
    
    # 4. 整理输出结果为 DataFrame
    result_df = pd.DataFrame(results)
    # 若存在module_id列，按模块ID排序
    if 'module_id' in result_df.columns:
        result_df = result_df.sort_values(by='module_id').reset_index(drop=True)
    adata.uns['module_filtering'] = result_df
    
    # 5. 更新 adata.uns['module_stats'] 中的 is_identity 和 type_tag 列
    # 确保 module_stats DataFrame 有 module_id 列并与 result_df 对接
    if 'module_id' in mod_stats_df.columns:
        # 根据 module_id 列进行映射更新
        mod_stats_df['is_identity'] = mod_stats_df['module_id'].map(is_identity_dict).fillna(False)
        mod_stats_df['type_tag'] = mod_stats_df['module_id'].map(type_tag_dict).fillna('filtered_module')
        # 将更新结果写回 adata.uns['module_stats']
        adata.uns['module_stats'] = mod_stats_df
    else:
        # 如果 module_id 不在列而是在索引：
        new_cols = []
        for idx in mod_stats_df.index:
            mod_id = str(idx)
            if idx in is_identity_dict:
                new_cols.append((is_identity_dict[idx], type_tag_dict.get(idx, 'filtered_module')))
            elif mod_id in is_identity_dict:
                new_cols.append((is_identity_dict[mod_id], type_tag_dict.get(mod_id, 'filtered_module')))
            else:
                new_cols.append((False, 'filtered_module'))
        mod_stats_df['is_identity'] = [col[0] for col in new_cols]
        mod_stats_df['type_tag'] = [col[1] for col in new_cols]
        adata.uns['module_stats'] = mod_stats_df
    
    return result_df
