
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
# 计算模块表达值
start_time = time.time()
sg.calculate_module_expression(adata, 
                               ggm_obj=ggm, 
                               top_genes=30,
                               weighted=True,
                               calculate_moran=True,
                               embedding_key='spatial',
                               k_neighbors=6,
                               add_go_anno=5)  
print(f"Time1: {time.time() - start_time:.5f} s")


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
                            random_state=42,
                            embedding_key='spatial',
                            k_neighbors=6
                            )
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])


# %%
adata.uns['module_stats'].to_csv("data/module_stats.csv", index=False)

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
adata.uns['module_stats'].to_csv("data/module_stats.csv", index=False)


# %%
# 开发模块质量评估函数
import scanpy as sc
import pandas as pd
import numpy as np

def classify_modules(adata, 
                     ref_cluster_key=None,
                     ref_cluster_method='leiden', 
                     ref_cluster_resolution=1.0, 
                     skew_threshold=2.0,
                     top1pct_threshold=2.0,
                     Moran_I_threshold=0.2,
                     cluster_similarity_threshold=0.3,
                     jaccard_threshold=0.6):
    """
    根据模块统计和注释信息，对空间转录组模块进行分类筛选，标记哪些模块可作为身份模块用于细胞注释整合。
    
    参数：
    - adata : AnnData对象，包含空间转录组数据以及模块分析结果（adata.uns['module_stats'], adata.obsm['module_expression'], adata.obs 中的模块注释列）。
    - ref_cluster_key : str，聚类标签的键名（默认 None），如果提供则直接使用该列作为聚类标签。
    - ref_cluster_method : {'leiden', 'louvain'} 或者 str，聚类算法名称（默认 'leiden'）。如果提供其他字符串且该字符串存在于 adata.obs，则直接使用该列作为聚类标签。
    - ref_cluster_resolution : float，聚类分辨率参数（默认1.0），仅当使用 leiden 或 louvain 算法时有效。
    - jaccard_threshold : float，判定“注释相似模块”的 Jaccard 重叠阈值（默认0.6）。
    
    执行后将在 adata.uns['module_filtering'] 保存结果 DataFrame，
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
        
    # 检查是否存在 GO 注释信息（top_go_terms 列）
    has_go_info = "top_go_terms" in mod_stats_df.columns

    # 检查模块ID列表
    module_ids = mod_stats_df['module_id'].tolist()
    
    # 1. 生成/获取空间聚类标签
    if ref_cluster_key is None:
        if ref_cluster_method in adata.obs.columns:
            # 如果指定的 ref_cluster_method 是现成的 obs 列
            ref_cluster_key = ref_cluster_method
        else:
            # 否则使用 leiden 或 louvain 算法进行聚类
            if ref_cluster_method.lower() == 'leiden':
                ref_cluster_key = 'tmp_leiden_for_filtering'
                sc.tl.leiden(adata, resolution=ref_cluster_resolution, key_added=ref_cluster_key)
            elif ref_cluster_method.lower() == 'louvain':
                ref_cluster_key = 'tmp_louvain_for_filtering'
                sc.tl.louvain(adata, resolution=ref_cluster_resolution, key_added=ref_cluster_key)
            else:
                ref_cluster_key = None
    else:
        if ref_cluster_key not in adata.obs.columns:
            raise ValueError(f"聚类标签列 {ref_cluster_key} 不存在于 adata.obs")
        
    # 获取每个模块对应的细胞集合（索引集合）以及细胞计数
    module_cells = {}  # 模块 -> 细胞索引集合
    module_cell_counts = {}  # 模块 -> 细胞数
    for module_id in module_ids:
        col = f"{module_id}_anno"
        if col not in adata.obs.columns:
            module_cells[module_id] = set()
            module_cell_counts[module_id] = 0
        else:
            # 选择该列非空的细胞索引作为模块细胞集合
            cells = adata.obs[~adata.obs[col].isna()].index
            module_cells[module_id] = set(cells)
            module_cell_counts[module_id] = len(cells)
    # 按细胞数目降序排序模块列表
    modules_sorted = sorted(module_ids, key=lambda m: module_cell_counts.get(m, 0), reverse=True)
    
    # 定义 Activity 模块判定的 GO 关键词集合
    activity_keywords = ["proliferation", "cell cycle", "cell division",
                         "DNA replication", "RNA processing", "translation",
                         "metabolic process", "biosynthetic process", "ribosome",
                         "chromosome segregation", "spindle", "mitotic", "cell growth"]
    activity_keywords = [kw.lower() for kw in activity_keywords]
    
    # 初始化结果列表
    results = []  # 每个元素是字典 {module_id, is_identity, type_tag, reason}
    is_identity_dict = {}
    type_tag_dict = {}
    reason_dict = {}
    
    # 2. 第一遍遍历：应用判定规则1-4
    for mod in modules_sorted:
        # 获取模块的统计指标：偏度、top1pct_ratio、Moran 指数
        if 'skew' in mod_stats_df.columns:
            skewness = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'skew'].iloc[0]) \
                       if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'skew'])
        else:
            skewness = None
        if 'top1pct_ratio' in mod_stats_df.columns:
            top1pct = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'top1pct_ratio'].iloc[0]) \
                      if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'top1pct_ratio'])
        else:
            top1pct = None
        if 'positive_moran_I' in mod_stats_df.columns:
            moranI = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'positive_moran_I'].iloc[0]) \
                     if 'module_id' in mod_stats_df.columns else float(mod_stats_df.loc[mod, 'positive_moran_I'])
        else:
            moranI = None
        
        # 先假定模块为身份模块
        is_identity = True
        type_tag = 'identity_module'
        reason = ''
        
        # (1) Activity 模块判定：利用 GO 注释信息
        is_activity = False
        if has_go_info:
            # 如果 mod_stats_df 中存在 top_go_terms 列，则直接读取
            go_terms_str = mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'top_go_terms']
            if not go_terms_str.empty:
                go_terms_str = go_terms_str.iloc[0]
                if pd.notnull(go_terms_str):
                    # 按 “ || ” 拆分，得到 GO 术语列表
                    terms = [t.strip().lower() for t in go_terms_str.split("||")]
                    for term in terms:
                        if any(kw in term for kw in activity_keywords):
                            is_activity = True
                            break
        else:
            # 若没有 top_go_terms 信息，尝试从 adata.uns['go_enrichment'] 获取
            if hasattr(adata, 'uns') and 'go_enrichment' in getattr(adata, 'uns', {}):
                go_info = adata.uns['go_enrichment']
            elif 'ggm' in adata.uns and hasattr(adata.uns['ggm'], 'go_enrichment'):
                go_info = adata.uns['ggm'].go_enrichment
            else:
                go_info = None
            if go_info is not None:
                terms = None
                if isinstance(go_info, dict):
                    terms = go_info.get(mod)
                elif isinstance(go_info, pd.DataFrame):
                    if 'module_id' in go_info.columns:
                        terms = go_info[go_info['module_id'] == mod]
                        terms = [str(t).lower() for t in terms['term']] if 'term' in go_info.columns else None
                if terms:
                    for term in terms:
                        if any(kw in term.lower() for kw in activity_keywords):
                            is_activity = True
                            break
        if is_activity:
            is_identity = False
            type_tag = 'activity_module'
            reason = '模块 GO 注释显示与细胞活动相关（增殖/代谢等），非特定细胞类型标志'
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'reason': reason
            })
            continue  # 直接跳到下一个模块
        
        # (2) 空间泛滥模块判定
        if skewness is not None and top1pct is not None:
            if skewness < skew_threshold and top1pct < top1pct_threshold:
                is_identity = False
                type_tag = 'spatially_pervasive_module'
                reason = '模块表达广泛分布于多数细胞（偏度和top1%细胞表达水平低）'
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
            if moranI < Moran_I_threshold and moranI > 0:
                is_identity = False
                type_tag = 'weak_spatial_module'
                reason = '模块 Moran’s I 指数低，空间聚集性弱'
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
        if ref_cluster_key is not None and module_cell_counts.get(mod, 0) > 0:
            cells = module_cells.get(mod, set())
            clusters = adata.obs.loc[list(cells), ref_cluster_key] if ref_cluster_key in adata.obs.columns else None
            if clusters is not None:
                cluster_counts = clusters.value_counts()
                if len(cluster_counts) > 1:
                    total = cluster_counts.sum()
                    max_frac = cluster_counts.max() / total
                    if max_frac < cluster_similarity_threshold:
                        is_identity = False
                        type_tag = 'cross_structure_module'
                        reason = '模块细胞分布于多个空间聚类，未集中在单一结构域'
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
    identity_modules = [mod for mod in modules_sorted if is_identity_dict.get(mod, False)]
    identity_modules = sorted(identity_modules)
    marked_as_similar = set()
    for i, modA in enumerate(identity_modules):
        for modB in identity_modules[i+1:]:
            if modA in marked_as_similar or modB in marked_as_similar:
                continue
            cellsA = module_cells.get(modA, set())
            cellsB = module_cells.get(modB, set())
            if not cellsA or not cellsB:
                continue
            inter = len(cellsA & cellsB)
            union = len(cellsA | cellsB)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= jaccard_threshold:
                # 比较两个模块的效应量决定保留哪一个（效应量越大，表明模块区分度越好）
                keep_mod = modA
                drop_mod = modB
                effect_available = False
                try:
                    effA = float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'effect_size'].iloc[0])
                    effB = float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'effect_size'].iloc[0])
                    if not (np.isnan(effA) or np.isnan(effB)):
                        effect_available = True
                except Exception:
                    effA = effB = None
                if effect_available:
                    # 如果两个模块均有 effect_size，则较小者被认为效果较差，作为冗余模块
                    if effA < effB:
                        keep_mod, drop_mod = modB, modA
                    else:
                        keep_mod, drop_mod = modA, modB
                else:
                    # 最后使用偏度和 top1pct_ratio 作为评分
                    scoreA = 0
                    scoreB = 0
                    if modA in type_tag_dict and modB in type_tag_dict:
                        scoreA = (float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'skew'].iloc[0]) if 'skew' in mod_stats_df.columns else 0) + \
                                (float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'top1pct_ratio'].iloc[0]) if 'top1pct_ratio' in mod_stats_df.columns else 0)
                        scoreB = (float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'skew'].iloc[0]) if 'skew' in mod_stats_df.columns else 0) + \
                                (float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'top1pct_ratio'].iloc[0]) if 'top1pct_ratio' in mod_stats_df.columns else 0)
                    if scoreB > scoreA:
                        keep_mod, drop_mod = modB, modA
                is_identity_dict[drop_mod] = False
                type_tag_dict[drop_mod] = 'annotation_similar_module'
                reason_dict[drop_mod] = f'与模块 {keep_mod} 注释重叠，高度相似'
                marked_as_similar.add(drop_mod)
                for rec in results:
                    if rec['module_id'] == drop_mod:
                        rec['is_identity'] = False
                        rec['type_tag'] = 'annotation_similar_module'
                        rec['reason'] = reason_dict[drop_mod]
                        break
                else:
                    results.append({
                        'module_id': drop_mod,
                        'is_identity': False,
                        'type_tag': 'annotation_similar_module',
                        'reason': reason_dict[drop_mod]
                    })
    # 4. 整理输出结果为 DataFrame
    result_df = pd.DataFrame(results)
    if 'module_id' in result_df.columns:
        result_df = result_df.sort_values(by='module_id').reset_index(drop=True)
    adata.uns['module_filtering'] = result_df
    
    # 5. 更新 adata.uns['module_stats'] 中的 is_identity 和 type_tag 列
    if 'module_id' in mod_stats_df.columns:
        mod_stats_df['is_identity'] = mod_stats_df['module_id'].map(is_identity_dict).fillna(False)
        mod_stats_df['type_tag'] = mod_stats_df['module_id'].map(type_tag_dict).fillna('filtered_module')
        adata.uns['module_stats'] = mod_stats_df
    else:
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




# %%









# %%
# 测试
class_info = classify_modules(adata, 
                              ref_cluster_key='graph_cluster',
                              #ref_cluster_method='leiden', 
                              #ref_cluster_method='none',  
                              ref_cluster_resolution=0.1, 
                              jaccard_threshold=0.3)

# %%
adata.uns['module_filtering'].to_csv("data/module_filtering.csv", index=False)

# %%
adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity'] == True]['module_id']

# %%
start_time = time.time()
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity'] == True]['module_id'],
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
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)
# %%
