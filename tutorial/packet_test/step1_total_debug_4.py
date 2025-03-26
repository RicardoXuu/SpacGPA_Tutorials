
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
                               ggm_obj=ggm_mulit_intersection,
                               ggm_key='intersection', 
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
                            ggm_key='intersection',
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
print(adata.uns['intersection_module_stats'])


# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="intersection_M001_anno", show=True)

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                        ggm_key='intersection',
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
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="intersection_M001_anno_smooth", show=True)


# %%
# 开发模块质量评估函数
import scanpy as sc
import pandas as pd
import numpy as np

def classify_modules(adata, 
                     ggm_key='ggm',
                     ref_anno=None,
                     ref_cluster_method='leiden', 
                     ref_cluster_resolution=1.0, 
                     skew_threshold=2.0,
                     top1pct_threshold=2.0,
                     Moran_I_threshold=0.2,
                     min_dominant_cluster_fraction=0.3,
                     anno_overlap_threshold=0.6):
    """
    Classify spatial specificity modules based on module statistics and annotation data.
    
    This function uses module-level statistics (stored in adata.uns[mod_stats_key]),
    module expression (adata.obsm['module_expression']), and annotation columns (e.g., "M01_anno")
    to determine which modules serve as robust markers for cell identity.
    
    Parameters:
      adata : AnnData object containing spatial transcriptomics data along with:
              - Module statistics in adata.uns[mod_stats_key]
              - Module expression matrix in adata.obsm['module_expression']
              - Module annotation columns in adata.obs (e.g., "M01_anno")
      ggm_key : str, key in adata.uns for the GGM object.
      ref_anno : str, key in adata.obs for reference cluster labels; if provided, this column is used.
      ref_cluster_method : str, clustering method to use if ref_anno is not provided (e.g., 'leiden' or 'louvain').
      ref_cluster_resolution : float, resolution parameter for clustering.
      skew_threshold : float, threshold for skewness to flag modules with ubiquitous expression.
      top1pct_threshold : float, threshold for the top 1% expression ratio to flag modules with ubiquitous expression.
      Moran_I_threshold : float, threshold for positive Moran's I to flag diffuse (weakly spatial) modules.
      min_dominant_cluster_fraction : float, the minimum fraction of a module's annotated cells that must be concentrated 
                                      in one reference cluster to avoid being flagged as mixed-regional.
      anno_overlap_threshold : float, the Jaccard index threshold above which two modules are considered redundant.
       
    The function updates adata.uns['module_filtering'] with a DataFrame containing:
      - module_id: Module identifier.
      - is_identity: Boolean flag indicating whether the module is suitable as a cell identity marker.
      - type_tag: Category tag, one of:
          * "cellular_activity_module"
          * "ubiquitous_module"
          * "diffuse_module"
          * "mixed_regional_module"
          * "redundant_module"
          * "cell_identity_module"
      - information: A brief explanation for exclusion/inclusion.
      
    Also, adata.uns[mod_stats_key] is updated with 'is_identity' and 'type_tag' for each module.
    
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    mod_filtering_key = adata.uns['ggm_keys'][ggm_key]['module_filtering']
    # Retrieve module statistics
    module_stats = adata.uns.get(mod_stats_key)
    if module_stats is None:
        raise ValueError("Module statistics not found in adata.uns[mod_stats_key]")
    mod_stats_df = module_stats if isinstance(module_stats, pd.DataFrame) else pd.DataFrame(module_stats)
    
    # Check if GO annotation info exists (i.e. 'top_go_terms' column)
    has_go_info = "top_go_terms" in mod_stats_df.columns

    # Get list of module IDs
    module_ids = mod_stats_df['module_id'].tolist()
    
    # (1) Obtain spatial clustering labels
    if ref_anno is None:
        if ref_cluster_method.lower() == 'leiden':
            ref_anno = 'tmp_leiden_for_filtering'
            sc.tl.leiden(adata, resolution=ref_cluster_resolution, key_added=ref_anno)
        elif ref_cluster_method.lower() == 'louvain':
            ref_anno = 'tmp_louvain_for_filtering'
            sc.tl.louvain(adata, resolution=ref_cluster_resolution, key_added=ref_anno)
        else:
            print(f"Unknown clustering method '{ref_cluster_method}'; skipping cluster assignment")
            ref_anno = None
    else:
        if ref_anno not in adata.obs.columns:
            raise ValueError(f"Cluster label column '{ref_anno}' not found in adata.obs")
        
    # (2) For each module, get the set of annotated cell indices and counts
    module_cells = {}
    module_cell_counts = {}
    for module_id in module_ids:
        col = f"{module_id}_anno"
        if col not in adata.obs.columns:
            module_cells[module_id] = set()
            module_cell_counts[module_id] = 0
        else:
            cells = adata.obs[~adata.obs[col].isna()].index
            module_cells[module_id] = set(cells)
            module_cell_counts[module_id] = len(cells)
    # Sort modules by number of annotated cells (descending)
    modules_sorted = sorted(module_ids, key=lambda m: module_cell_counts.get(m, 0), reverse=True)
    
    # Define GO keywords indicative of general cellular activity
    activity_keywords = [kw.lower() for kw in [
        "proliferation", "cell cycle", "cell division", "DNA replication", 
        "RNA processing", "translation", "metabolic process", "biosynthetic process", 
        "ribosome", "chromosome segregation", "spindle", "mitotic", "cell growth"
    ]]
    
    # Initialize result dictionaries
    results = []  # Each record: {module_id, is_identity, type_tag, reason, information}
    is_identity_dict = {}
    type_tag_dict = {}
    reason_dict = {}
    
    # (3) First pass: apply filters sequentially
    for mod in modules_sorted:
        # Retrieve module stats: skew, top1pct_ratio, and positive Moran's I
        if 'skew' in mod_stats_df.columns:
            skewness = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'skew'].iloc[0])
        else:
            skewness = None
        if 'top1pct_ratio' in mod_stats_df.columns:
            top1pct = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'top1pct_ratio'].iloc[0])
        else:
            top1pct = None
        if 'positive_moran_I' in mod_stats_df.columns:
            moranI = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'positive_moran_I'].iloc[0])
        else:
            moranI = None
        
        # Default: module is considered a cell identity marker
        is_identity = True
        type_tag = 'cell_identity_module'
        reason = "Passed all filters"
        
        # (a) Exclude modules with GO terms indicating general cellular activity.
        is_activity = False
        if has_go_info:
            go_terms_str = mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'top_go_terms']
            if not go_terms_str.empty:
                go_terms_str = go_terms_str.iloc[0]
                if pd.notnull(go_terms_str):
                    terms = [t.strip().lower() for t in go_terms_str.split("||")]
                    for term in terms:
                        if any(kw in term for kw in activity_keywords):
                            go_terms_str = term
                            is_activity = True
                            break
        if is_activity:
            is_identity = False
            type_tag = 'cellular_activity_module'
            reason = f'Excluded: GO enrichment indicates cellular activity ({go_terms_str})'
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # (b) Exclude modules with ubiquitous expression (low skew and low top1pct_ratio)
        if skewness is not None and top1pct is not None:
            if skewness < skew_threshold and top1pct < top1pct_threshold:
                is_identity = False
                type_tag = 'ubiquitous_module'
                reason = f'Excluded: Skew ({skewness:.2f}) and top1pct_ratio ({top1pct:.2f}) below thresholds'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # (c) Exclude modules with diffuse spatial patterns (weak spatial autocorrelation)
        if moranI is not None:
            if moranI < Moran_I_threshold and moranI > 0:
                is_identity = False
                type_tag = 'diffuse_module'
                reason = f'Excluded: Positive Moran’s I ({moranI:.2f}) below threshold'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # (d) Exclude modules that are not concentrated in a single spatial domain.
        if ref_anno is not None and module_cell_counts.get(mod, 0) > 0:
            cells = module_cells.get(mod, set())
            clusters = adata.obs.loc[list(cells), ref_anno] if ref_anno in adata.obs.columns else None
            if clusters is not None:
                cluster_counts = clusters.value_counts()
                if len(cluster_counts) > 1:
                    total = cluster_counts.sum()
                    dominant_fraction = cluster_counts.max() / total
                    if dominant_fraction < min_dominant_cluster_fraction:
                        is_identity = False
                        type_tag = 'mixed_regional_module'
                        reason = f'Excluded: Dominant cluster fraction ({dominant_fraction:.2f}) below {min_dominant_cluster_fraction}'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # If the module passes all filters, mark it as a cell identity module.
        is_identity = True
        type_tag = 'cell_identity_module'
        reason = "Included: Passed all filters"
        is_identity_dict[mod] = is_identity
        type_tag_dict[mod] = type_tag
        reason_dict[mod] = reason
        results.append({
            'module_id': mod,
            'is_identity': is_identity,
            'type_tag': type_tag,
            'information': reason
        })
    
    # (4) Second pass: Identify redundant modules by pairwise comparison.
    # For modules with high annotation overlap (Jaccard index >= anno_overlap_threshold),
    # the module with the lower effect_size is considered less discriminative and is marked as redundant.
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
            if jaccard >= anno_overlap_threshold:
                # Use effect_size first to decide which module to keep.
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
                    # The module with the smaller effect_size (lower discriminative power) is dropped.
                    if effA < effB:
                        keep_mod, drop_mod = modB, modA
                    else:
                        keep_mod, drop_mod = modA, modB
                    info_str = (f'Excluded: Jaccard index {jaccard:.2f} >= {anno_overlap_threshold} and ' 
                                f'lower effect_size ({min(effA, effB):.2f})')
                else:
                    # Fallback: use sum of skew and top1pct_ratio as a score.
                    scoreA = 0
                    scoreB = 0
                    if modA in type_tag_dict and modB in type_tag_dict:
                        scoreA = (float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'skew'].iloc[0]) 
                                  if 'skew' in mod_stats_df.columns else 0) + \
                                 (float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'top1pct_ratio'].iloc[0]) 
                                  if 'top1pct_ratio' in mod_stats_df.columns else 0)
                        scoreB = (float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'skew'].iloc[0]) 
                                  if 'skew' in mod_stats_df.columns else 0) + \
                                 (float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'top1pct_ratio'].iloc[0]) 
                                  if 'top1pct_ratio' in mod_stats_df.columns else 0)
                    if scoreB > scoreA:
                        keep_mod, drop_mod = modB, modA
                    info_str = f'Excluded: Jaccard index {jaccard:.2f} >= {anno_overlap_threshold}; fallback score used'
                is_identity_dict[drop_mod] = False
                type_tag_dict[drop_mod] = 'redundant_module'
                reason_dict[drop_mod] = f'Overlap with module {keep_mod}'
                marked_as_similar.add(drop_mod)
                for rec in results:
                    if rec['module_id'] == drop_mod:
                        rec['is_identity'] = False
                        rec['type_tag'] = 'redundant_module'
                        rec['information'] = info_str
                        break
                else:
                    results.append({
                        'module_id': drop_mod,
                        'is_identity': False,
                        'type_tag': 'redundant_module',
                        'information': info_str
                    })
    # (5) Assemble final results DataFrame
    result_df = pd.DataFrame(results)
    if 'module_id' in result_df.columns:
        result_df = result_df.sort_values(by='module_id').reset_index(drop=True)
    adata.uns[mod_filtering_key] = result_df
    
    # Update adata.uns[mod_stats_key] with is_identity and type_tag
    if 'module_id' in mod_stats_df.columns:
        mod_stats_df['is_identity'] = mod_stats_df['module_id'].map(is_identity_dict).fillna(False)
        mod_stats_df['type_tag'] = mod_stats_df['module_id'].map(type_tag_dict).fillna('filtered_module')
        adata.uns[mod_stats_key] = mod_stats_df
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
        adata.uns[mod_stats_key] = mod_stats_df


# %%
# 测试
sg.classify_modules(adata, 
                 ggm_key='intersection',
                 #ref_anno='graph_cluster',
                 ref_cluster_method='leiden', 
                 #ref_cluster_method='none',  
                 ref_cluster_resolution=0.5, 
                 skew_threshold=2,
                 top1pct_threshold=2,
                 Moran_I_threshold=0.6,
                 min_dominant_cluster_fraction=0.3,
                 anno_overlap_threshold=0.4)

# %%
adata.uns['intersection_module_stats']

# %%
adata.uns['intersection_module_filtering']['type_tag'].value_counts()

# %%
adata.uns['intersection_module_filtering'][adata.uns['intersection_module_filtering']['is_identity'] == True]['module_id']

# %%
start_time = time.time()
sg.integrate_annotations(adata,
                        ggm_key='intersection',
                        modules_used=adata.uns['intersection_module_filtering'][adata.uns['intersection_module_filtering']['is_identity'] == True]['module_id'],
                        #modules_used=None,
                        #modules_used=adata.uns[mod_stats_key][adata.uns[mod_stats_key]['module_moran_I'] > 0.7]['module_id'],
                        #modules_preferred=adata.uns[mod_stats_key][adata.uns[mod_stats_key]['module_moran_I'] > 0.9]['module_id'],
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
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)
# %%
