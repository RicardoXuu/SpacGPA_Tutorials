# %%
# 一些问题修复, 切片联合分析相关
# 使用 小鼠大脑多张切片
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
# 读取数据
adata_1 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep1",
                       count_file="CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5")
adata_2 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep2",
                          count_file="CytAssist_FFPE_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata_3 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep1",
                            count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5")
adata_4 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                            count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata_5 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_Fresh_Frozen_Mouse_Brain",
                            count_file="CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_filtered_feature_bc_matrix.h5")

# %%
adata_list = [adata_1, adata_2, adata_3, adata_4, adata_5]
for adata in adata_list:
    print(adata.X.shape)
    adata.var_names_make_unique()
    adata.var_names = adata.var['gene_ids']
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.filter_genes(adata,min_cells=10)
    print(adata.X.shape)
    print('------------------')

# %%
adata_list

# %%
# 使用交集基因
start_time = time.time()
ggm_mulit_intersection = sg.create_ggm_multi(adata_list,  
                                method='intersection',
                                #round_num=500,
                                #selected_num=200,  
                                #target_sampling_time=200,
                                project_name = "mouse_brain_mulit_intersection", 
                                cut_off_pcor=0.03,
                                run_mode=2, 
                                double_precision=False,
                                use_chunking=True,
                                chunk_size=5000,
                                stop_threshold=0,
                                FDR_control=True,
                                FDR_threshold=0.01,
                                auto_adjust=True
                                )

SigEdges_mulit_intersection = ggm_mulit_intersection.SigEdges
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.save_ggm(ggm_mulit_intersection, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
del ggm_mulit_intersection
gc.collect()
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
ggm_mulit_intersection

# %%
start_time = time.time()
ggm_mulit_intersection.fdr_control()
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.save_ggm(ggm_mulit_intersection, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
del ggm_mulit_intersection
gc.collect()
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
ggm_mulit_intersection

# %%
print(ggm_mulit_intersection.fdr.summary[ggm_mulit_intersection.fdr.summary['FDR'] <= 0.01])

# %%
ggm_mulit_intersection.adjust_cutoff(pcor_threshold=0.033)

# %%
ggm_mulit_intersection.find_modules(methods='mcl-hub',
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=True, 
                        convert_to_symbols=True, species='mouse')
print(ggm_mulit_intersection.modules_summary)

# %%
ggm_mulit_intersection.modules

# %%
sg.save_ggm(ggm_mulit_intersection, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
del ggm_mulit_intersection
gc.collect()
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
ggm_mulit_intersection

# %%
ggm_mulit_intersection.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
ggm_mulit_intersection.go_enrichment

# %%
sg.save_ggm(ggm_mulit_intersection, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
del ggm_mulit_intersection
gc.collect()
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
ggm_mulit_intersection

# %%
ggm_mulit_intersection.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
ggm_mulit_intersection.mp_enrichment

# %%
sg.save_ggm(ggm_mulit_intersection, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
del ggm_mulit_intersection
gc.collect()
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
ggm_mulit_intersection


# %%
M1_edges = ggm_mulit_intersection.get_module_edges("M001")
M1_anno = ggm_mulit_intersection.get_module_anno("M001", add_enrich_info=True, top_n=5)

# %%
M1_edges = sg.get_module_edges(ggm_mulit_intersection, "M001")
M1_anno = sg.get_module_anno(ggm_mulit_intersection, "M001", add_enrich_info=True, top_n=5)


# %%
sg.save_ggm(ggm_mulit_intersection, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
del ggm_mulit_intersection
gc.collect()
ggm_mulit_intersection = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_intersection.h5")
ggm_mulit_intersection

# %%




# %%
# 使用全部基因
start_time = time.time()
ggm_mulit_union = sg.create_ggm_multi(adata_list,  
                                method="union",
                                #round_num=500,
                                #selected_num=200,  
                                #target_sampling_time=200,
                                project_name = "mouse_brain_mulit_union", 
                                cut_off_pcor=0.03,
                                run_mode=2, 
                                double_precision=False,
                                use_chunking=True,
                                chunk_size=5000,
                                stop_threshold=0,
                                FDR_control=True,
                                FDR_threshold=0.01,
                                auto_adjust=True
                                )

SigEdges_gpu_32 = ggm_mulit_union.SigEdges
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.save_ggm(ggm_mulit_union, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
del ggm_mulit_union
gc.collect()
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
ggm_mulit_union


# %%
start_time = time.time()
ggm_mulit_union.fdr_control()
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.save_ggm(ggm_mulit_union, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
del ggm_mulit_union
gc.collect()
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
ggm_mulit_union

# %%
print(ggm_mulit_union.fdr.summary[ggm_mulit_union.fdr.summary['FDR'] <= 0.01])

# %%
ggm_mulit_union.adjust_cutoff(pcor_threshold=0.033)

# %%
ggm_mulit_union.find_modules(methods='mcl-hub',
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=True, 
                        convert_to_symbols=True, species='mouse')
print(ggm_mulit_union.modules_summary)

# %%
ggm_mulit_union.modules

# %%
sg.save_ggm(ggm_mulit_union, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
del ggm_mulit_union
gc.collect()
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
ggm_mulit_union

# %%
ggm_mulit_union.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
ggm_mulit_union.go_enrichment

# %%
sg.save_ggm(ggm_mulit_union, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
del ggm_mulit_union
gc.collect()
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
ggm_mulit_union

# %%
ggm_mulit_union.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
ggm_mulit_union.mp_enrichment

# %%
sg.save_ggm(ggm_mulit_union, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
del ggm_mulit_union
gc.collect()
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
ggm_mulit_union


# %%
M1_edges = ggm_mulit_union.get_module_edges("M001")
M1_anno = ggm_mulit_union.get_module_anno("M001", add_enrich_info=True, top_n=5)

# %%
M1_edges = sg.get_module_edges(ggm_mulit_union, "M001")
M1_anno = sg.get_module_anno(ggm_mulit_union, "M001", add_enrich_info=True, top_n=5)


# %%
sg.save_ggm(ggm_mulit_union, "data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
del ggm_mulit_union
gc.collect()
ggm_mulit_union = sg.load_ggm("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.ggm_mulit_union.h5")
ggm_mulit_union

# %%
