# %%
# r3版本，对多张visium切片进行联合分析
# 使用SpacGPA对 6张visium切片 进行联合分析
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

# %% 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
import SpacGPA as sg


# %%
# 读取数据
adata_1 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_Fresh_Frozen_Mouse_Brain",
                            count_file="CytAssist_Fresh_Frozen_Mouse_Brain_filtered_feature_bc_matrix.h5")
adata_1.var_names = adata_1.var['gene_ids']

adata_2 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep1",
                       count_file="CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5")
adata_2.var_names = adata_2.var['gene_ids']

adata_3 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep2",
                          count_file="CytAssist_FFPE_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata_3.var_names = adata_3.var['gene_ids']

adata_4 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/V1_Adult_Mouse_Brain", 
                        count_file='V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5')
adata_4.var_names = adata_4.var['gene_ids']

adata_5 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/Visium_Fresh_Frozen_Adult_Mouse_Brain", 
                        count_file='Visium_Fresh_Frozen_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5')
adata_5.var_names = adata_5.var['gene_ids']

adata_6 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/Visium_FFPE_Mouse_Brain", 
                         count_file='Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5')
adata_6.var_names = adata_6.var['gene_ids']

# %%
adata = sc.concat([adata_1, adata_2, adata_3, adata_4, adata_5, adata_6], axis=0, join='outer', 
                           label='batch', keys=[
                                               'coronal_ff','coronal_ffpe1','coronal_ffpe2',
                                                'coronal_ff_v1','brain_ff_adult','brain_ffpe_adult'])
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata,min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)


# %%
start_time = time.time()
ggm = sg.create_ggm(adata,
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.05,
                    auto_adjust=True,
                    )
start_time = time.time()

# %%
# 调整Pcor阈值
cut_pcor = ggm.fdr.summary[ggm.fdr.summary['FDR'] <= 0.05]['Pcor'].min()
if cut_pcor < 0.02:
    cut_pcor = 0.02
print("Adjust cutoff pcor:", cut_pcor)
ggm.adjust_cutoff(pcor_threshold=0.032)


# %%
ggm.fdr.summary[0:30]


# %%
# 使用改进的mcl聚类识别共表达模块
start_time = time.time()
ggm.find_modules(methods='mcl-hub',
                 expansion=2, inflation=2, 
                 #max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                 min_module_size=10, topology_filtering=True, 
                 convert_to_symbols=True, species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.modules_summary)


# %%



# %%
# GO富集分析
start_time = time.time()
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.go_enrichment)

# %%
# MP富集分析
start_time = time.time()
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.mp_enrichment)

# %%
# 打印GGM信息
ggm

# %%
# 保存GGM
start_time = time.time()
sg.save_ggm(ggm, "data/Visium_Mouse_Brain_Multi_Concat.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/Visium_Mouse_Brain_Multi_Concat.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
ggm

# %%
print(ggm.modules_summary[0:10])
ggm.modules_summary.to_csv("data/Visium_Mouse_Brain_Multi_Concat_ggm_modules_summary_r3.csv")


# %%
adata_visium = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep1",
                       count_file="CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5")
adata_visium.var_names = adata_visium.var['gene_ids']
sc.pp.normalize_total(adata_visium,target_sum=1e4)
sc.pp.log1p(adata_visium)

# %%
adata_hd = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium_HD/Mouse_Brain_Fixed_Frozen/binned_outputs/square_016um/",
                       count_file="filtered_feature_bc_matrix.h5")
adata_hd.var_names = adata_hd.var['gene_ids']
adata_hd.var_names_make_unique()
coor_int = [[float(x[0]),float(x[1])] for x in adata_hd.obsm["spatial"]]
adata_hd.obsm["spatial"] = np.array(coor_int)
sc.pp.normalize_total(adata_hd, target_sum=1e4)
sc.pp.log1p(adata_hd)

# %%
adata_sc = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/Sc_Data/WMB-10Xv3/WMB-10Xv3-all-downsampled.h5ad')

# %%
