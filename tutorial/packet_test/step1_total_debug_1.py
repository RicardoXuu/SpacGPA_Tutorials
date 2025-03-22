
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
# 读取数据
adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                       count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()

adata.var_names = adata.var['gene_ids']

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

sc.pp.filter_genes(adata,min_cells=10)
print(adata.X.shape)

# 设置基因数目
# sc.pp.highly_variable_genes(adata, n_top_genes=5000)
# adata = adata[:, adata.var.highly_variable]
# print(adata.X.shape)

# %%
start_time = time.time()
ggm_gpu_32 = sg.create_ggm(adata,  
                        #round_num=500,
                        #selected_num=200,  
                        #target_sampling_count=200,
                        project_name = "mouse_brain", 
                        cut_off_pcor=0.03,
                        run_mode=2, 
                        double_precision=False,
                        use_chunking=True,
                        chunk_size=5000,
                        stop_threshold=0,
                        FDR_control=False,
                        FDR_threshold=0.05,
                        auto_adjust=False
                        )

SigEdges_gpu_32 = ggm_gpu_32.SigEdges
print(f"Time: {time.time() - start_time:.5f} s")

# %%
ggm_gpu_32

# %%
sg.save_ggm(ggm_gpu_32, "data/ggm_gpu_32.h5")
del ggm_gpu_32
gc.collect()
ggm_gpu_32 = sg.load_ggm("data/ggm_gpu_32.h5")
ggm_gpu_32

# %%
start_time = time.time()
ggm_gpu_32.fdr_control()
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sg.save_ggm(ggm_gpu_32, "data/ggm_gpu_32.h5")
del ggm_gpu_32
gc.collect()
ggm_gpu_32 = sg.load_ggm("data/ggm_gpu_32.h5")
ggm_gpu_32

# %%
print(ggm_gpu_32.fdr.summary[ggm_gpu_32.fdr.summary['FDR'] <= 0.05])

# %%
ggm_gpu_32.adjust_cutoff(pcor_threshold=0.059)

# %%
ggm_gpu_32.find_modules(methods='mcl-hub',
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=True, 
                        convert_to_symbols=True, species='mouse')
print(ggm_gpu_32.modules_summary)

# %%
ggm_gpu_32.modules
# %%
sg.save_ggm(ggm_gpu_32, "data/ggm_gpu_32.h5")
del ggm_gpu_32
gc.collect()
ggm_gpu_32 = sg.load_ggm("data/ggm_gpu_32.h5")
ggm_gpu_32

# %%
ggm_gpu_32.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
ggm_gpu_32.go_enrichment

# %%
sg.save_ggm(ggm_gpu_32, "data/ggm_gpu_32.h5")
del ggm_gpu_32
gc.collect()
ggm_gpu_32 = sg.load_ggm("data/ggm_gpu_32.h5")
ggm_gpu_32

# %%
ggm_gpu_32.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
ggm_gpu_32.mp_enrichment

# %%
sg.save_ggm(ggm_gpu_32, "data/ggm_gpu_32.h5")
del ggm_gpu_32
gc.collect()
ggm_gpu_32 = sg.load_ggm("data/ggm_gpu_32.h5")
ggm_gpu_32


# %%
M1_edges = ggm_gpu_32.get_module_edges("M01")
M1_anno = ggm_gpu_32.get_module_anno("M01", add_enrich_info=True, top_n=5)

# %%
M1_edges = sg.get_module_edges(ggm_gpu_32, "M01")
M1_anno = sg.get_module_anno(ggm_gpu_32, "M01", add_enrich_info=True, top_n=5)


# %%
sg.save_ggm(ggm_gpu_32, "data/ggm_gpu_32.h5")
del ggm_gpu_32
gc.collect()
ggm_gpu_32 = sg.load_ggm("data/ggm_gpu_32.h5")
ggm_gpu_32


# %%
# GGM计算相关的问题
# 问题1，ggm计算的部分改为，create_ggm, 并整理ggm的数据架构，使其可以作为h5文件储存。
# 解决

# %%
# 问题2，GO和MP函数内置到ggm。
# 解决

# %%
# 问题3，select_number的自动化选择：
#        5000个基因到20000个基因，取 1/10的基因数目，平均采样100次。
#        大于20000，取2000个基因，平均采样100次。
#        小于5000，取500个基因，平均采样100次。
#        小于500，用全部的基因，只算1次。
# 解决

# %%
# 问题4，Pcor的阈值选择。优先考虑FDR, 细胞数目越多，可以接受越小的Pcor阈值。
# 解决

# %%
# 问题5，Chuking_size的默认阈值的设置改为较大一点的值。
# 解决

# %%
# 问题6，放射状模块的去除需要更加严格
# 待定，没有合适的标准

# %%
# 问题7，设计函数，提取指定模块的edges用于绘图。
# 解决

# %%
# 问题8，设计函数，提取模块的Annotation信息，添加可选参数提取GO或者MP的注释信息。
# 解决

# %%
# 问题9，修改部分代码的打印信息
# 解决
# 1， MCL改为MCL-Hub，MCL-Original改为MCL
# 2， fdr.fdr改为fdr.summary
# 3， 设计了__repr__函数，用于打印ggm的基本信息
# 4， 在fdr_control和adjust_cutoff函数中添加了部分打印信息
