
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
from SpacGPA import *

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

# %%
start_time = time.time()
ggm_gpu_32 = ST_GGM_Pytorch(adata,  
                            round_num=20000, 
                            dataset_name = "mouse_brain", 
                            selected_num=2000, 
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
start_time = time.time()
ggm_gpu_32.fdr_control()
print(f"Time: {time.time() - start_time:.5f} s")

# %%
print(ggm_gpu_32.fdr.fdr[ggm_gpu_32.fdr.fdr['FDR'] <= 0.05])

# %%
ggm_gpu_32.adjust_cutoff(pcor_threshold=0.056)

# %%
ggm_gpu_32.find_modules(methods='mcl',
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=True, 
                        convert_to_symbols=False, species='human')
print(ggm_gpu_32.modules_summary)




# %%
# 细胞注释相关的问题
# 问题1，关于计算平均表达值。当使用新的ggm结果注释已经存在module expression的adata时，会报错

# %%
# 问题2，关于计算平均表达值。添加可选参数，计算模块内每个基因的莫兰指数。

# %%
# 问题3，关于模块注释的全部函数，添加反选参数，用来反向排除模块。

# %%
# 问题4，关于模块注释的全部函数, 细胞按模块的注释结果改为category类型。而不是现在的0，1，int类型。并注意，之后在涉及到使用这些数据的时候还要换回int类型。

# %%
# 问题5，关于高斯混合分布，设计activity模块的排除标准。尽量不使用先验知识，

# %%
# 问题6，关于高斯混合分布，阈值和主成分数目的关系优化。

# %%
# 问题7，关于高斯混合分布，除了使用高斯混合分布，也考虑表达值的排序。
#       对于一个模块，只有那些表达水平大于模块最大表达水平（或者为了防止一些离散的点，可以考虑前20个或者30个细胞的平均值作为模块最大表达水平）的一定比例的细胞才被认为是注释为该模块的

# %%
# 问题8，关于平滑处理，在使用的时候，无法仅处理部分模块。

# %%
# 问题9，关于合并注释，优化keep modules的参数。

# %%
# 问题10，关于合并注释，尝试引入模块的整体莫兰指数，来评估模块的空间分布。如果一个模块的莫兰指数很高，则优先考虑该模块的细胞的可信度。

# %%
# 问题11，关于合并注释，尝试结合louvain或者leiden的聚类结果，在每个聚类之内使用模块来精准注释。

# %%
# 问题12，关于合并注释，注释结果中，字符串None改为空值的None。

# %%
# 问题13，关于合并注释，在adata的uns中添加一个配色方案，为每个模块指定配色，特别是模块过多的时候。

# %%
# 问题14，关于合并注释，neighbor_majority_frac参数似乎会导致activity模块的权重过高。考虑将其设置为大于1的值。