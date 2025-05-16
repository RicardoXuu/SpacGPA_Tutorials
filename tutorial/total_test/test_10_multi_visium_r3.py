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
adata_5 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_Fresh_Frozen_Mouse_Brain",
                            count_file="CytAssist_Fresh_Frozen_Mouse_Brain_filtered_feature_bc_matrix.h5")
adata_5.var_names = adata_5.var['gene_ids']

adata_6 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep1",
                       count_file="CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5")
adata_6.var_names = adata_6.var['gene_ids']

adata_7 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FFPE_Mouse_Brain_Rep2",
                          count_file="CytAssist_FFPE_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata_7.var_names = adata_7.var['gene_ids']



adata_3 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep1",
                            count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5")
adata_3.var_names = adata_3.var['gene_ids']
adata_4 = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                            count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata_4.var_names = adata_4.var['gene_ids']

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

