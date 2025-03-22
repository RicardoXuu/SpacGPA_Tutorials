
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


# %%
# 读取 ggm
ggm = sg.load_ggm("data/ggm_gpu_32.h5")

# %%
# 读取联合分析的ggm
ggm_mulit_intersection = sg.load_ggm("data/ggm_mulit_intersection.h5")
ggm_mulit_union = sg.load_ggm("data/ggm_mulit_union.h5")

# %%
print(ggm)
print(ggm_mulit_intersection)
print(ggm_mulit_union)


# %%
# 计算模块的加权表达值
start_time = time.time()
sg.calculate_module_expression(adata, ggm_mulit_intersection, 
                            top_genes=30,
                            weighted=True)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_info'])

# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                         #modules_list=None,
                         #modules_list=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #modules_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #modules_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         max_iter=200,
                         prob_threshold=0.99,
                         min_samples=10,
                         n_components=3,
                         enable_fallback=True,
                         random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                    #module_list=None,
                    #module_list=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                    #module_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                    module_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                    embedding_key='spatial',
                    k_neighbors=18,
                    min_annotated_neighbors=2
                    )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
# 合并注释（考虑空间坐标和模块表达值）
start_time = time.time()
sg.integrate_annotations(adata,
                  #module_list=None,
                  module_list = adata.uns['module_info']['module_id'].unique(), 
                  #module_list=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #module_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #module_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_majority_frac=1.1
                  )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
adata.uns['module_info']['module_id'].unique()

# %%
# 合并注释（仅考虑模块注释的细胞数目）
start_time = time.time()
sg.integrate_annotations_old(adata,
                         #module_list=None,
                         module_list = adata.uns['module_info']['module_id'].unique(), 
                         #module_list=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                         #module_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                         #module_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                         result_anno = "annotation_old",
                         use_smooth=True
                         )
print(f"Time: {time.time() - start_time:.5f} s")


# %%
# 计算模块重叠
start_time = time.time()
overlap_records = sg.calculate_module_overlap(adata, 
                                           module_list = adata.uns['module_stats']['module_id'].unique())
print(f"Time: {time.time() - start_time:.5f} s")
print(overlap_records[overlap_records['module_a'] == 'M01'])

# %%
# 保存注释结果
adata.obs.loc[:,['annotation','annotation_old']].to_csv("data/CytAssist_FreshFrozen_Mouse_Brain_Rep2.annotation.csv")

# %%
# 注释结果可视化
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)

# %%
# 保存可视化结果
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_old", 
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_All_modules_anno_old.pdf",show=False)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", 
              save="/CytAssist_FreshFrozen_Mouse_Brain_Rep2_All_modules_anno.pdf",show=False)


# %%
# 保存adata
adata.write("data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_ggm_anno.h5ad")


# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id'].unique()
# 1. 原始注释绘图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Raw.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()    
        sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_anno", frameon = False, color="plot_anno",show=False)
        raw_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Raw.png"
        plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# 2. 平滑注释绘图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Smooth.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno_smooth"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()
        sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_anno_smooth", frameon = False, color="plot_anno",show=False)
        smooth_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Anno_Smooth.png"
        plt.savefig(smooth_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(smooth_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# 3. 模块加权表达图
pdf_file = "figures/visium/All_modules_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Exp.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, img_key = "hires", alpha_img = 0.5, size = 1.6, title= f"{module}_exp", frameon = False, color=f"{module}_exp", color_map="Reds", show=False)
    raw_png_file = f"figures/visium/{module}_in_CytAssist_FreshFrozen_Mouse_Brain_Rep2_Module_Exp.png"
    plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# %%









# %%
# 细胞注释相关的问题
# 问题1，关于计算平均表达值。当使用新的ggm结果注释已经存在module expression的adata时，会报错
# 添加ggm_id 参数，为GGM指定一个id，用来区分不同的ggm结果。

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