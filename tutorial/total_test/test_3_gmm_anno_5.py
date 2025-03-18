

# %%
# 封包测试3，基本功能完整测试
# 使用 xenium - Whole Mouse Pup (FFPE) with 5K Mouse Pan Tissue and Pathways Panel and Snap25a/Snap25b Add-on 数据
import numpy as np
import pandas as pd
import random
import time
import torch
import scanpy as sc
import anndata
import os
import gc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# %%
# 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
from ST_GGM_dev_1 import *

# %%
# 读取数据
adata = sc.read_10x_h5('data/xenium/pup_xenium/cell_feature_matrix.h5')
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
meta = pd.read_csv('data/xenium/pup_xenium/cells.csv.gz')
adata.obs = meta
adata.obsm['spatial'] = adata.obs[['x_centroid','y_centroid']].values*[-1,-1]

sc.pp.log1p(adata)
print(adata.X.shape)

sc.pp.filter_genes(adata,min_cells=10)
print(adata.X.shape)

sc.pp.filter_cells(adata, min_genes=100)
print(adata.X.shape)

# %%
# 使用 GPU 计算GGM，double_precision=False
start_time = time.time()
ggm_gpu_32 = ST_GGM_Pytorch(adata,  
                            round_num=20000, 
                            dataset_name = "Mouse_Pup_5K", 
                            selected_num=500, 
                            cut_off_pcor=0.03,
                            run_mode=2, 
                            double_precision=False,
                            use_chunking=True,
                            chunk_size=50000,
                            stop_threshold=0,
                            FDR_control=False,
                            FDR_threshold=0.05,
                            auto_adjust=False
                            )

SigEdges_gpu_32 = ggm_gpu_32.SigEdges
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# FDR检验
start_time = time.time()
ggm_gpu_32.fdr_control()
print(f"Time: {time.time() - start_time:.5f} s")

# %%
print(ggm_gpu_32.fdr.fdr[ggm_gpu_32.fdr.fdr['FDR'] <= 0.05])

# %%
# 调整pcor阈值
cut_pcor = ggm_gpu_32.fdr.fdr[ggm_gpu_32.fdr.fdr['FDR'] <= 0.05]['Pcor'].min()
if cut_pcor < 0.02:
    cut_pcor = 0.02
print("Adjust cutoff pcor:", cut_pcor)
ggm_gpu_32.adjust_cutoff(pcor_threshold=cut_pcor)

# %%
ggm_gpu_32.find_modules(methods='mcl',
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=True, 
                        convert_to_symbols=False, species='mouse')
print(ggm_gpu_32.modules_summary)

# %%
start_time = time.time()
go_enrichment_analysis(ggm_gpu_32, 
                       padjust_method='BH',
                       pvalue_cutoff=0.05,
                       species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm_gpu_32.go_enrichment)

# %%
start_time = time.time()
mp_enrichment_analysis(ggm_gpu_32,
                       padjust_method='BH',
                       pvalue_cutoff=0.05,
                       species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm_gpu_32.mp_enrichment)

# %%
# 重新读取数据
del adata
adata = sc.read_10x_h5('data/xenium/pup_xenium/cell_feature_matrix.h5')
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
meta = pd.read_csv('data/xenium/pup_xenium/cells.csv.gz')
adata.obs = meta
adata.obsm['spatial'] = adata.obs[['x_centroid','y_centroid']].values*[-1,-1]

sc.pp.log1p(adata)
print(adata.X.shape)

# %%
# 计算模块的加权表达值
start_time = time.time()
calculate_module_expression(adata, ggm_gpu_32, 
                            top_genes=30,
                            weighted=True)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_info'])

# %%
# 计算GMM注释
start_time = time.time()
calculate_gmm_annotations(adata, 
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
smooth_annotations(adata, 
                    #module_list=None,
                    #module_list=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                    #module_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                    #module_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                    embedding_key='spatial',
                    k_neighbors=18,
                    min_annotated_neighbors=2
                    )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
# 合并注释（考虑空间坐标和模块表达值）
start_time = time.time()
integrate_annotations(adata,
                  #module_list=None,
                  #module_list=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                  #module_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #module_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='annotation',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_majority_frac=0.90
                  )
print(f"Time: {time.time() - start_time:.5f} s")


# %%
# 合并注释（仅考虑模块注释的细胞数目）
start_time = time.time()
integrate_annotations_old(adata,
                         #module_list=None,
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
overlap_records = calculate_module_overlap(adata, 
                                           module_list = adata.uns['module_stats']['module_id'].unique())
print(f"Time: {time.time() - start_time:.5f} s")
print(overlap_records[overlap_records['module_a'] == 'M01'])

# %%
# 保存注释结果
adata.obs.loc[:,['annotation','annotation_old']].to_csv("data/Mouse_Pup_5K.annotation.csv")

# %%
# 注释结果可视化
sc.pl.spatial(adata, spot_size=10, title= "", frameon = False, color="annotation_old", show=True)

# %%
sc.pl.spatial(adata, spot_size=10, title= "", frameon = False, color="annotation", show=True)

# %%
# 保存可视化结果
sc.pl.spatial(adata, spot_size=10, title= "", frameon = False, color="annotation_old", 
              save="/Mouse_Pup_5K_All_modules_anno_old.pdf",show=False)

# %%
sc.pl.spatial(adata, spot_size=10, title= "", frameon = False, color="annotation", 
              save="/Mouse_Pup_5K_All_modules_anno.pdf",show=False)


# %%
# 保存adata
adata.write("data/xenium/Mouse_Pup_5K_ggm_anno.h5ad")









# %%
adata = sc.read("data/xenium/Mouse_Pup_5K_ggm_anno.h5ad")



# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id'].unique()

# %%
# 1. 原始注释绘图
pdf_file = "figures/xenium/All_modules_in_Mouse_Pup_5K_Module_Anno_Raw.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()    
        sc.pl.spatial(adata, spot_size=20, title= f"{module}_anno", frameon = False, color="plot_anno",show=False)
        raw_png_file = f"figures/xenium/{module}_in_Mouse_Pup_5K_Module_Anno_Raw.png"
        plt.savefig(raw_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        gc.collect()
        image_files.append(raw_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    


# %%
# 2. 平滑注释绘图
anno_modules = adata.uns['module_stats']['module_id'].unique()
pdf_file = "figures/xenium/All_modules_in_Mouse_Pup_5K_Module_Anno_Smooth.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno_smooth"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()
        sc.pl.spatial(adata, spot_size=20, title= f"{module}_anno_smooth", frameon = False, color="plot_anno",show=False)
        smooth_png_file = f"figures/xenium/{module}_in_Mouse_Pup_5K_Module_Anno_Smooth.png"
        plt.savefig(smooth_png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        image_files.append(smooth_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    


# %%
# 3. 模块加权表达图
anno_modules = adata.uns['module_stats']['module_id'].unique()
pdf_file = "figures/xenium/All_modules_in_Mouse_Pup_5K_Module_Exp.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, spot_size=20, title= f"{module}_exp", frameon = False, color=f"{module}_exp", color_map="Reds", show=False)
    raw_png_file = f"figures/xenium/{module}_in_Mouse_Pup_5K_Module_Exp.png"
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
