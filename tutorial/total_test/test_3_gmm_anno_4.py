

# %%
# 封包测试3，基本功能完整测试
# 使用 visium_HD - Mouse_Brain_Fixed_Frozen -16um 数据
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
# Before Read Data, tran 'tissue_positions.parquet' to 'tissue_positions.csv'
# df_tissue_positions=pd.read_parquet("data/visium_HD/Mouse_Brain_Fixed_Frozen/binned_outputs/square_016um/spatial/tissue_positions.parquet")
# df_tissue_positions.to_csv("data/visium_HD/Mouse_Brain_Fixed_Frozen/binned_outputs/square_016um/spatial/tissue_position.csv", index=False, header=None)
# df_tissue_positions.to_csv("data/visium_HD/Mouse_Brain_Fixed_Frozen/binned_outputs/square_016um/spatial/tissue_positions_list.csv", index=False, header=None)

# %%
# 读取数据
adata = sc.read_visium("data/visium_HD/Mouse_Brain_Fixed_Frozen/binned_outputs/square_016um",
                       count_file="filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

sc.pp.filter_genes(adata,min_cells=10)
print(adata.X.shape)

# %%
# 使用 GPU 计算GGM，double_precision=False
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
cut_pcor = ggm_gpu_32.fdr.fdr[ggm_gpu_32.fdr.fdr['FDR'] <= 0.05]['Pcor'].min()
if cut_pcor < 0.03:
    cut_pcor = 0.03
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
adata = sc.read_visium("data/visium_HD/Mouse_Brain_Fixed_Frozen/binned_outputs/square_016um",
                       count_file="filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

sc.pp.filter_genes(adata,min_cells=10)
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
adata.obs.loc[:,['annotation','annotation_old']].to_csv("data/Mouse_Brain_Fixed_Frozen.annotation.csv")

# %%
# 注释结果可视化
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.2, title= "", frameon = False, color="annotation_old", show=True)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.2, title= "", frameon = False, color="annotation", show=True)

# %%
# 保存可视化结果
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.2, title= "", frameon = False, color="annotation_old", 
              save="/Mouse_Brain_Fixed_Frozen_All_modules_anno_old.pdf",show=False)

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.2, title= "", frameon = False, color="annotation", 
              save="/Mouse_Brain_Fixed_Frozen_All_modules_anno.pdf",show=False)


# %%
# 保存adata
adata.write("data/visium_HD/Mouse_Brain_Fixed_Frozen_ggm_anno.h5ad")


# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id'].unique()
# 1. 原始注释绘图
pdf_file = "figures/visium_HD/All_modules_in_Mouse_Brain_Fixed_Frozen_Module_Anno_Raw.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()    
        sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= f"{module}_anno", frameon = False, color="plot_anno",show=False)
        raw_png_file = f"figures/visium_HD/{module}_in_Mouse_Brain_Fixed_Frozen_Module_Anno_Raw.png"
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
pdf_file = "figures/visium_HD/All_modules_in_Mouse_Brain_Fixed_Frozen_Module_Anno_Smooth.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    adata.obs['plot_anno'] = adata.obs[f"{module}_anno_smooth"].apply(lambda x: module if x else np.nan)
    if len(adata.obs['plot_anno'][adata.obs['plot_anno'] == module]) > 1:
        plt.figure()
        sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= f"{module}_anno_smooth", frameon = False, color="plot_anno",show=False)
        smooth_png_file = f"figures/visium_HD/{module}_in_Mouse_Brain_Fixed_Frozen_Module_Anno_Smooth.png"
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
pdf_file = "figures/visium_HD/All_modules_in_Mouse_Brain_Fixed_Frozen_Module_Exp.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= f"{module}_exp", frameon = False, color=f"{module}_exp", color_map="Reds", show=False)
    raw_png_file = f"figures/visium_HD/{module}_in_Mouse_Brain_Fixed_Frozen_Module_Exp.png"
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
adata = sc.read("data/visium_HD/Mouse_Brain_Fixed_Frozen_ggm_anno.h5ad")

# %%
sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= "M02_exp", frameon = False, color="M02_exp", color_map="Reds", show=False)
# %%
adata.uns['module_info']

# %%
print(adata.uns['module_info'][adata.uns['module_info']['module_id'] == 'M02'])

# %%
sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= "ENSMUSG00000021948", frameon = False, color="ENSMUSG00000021948", color_map="Reds", show=False)
# %%
sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= "ENSMUSG00000027833", frameon = False, color="ENSMUSG00000027833", color_map="Reds", show=False)

# %%
sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= "ENSMUSG00000038764", frameon = False, color="ENSMUSG00000038764", color_map="Reds", show=False)

# %%
sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= "ENSMUSG00000024985", frameon = False, color="ENSMUSG00000024985", color_map="Reds", show=False)

# %%
sc.pl.spatial(adata, img_key = "lowres", alpha_img = 0.5, size = 1.2, title= "ENSMUSG00000030500", frameon = False, color="ENSMUSG00000030500", color_map="Reds", show=False)

# %%
start_time = time.time()
integrate_annotations(adata,
                  #module_list=None,
                  module_list=['M18', 'M21'],
                  #module_list=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                  #module_list={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                  result_anno='annotation_te',
                  embedding_key='spatial',
                  k_neighbors=18,
                  use_smooth=True,
                  neighbor_majority_frac=0.90
                  )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
module_colors = {
    "M18": "#fb8072",  # 未映射模块
    "M21": "#7570b3",  # 未映射模块
    "None": "whitesmoke"
}
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.2, title= "", frameon = False, color="annotation_te", palette=module_colors, show=True)
# %%
