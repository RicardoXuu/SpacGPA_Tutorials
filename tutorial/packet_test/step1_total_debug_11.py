
# %%
# 使用 Visium_HD mouse small intestine 数据集
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
adata = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium_HD/Visium_HD_Mouse_Small_Intestine/binned_outputs/square_016um/",
                       count_file="filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

sc.pp.filter_genes(adata,min_cells=10)
print(adata.X.shape)

# %%
start_time = time.time()
ggm = sg.create_ggm(adata,
                    project_name = "mouse_small_intestine",
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=False,
                    FDR_threshold=0.01,
                    auto_adjust=False,
                    auto_find_modules=False,
                    )  
print(f"Time: {time.time() - start_time:.5f} s")

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
sg.save_ggm(ggm, "data/Mouse_Small_Intestine_16um.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/Mouse_Small_Intestine_16um.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
ggm.adjust_cutoff(pcor_threshold=0.02)

# %%
best_inf_1,_ = sg.find_best_inflation(ggm, min_inflation=1.1, phase=3, show_plot=True)

# %%
ggm.find_modules(methods='mcl-hub',
                 expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                 min_module_size=10, topology_filtering=True, 
                 convert_to_symbols=False, species='mouse')
print(ggm.modules_summary.shape)


# %%
# 重新读取数据
# del adata
adata = sc.read_visium("/dta/ypxu/ST_GGM/Raw_Datasets/visium_HD/Visium_HD_Mouse_Small_Intestine/binned_outputs/square_016um/",
                       count_file="filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)


# %%
# 使用模块信息注释细胞
sg.annotate_with_ggm(adata,ggm,
                     ggm_key='ggm')

# %%
# 保存产生的模块注释信息
module_auto = ggm.modules
ggm.modules.to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_modules.csv",index=False)
ggm.modules_summary.to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_modules_summary.csv",index=False)
ggm.go_enrichment.to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_go_enrichment.csv",index=False)
ggm.mp_enrichment.to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_mp_enrichment.csv",index=False)

adata.obs.to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_annotate.csv",index=False)
module_auto_info = adata.uns['module_info']
adata.uns['module_info'].to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_module_info.csv",index=False)
adata.uns['module_stats'].to_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_module_stats.csv",index=False)


# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id']
pdf_file = "figures/visium_HD/Mouse_Small_Intestine_16um_auto_fdr_0_01_all_modules_Anno.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, size=1.2, alpha_img=0.5, frameon = False, color_map="Reds", 
                  color=[f"{module}_exp",f"{module}_exp_trim",f"{module}_anno",f"{module}_anno_smooth"],show=False)
    show_png_file = f"figures/visium_HD/Mouse_Small_Intestine_16um_auto_fdr_0_01_{module}_Anno.png"
    plt.savefig(show_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(show_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    


# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        result_anno='annotation_auto',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )

# %%
sc.pl.spatial(adata, size=1.2, alpha_img=0.5, title= "", frameon = False, color="annotation_auto", show=True,
              save="/Mouse_Small_Intestine_16um_auto_fdr_0_01_annotation_auto.pdf")





# %%
# 调整Pcor阈值，重新注释
ggm.adjust_cutoff(pcor_threshold=0.02)
ggm.find_modules(methods='mcl-hub',
                 expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                 min_module_size=10, topology_filtering=True, 
                 convert_to_symbols=False, species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# GO富集分析
start_time = time.time()
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# MP富集分析
start_time = time.time()
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.mp_enrichment)

# %%
# 使用模块信息注释细胞
sg.annotate_with_ggm(adata,ggm,
                     ggm_key='ggm')

# %%
# 保存产生的模块注释信息
module_manual = ggm.modules
ggm.modules.to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_modules.csv",index=False)
ggm.modules_summary.to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_modules_summary.csv",index=False)
ggm.go_enrichment.to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_go_enrichment.csv",index=False)
ggm.mp_enrichment.to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_mp_enrichment.csv",index=False)

adata.obs.to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_annotate.csv",index=False)
module_manual_info = adata.uns['module_info']
adata.uns['module_info'].to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_module_info.csv",index=False)
adata.uns['module_stats'].to_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_module_stats.csv",index=False)


# %%
del ggm
gc.collect()

# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id']
pdf_file = "figures/visium_HD/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_all_modules_Anno.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, size=1.2, alpha_img=0.5, frameon = False, color_map="Reds", 
                  color=[f"{module}_exp",f"{module}_exp_trim",f"{module}_anno",f"{module}_anno_smooth"],show=False)
    show_png_file = f"figures/visium_HD/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_{module}_Anno.png"
    plt.savefig(show_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(show_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()


# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        result_anno='annotation_manual',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
sc.pl.spatial(adata, size=1.2, alpha_img=0.5, title= "", frameon = False, color="annotation_manual", show=True,
              save="/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_annotation_manual.pdf")





# %%
# 比较自动选择和手动选择得到的模块异同
module_auto = pd.read_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_modules.csv")
module_auto_info = pd.read_csv("data/Mouse_Small_Intestine_16um_auto_fdr_0_01_module_info.csv")
module_manual = pd.read_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_modules.csv")
module_manual_info = pd.read_csv("data/Mouse_Small_Intestine_16um_pcor_0_02_inflation_2_module_info.csv")   

# %%
# 计算模块间的Jaccard相似度
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
def compare_modules(module_auto, module_manual):
    # 提取模块ID和基因
    auto_modules = module_auto.groupby('module_id')['gene'].apply(set).reset_index()
    manual_modules = module_manual.groupby('module_id')['gene'].apply(set).reset_index()

    # 计算Jaccard相似度
    similarities = []
    for _, auto_row in auto_modules.iterrows():
        for _, manual_row in manual_modules.iterrows():
            similarity = jaccard_similarity(auto_row['gene'], manual_row['gene'])
            similarities.append({
                'auto_module': auto_row['module_id'],
                'manual_module': manual_row['module_id'],
                'similarity': similarity
            })

    return pd.DataFrame(similarities)

# %%
similarities_all = compare_modules(module_auto, module_manual)
similarities_top = compare_modules(module_auto_info, module_manual_info)

# %%
similarities_top[similarities_top['similarity'] > 0.3]
