
# %%
# 使用SpacGPA对 MOATA_E16.5_E1S1 数据进行分析
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
# 读取空转数据
adata = sc.read_h5ad("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/MOSTA/E16.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata.X.shape)

sc.pp.filter_cells(adata, min_genes=1000)
print(adata.X.shape)

sc.pp.filter_genes(adata, min_cells=10)
print(adata.X.shape)

# %%
# 使用 GPU 计算GGM，double_precision=False
ggm = sg.create_ggm(adata,
                    project_name = "E16.5_E1S1", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.01,
                    auto_adjust=True,
                    )  
print(ggm.SigEdges)


# %%
# 调整Pcor阈值
ggm.adjust_cutoff(pcor_threshold=0.03)

# %%
# 使用改进的mcl聚类识别共表达模块
start_time = time.time()
ggm.find_modules(methods='mcl-hub',
                 expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                 min_module_size=10, topology_filtering=True, 
                 convert_to_symbols=False, species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm.modules_summary)

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
sg.save_ggm(ggm, "data/MOSTA_E16.5_E1S1.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/MOSTA_E16.5_E1S1.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
ggm

# %%
# 重新读取数据
#del adata
adata = sc.read_h5ad("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/MOSTA/E16.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata.X.shape)


# %%
# 计算模块的加权表达值
start_time = time.time()
sg.calculate_module_expression(adata, 
                               ggm, 
                               ggm_key='ggm',
                               top_genes=30,
                               weighted=True,
                               calculate_moran=False,
                               embedding_key='spatial',
                               k_neighbors=6,
                               add_go_anno=5,
                               )
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_info'])

# %%
# 删除GGM对象，释放内存
del ggm
gc.collect()

# %%
# 使用leiden聚类和louvain聚类基于模块表达矩阵归一化矩阵进行聚类
# start_time = time.time()
# sc.pp.neighbors(adata, n_neighbors=18, use_rep='module_expression_scaled',n_pcs=adata.obsm['module_expression_scaled'].shape[1])
# sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5_ggm')
# sc.tl.leiden(adata, resolution=1, key_added='leiden_1_ggm')
# sc.tl.louvain(adata, resolution=0.5, key_added='louvan_0.5_ggm')
# sc.tl.louvain(adata, resolution=1, key_added='louvan_1_ggm')
# print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 可视化聚类结果
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="annotation", show=True)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="leiden_0.5_ggm", show=True)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="leiden_1_ggm", show=True)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="louvan_0.5_ggm", show=True)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="louvan_1_ggm", show=True)

# %%
# 保存可视化结果
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="annotation",
                save="/MOSTA_E16_5_E1S1_raw_cell_type_annotation.pdf",show=False)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="leiden_0.5_ggm", 
#               save="/MOSTA_E16_5_E1S1_ggm_modules_leiden_0.5.pdf",show=False)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="leiden_1_ggm",
#                 save="/MOSTA_E16_5_E1S1_ggm_modules_leiden_1.pdf",show=False)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="louvan_0.5_ggm",
#                 save="/MOSTA_E16_5_E1S1_ggm_modules_louvan_0.5.pdf",show=False)
# sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="louvan_1_ggm",
#                 save="/MOSTA_E16_5_E1S1_ggm_modules_louvan_1.pdf",show=False)



# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            embedding_key='spatial',
                            k_neighbors=6,
                            random_state=42)
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])


# %%
adata.uns['module_stats'].to_csv("data/MOSTA_E16.5_E1S1_ggm_module_stats.csv")

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                      ggm_key='ggm',
                      embedding_key='spatial',
                      k_neighbors=24,
                      min_annotated_neighbors=2
                      )
print(f"Time: {time.time() - start_time:.5f} s")    


# %%
# 分析模块类型
start_time = time.time()
sg.classify_modules(adata, 
                    ggm_key='ggm',
                    ref_anno='annotation',
                    #ref_cluster_method='leiden',
                    #ref_resolution=0.5,
                    skew_threshold=2,
                    top1pct_threshold=2,
                    Moran_I_threshold=0.2,
                    min_dominant_cluster_fraction=0.2,
                    anno_overlap_threshold=0.4)

# %%
adata.uns['module_filtering']['type_tag'].value_counts()


# %%
# 测试新版函数
sg.integrate_annotations(
    adata,
    ggm_key='ggm',
    #modules_excluded=['M15', 'M18'],        
    #modules_preferred=['M28', 'M38'],
    result_anno='annotation_new',
    k_neighbors=24,
    lambda_pair=0.3,
    purity_adjustment=True,
    w_floor=0.01,
    lr=0.5,
    target_purity=0.85,
    # alpha=0.5,
    # beta=0.3
    gamma=0.3,
    # delta=0.4,   
    max_iter=100,
    random_state=0)


# %%
# 合并注释（考虑空间坐标和模块表达值）
# 1. 使用全部模块
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        result_anno='ggm_annotation',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )
# 2. 使用经过鉴别的模块
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity']== True]['module_id'],
                        result_anno='ggm_annotation_filtered',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )
# 3. 不使用activity模块
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_excluded=adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cellular_activity_module']['module_id'],
                        result_anno='ggm_annotation_no_activity',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0.9,
                        )

# %%
# 合并注释（不考虑空间坐标）
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        result_anno='ggm_annotation_no_spatial',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
# 2. 使用经过鉴别的模块
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity']== True]['module_id'],
                        result_anno='ggm_annotation_filtered_no_spatial',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
# 3. 不使用activity模块
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        modules_excluded=adata.uns['module_filtering'][adata.uns['module_filtering']['type_tag']=='cellular_activity_module']['module_id'],
                        result_anno='ggm_annotation_no_activity_no_spatial',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )


# %%
# 保存注释结果
adata.obs.to_csv("data/MOSTA_E16.5_E1S1_ggm_annotation.csv")


# %%
# 注释结果可视化
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation", show=True)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_filtered", show=True)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_no_activity", show=True)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_no_spatial", show=True)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_filtered_no_spatial", show=True)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_no_activity_no_spatial", show=True)


# %%
# 保存可视化结果
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation", 
              save="/MOSTA_E16_5_E1S1_All_modules_annotation.pdf",show=False)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_filtered",
                save="/MOSTA_E16_5_E1S1_Filtered_modules_annotation.pdf",show=False)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_no_activity",
                save="/MOSTA_E16_5_E1S1_No_activity_modules_annotation.pdf",show=False)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_no_spatial",
                save="/MOSTA_E16_5_E1S1_All_modules_annotation_no_spatial.pdf",show=False)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_filtered_no_spatial",
                save="/MOSTA_E16_5_E1S1_Filtered_modules_annotation_no_spatial.pdf",show=False)
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="ggm_annotation_no_activity_no_spatial",
                save="/MOSTA_E16_5_E1S1_No_activity_modules_annotation_no_spatial.pdf",show=False)


# %%
# 保存adata
adata.write("data/MOSTA_E16.5_E1S1_ggm_anno.h5ad")

# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id']
pdf_file = "figures/MOSTA/E16_5_E1S1_all_modules_Anno.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, spot_size=1.2, frameon = False, color_map="Reds", 
                  color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"],show=False)
    show_png_file = f"figures/MOSTA/E16_5_E1S1_{module}_Anno.png"
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
