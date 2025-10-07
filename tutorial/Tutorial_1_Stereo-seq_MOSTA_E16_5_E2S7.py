
# %%
# Analyze MOSTA E16.5 E2S7 Spatial Transcriptomics Data using SpacGPA
# Data source: https://db.cngb.org/stomics/mosta/
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

# %%
# import SpacGPA
import SpacGPA as sg

# %%
# Set working directory
workdir = '..'
os.chdir(workdir)

# %%
# Load Spatial Transcriptomics data
adata = sc.read_h5ad("data/Stereo-seq/MOSTA/E16.5_E2S7.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata)

# %%
# Data preprocessing
adata.X = adata.layers['count']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
print(adata.X.shape)

# %%
# Calculate Coexpression Network using SpacGPA
ggm = sg.create_ggm(adata,project_name = "E16.5_E2S7")  

# %%
# show significant co-expression gene pairs
print(ggm.SigEdges.head(5))

# %%
# identify gene programs via MCL-Hub algorithm
ggm.find_modules(method='mcl-hub',inflation=2)

# %%
# show identified programs
print(ggm.modules_summary.head(5))

# %%
# visualize network of program M1 via sg.module_network_plot
M1_edges = ggm.get_module_edges('M1')

# %%
M1_anno = ggm.get_module_anno('M1')

# %%
sg.module_network_plot(
    nodes_edges = M1_edges,
    nodes_anno = M1_anno
) 
# %%

# %%
# Show Relationship between degree and Moran's I of genes within program M1 via sg.module_degree_vs_moran_plot



# %%
# GO Enrichment Analysis
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
# MP Enrichment Analysis
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)


# %%
# visualize GO enrichment results for programs M1-M5 via sg.module_go_enrichment_plot

# %%
# visualize MP enrichment results for programs M1-M5 via sg.module_mp_enrichment_plot



# %%
# Save GGM object as h5 file
sg.save_ggm(ggm, "data/MOSTA_E16.5_E2S7.ggm.h5")

# %%
# Load GGM object
ggm = sg.load_ggm("data/MOSTA_E16.5_E2S7.ggm.h5")

# %%
# Calculate program expression in each spot
sg.calculate_module_expression(adata, ggm)

# %%
# Show expression distribution of top 5 programs


# %%
# Anno spots with GMM based on program expression
sg.calculate_gmm_annotations(adata, ggm_key='ggm')

# %%
# Smooth annotations based on spatial neighbors
sg.smooth_annotations(adata, ggm_key='ggm', embedding_key='spatial', k_neighbors=24)

# %%
# Calculate correlation between programs and visualize correlation heatmap
mod_cor = sg.module_similarity_plot(adata,
                                    ggm_key='ggm',
                                    corr_method='pearson',
                                    heatmap_metric='correlation',   
                                    fig_height=20,
                                    fig_width=21,
                                    dendrogram_height=0.1,
                                    dendrogram_space=0.05)


# %%
# integrate annotations
sg.integrate_annotations(adata, ggm_key='ggm')

# %%
# visualize integrated annotations
sc.pl.spatial(adata, spot_size = 2, color='annotation')


# %%
# Save annotated anndata object
adata.write("data/MOSTA_E16.5_E2S7_ggm_anno.h5ad")

# %%
adata.obs
# %%
