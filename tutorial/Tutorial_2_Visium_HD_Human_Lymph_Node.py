
# %%
# Analyze Human Lymph Node Spatial Transcriptomics data with SpacGPA.
# Data source: https://www.10xgenomics.com/cn/datasets/visium-hd-cytassist-gene-expression-libraries-human-lymph-node-v4 
# This is a human lymph node sample generated with Visium HD.

# %%
# Import SpacGPA and other required packages.
import SpacGPA as sg
import scanpy as sc
import matplotlib.pyplot as plt
import os

# %%
# Set the working directory to your local path.
workdir = '..'
os.chdir(workdir)

### Part 1: Gene program analysis via SpacGPA ####
# %%
# For Visium HD data, we recommend using the binned outputs (e.g. 16μm binning here) for analysis.
# Before first-time use, construct the tissue_positions.csv from the tissue_positions.parquet file for easy reading.
# Demo codes for this conversion:
# import pandas as pd
# df_tissue_positions=pd.read_parquet("data/visium_HD/Human_Lymph_Node/binned_outputs/square_016um/spatial/tissue_positions.parquet")
# df_tissue_positions.to_csv("data/visium_HD/Human_Lymph_Node/binned_outputs/square_016um/spatial/tissue_position.csv", index=False, header=None)
# df_tissue_positions.to_csv("data/visium_HD/Human_Lymph_Node/binned_outputs/square_016um/spatial/tissue_positions_list.csv", index=False, header=None)

# %%
# Load spatial transcriptomics data.
adata = sc.read_visium("data/visium_HD/Human_Lymph_Node/binned_outputs/square_016um/",
                       count_file="filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
print(adata)

# %%
# Preprocessing: library-size normalize and log1p-transform.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
print(adata.X.shape)

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata,project_name = "Human_Lymph_Node")  

# %%
# Show statistically significant co-expression gene pairs.
print(ggm.SigEdges.head(5))

# %%
# Identify gene programs using the MCL-Hub algorithm (set inflation to 2).
ggm.find_modules(method='mcl-hub',inflation=2)

# %%
# Inspect the top 5 identified gene programs.
print(ggm.modules_summary.head(5))

# %%
# Visualize the subnetwork of program M1 (top 30 genes by degree/connectivity for readability).
M1_edges = ggm.get_module_edges('M1')
M1_anno = ggm.get_module_anno('M1')
sg.module_network_plot(
    nodes_edges = M1_edges,
    nodes_anno = M1_anno,
    seed=2 # Fix layout randomness for reproducibility
)

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
# Visualize top enriched GO terms for programs M1–M5.
sg.module_go_enrichment_plot(ggm, top_n_modules=5, go_per_module=2)

# %%
# Mammalian Phenotype (MP) Ontology enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
# Visualize top enriched MP terms for programs M1–M5.
sg.module_mp_enrichment_plot(ggm, top_n_modules=5, mp_per_module=2)

# %%
# Visualize the M1 network with nodes highlighted by a selected GO or MP term ID.
M1_edges = ggm.get_module_edges('M1')
M1_anno = ggm.get_module_anno('M1')
print(ggm.go_enrichment.iloc[0, :6])
sg.module_network_plot(
    nodes_edges = M1_edges,
    nodes_anno = M1_anno,
    highlight_anno="GO:0030016",
    seed=2
)
print(ggm.mp_enrichment.iloc[0, :5])
sg.module_network_plot(
    nodes_edges = M1_edges,
    nodes_anno = M1_anno,
    highlight_anno="MP:0005369",
    seed=2
)

# %%
# Print a summary of the GGM analysis.
print(ggm)

# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, "data/Human_Lymph_Node.ggm.h5")
# Then you can reload it via:
# ggm = sg.load_ggm("data/Human_Lymph_Node.ggm.h5")



#### Part 2: Spot annotation based on program expression ####
# %%
# Compute per-spot expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)

# %%
# Visualize the spatial distribution of the top 20 program-expression scores.
sc.pl.spatial(adata, spot_size = 2, color=['M1_exp','M2_exp','M3_exp','M4_exp','M5_exp',
                                           'M6_exp','M7_exp','M8_exp','M9_exp','M10_exp',
                                           'M11_exp','M12_exp','M13_exp','M14_exp','M15_exp',
                                           'M16_exp','M17_exp','M18_exp','M19_exp','M20_exp'], 
              cmap='RdYlBu_r',ncols=5)

# %%
# Compute pairwise program similarity and plot the correlation heatmap with dendrograms.
mod_cor = sg.module_similarity_plot(adata,
                                    ggm_key='ggm',
                                    corr_method='pearson',
                                    heatmap_metric='correlation',   
                                    fig_height=20,
                                    fig_width=21,
                                    dendrogram_height=0.1,
                                    dendrogram_space=0.05)

# %%
# Assign spot-level annotations via Gaussian Mixture Models (GMMs) based on program expression.
sg.calculate_gmm_annotations(adata, ggm_key='ggm')
# Optionally smooth the annotations using spatial k-NN (on the 'spatial' embedding).
sg.smooth_annotations(adata, ggm_key='ggm', embedding_key='spatial', k_neighbors=24)

# %%
# Display smoothed annotations for the top 20 programs alongside the original annotation.
# 'annotation' refers to the original labels in adata.obs['annotation'].
# If smoothing is skipped, use 'M1_anno' … 'M20_anno' instead.
sc.pl.spatial(adata, spot_size = 2, color=['M1_anno_smooth','M2_anno_smooth','M3_anno_smooth','M4_anno_smooth','M5_anno_smooth',
                                           'M6_anno_smooth','M7_anno_smooth','M8_anno_smooth','M9_anno_smooth','M10_anno_smooth',
                                           'M11_anno_smooth','M12_anno_smooth','M13_anno_smooth','M14_anno_smooth','M15_anno_smooth',
                                           'M16_anno_smooth','M17_anno_smooth','M18_anno_smooth','M19_anno_smooth','M20_anno_smooth',
                                           'annotation'], 
              ncols=5)


# %%
# Summarize program-expression across the existing annotation categories as a dot plot.
sg.module_dot_plot(
    adata,
    ggm_key='ggm',
    groupby= 'annotation',
    dendrogram_height = 0.1,
    dendrogram_space= 0.2,
    fig_width = 24,
    axis_fontsize=10,
)

# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key='ggm',result_anno='ggm_annotation')
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.

# %%
# Visualize the integrated annotation and compare it to the existing annotation.
sc.pl.spatial(adata, spot_size = 2, color=['ggm_annotation'], frameon=False, title='Integrated annotation')
sc.pl.spatial(adata, spot_size = 2, color=['annotation'], frameon=False, title='Original annotation')

# %%
# Save the annotated AnnData object.
adata.write("data/Human_Lymph_Node_ggm_anno.h5ad")

# %%