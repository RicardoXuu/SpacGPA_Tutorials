# %% [markdown]
## Tutorial 4: MERFISH: Mouse Brain

# %% [markdown]
# <div style="margin:0; line-height:1.2">
# Analyze mouse brain spatial transcriptomics data with SpacGPA.<br/>  
#
# Data source: https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html  <br/> 
#
# This is a mouse brain sample generated with MERFISH.<br/>  
# <div>

# %%
# Import SpacGPA and other required packages.
import SpacGPA as sg
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
# Set the working directory to your local path.
workdir = '..'
os.chdir(workdir)

# %% [markdown]
#### Part 1: Gene program analysis via SpacGPA ###

# %%
# Load spatial transcriptomics data.
# Read the count matrix from the provided HDF5 file.
adata_all = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MERFISH/Zhuang-ABCA-1/Zhuang-ABCA-1-log2.h5ad')

# %%
# Read the cell metadata (including spatial coordinates) from the provided CSV file.
meta = pd.read_csv('/dta/ypxu/ST_GGM/Raw_Datasets/MERFISH/Zhuang-ABCA-1/cell_metadata.csv',header=0,index_col=0)
meta = meta.drop(columns=['brain_section_label'])
adata_all = adata_all[meta.index,:]
adata_all.obs = adata_all.obs.join(meta, how='left')
# Set the spatial coordinates.
adata_all.obsm['spatial']=meta[['x','y']].values
print(adata_all)

# %%
# Show each brain section label counts.
adata_all.obs['brain_section_label'].value_counts()

# %%
adata = adata_all[adata_all.obs['brain_section_label']=='Zhuang-ABCA-1.080']
print(adata.X.shape) 

# %%
# Visualize the Section Label.
plt.rcParams["figure.figsize"] = (4.5, 6)
sc.pl.spatial(adata, spot_size = 0.03, color = 'brain_section_label', frameon = False, title = 'Brain Section Label')

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
# Here we show a CPU-based calculation mode. For faster computation, you can also use GPU mode if a compatible GPU is available.
ggm = sg.create_ggm(adata, project_name = "Mouse Brain",run_mode= 0)
# For run_mode parameter: 0 for CPU mode; 1 for hybrid CPU-GPU mode; 2 for GPU mode.

# %%
# Show statistically significant co-expression gene pairs.
print(ggm.SigEdges.head(5))

# %%
# Identify gene programs using the MCL-Hub algorithm (set inflation to 2).
# Here we set min_module_size=5 to capture smaller modules in this dataset.
ggm.find_modules(method = 'mcl-hub', inflation = 2, min_module_size=5, convert_to_symbols = True, species = 'mouse')
# the parameter 'convert_to_symbols' provides gene symbols in the output programs for better interpretability.

# %%
# Inspect the top 5 identified gene programs.
print(ggm.modules_summary.head(5))

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.go_enrichment_analysis(species = 'mouse', padjust_method = "BH", pvalue_cutoff = 0.05)

# %%
# Visualize top enriched GO terms for top 20 identified programs.
program_list = ggm.modules_summary['module_id'].tolist()
ggm.module_go_enrichment_plot(shown_modules = program_list[:20], go_per_module = 1)

# %%
# Mammalian Phenotype (MP) Ontology enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.mp_enrichment_analysis(species = 'mouse', padjust_method = "BH", pvalue_cutoff = 0.05)

# %%
# Visualize top enriched MP terms for top 20 identified programs.
ggm.module_mp_enrichment_plot(shown_modules = program_list[:10], mp_per_module = 1)

# %%
# Print a summary of the GGM analysis.
print(ggm)

# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, "data/Mouse_Brain_MERFISH_T080.ggm.h5")

# %% [markdown]
#### Part 2: Spot annotation based on program expression ###

# %%
# Compute per-spot expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)

# %%
# Visualize the spatial distribution of the top 20 program-expression scores.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, spot_size = 0.03, color = program_list[:20], cmap = 'Reds', ncols = 5)

# %%
# Compute pairwise program similarity and plot the correlation heatmap with dendrograms.
sg.module_similarity_plot(adata, ggm_key = 'ggm', corr_method = 'pearson', heatmap_metric = 'correlation', 
                          fig_height = 19, fig_width = 20, dendrogram_height = 0.1, dendrogram_space = 0.06, return_summary = False)

# %%
# Assign spot-level annotations via Gaussian Mixture Models (GMMs) based on program expression.
sg.calculate_gmm_annotations(adata, ggm_key = 'ggm')

# %%
# Optionally smooth the annotations using spatial k-NN (on the 'spatial' embedding).
sg.smooth_annotations(adata, ggm_key = 'ggm', embedding_key = 'spatial', k_neighbors = 24)

# %%
# Display smoothed annotations for top 20 programs.
# If smoothing is skipped, use 'M1_anno' … 'M20_anno' instead.
program_list = ggm.modules_summary['module_id'] + '_anno_smooth'
sc.pl.spatial(adata, spot_size = 0.03, color = program_list[:20], legend_loc = None, ncols = 5)
# Where the blue nodes indicate the spots annotated by the program, and gray nodes are unassigned.

# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key = 'ggm', use_smooth = False, neighbor_similarity_ratio = 0.6, result_anno = 'ggm_annotation')
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.

# %%
# Visualize the integrated annotation.
plt.rcParams["figure.figsize"] = (7, 7)
sc.pl.spatial(adata, spot_size = 0.03, color = ['ggm_annotation'], palette = adata.uns['module_colors'], frameon = False, title = 'Integrated annotation')

# %%
# Save the annotated AnnData object.
adata.write("data/Mouse_Brain_MERFISH_T080_ggm_anno.h5ad")
