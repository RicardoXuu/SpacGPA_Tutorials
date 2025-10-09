# %% [markdown]
## Tutorial 3: Xenium Prime 5K: Mouse Brain

# %% [markdown]
# <div style="margin:0; line-height:1.2">
# Analyze mouse brain spatial transcriptomics data with SpacGPA.<br/>  
#
# Data source: https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain  <br/> 
#
# This is a mouse brain sample generated with Xenium Prime 5K.<br/>  
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
# Read the count matrix from the 10x Genomics Cell Ranger output.
adata = sc.read_10x_h5('data/Xenium_5k/Mouse_Brain_5K/cell_feature_matrix.h5')
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
# Read the cell metadata (including spatial coordinates) from the provided CSV file.
meta = pd.read_csv('data/Xenium_5k/Mouse_Brain_5K/cells.csv.gz')
meta.index = meta['cell_id'].astype(str)
meta = meta.reindex(adata.obs_names)
adata.obs = adata.obs.join(meta, how='left')
# Set the spatial coordinates.
adata.obsm['spatial'] = adata.obs[['x_centroid','y_centroid']].values
print(adata)



# %%
# Preprocessing: log1p-transform.
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata.X.shape)

# %%
# Visualize the provided cell type annotation.
plt.rcParams["figure.figsize"] = (4.5, 6)
sc.pl.spatial(adata, spot_size = 30, color = 'cell_type', frameon = False, title = 'Cell type annotation')

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata, project_name = "Mouse Brain")

# %%
# Show statistically significant co-expression gene pairs.
print(ggm.SigEdges.head(5))

# %%
# Identify gene programs using the MCL-Hub algorithm (set inflation to 2).
ggm.find_modules(method = 'mcl-hub', inflation = 2, convert_to_symbols = True, species = 'human')
# the parameter 'convert_to_symbols' provides gene symbols in the output programs for better interpretability.

# %%
# Inspect the top 5 identified gene programs.
print(ggm.modules_summary.head(5))

# %%
# Visualize the subnetwork of program M1 (top 30 genes by degree/connectivity for readability).
ggm.module_network_plot(module_id = 'M1', seed = 2, layout_iterations = 60) 
# Fix layout randomness for reproducibility via set seed.

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
# sg.download_go_annotations(species = 'human', outdir = 'data/go_annotations')
ggm.go_enrichment_analysis(species = 'human', padjust_method = "BH", pvalue_cutoff = 0.05)

# %%
# Visualize top enriched GO terms for top 5 identified programs.
ggm.module_go_enrichment_plot(shown_modules = ['M1','M2','M3','M4','M5'], go_per_module = 1)

# %%
# Visualize the M1 network with nodes highlighted by a selected GO or MP term.
print(ggm.go_enrichment.iloc[0, :6])
ggm.module_network_plot(module_id = 'M1', highlight_anno = "T cell activation", seed = 2, layout_iterations = 60)

# %%
# Print a summary of the GGM analysis.
print(ggm)

# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, "data/Mouse_Brain_5K.ggm.h5")

# %% [markdown]
#### Part 2: Spot annotation based on program expression ###

# %%
# Compute per-spot expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)

# %%
# Visualize the spatial distribution of all program-expression scores.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, spot_size = 30, color = program_list, cmap = 'RdYlBu_r', ncols = 6)

# %%
# Compute pairwise program similarity and plot the correlation heatmap with dendrograms.
sg.module_similarity_plot(adata, ggm_key = 'ggm', corr_method = 'pearson', heatmap_metric = 'correlation', 
                          fig_height = 9, fig_width = 10, dendrogram_height = 0.1, dendrogram_space = 0.12, return_summary = False)

# %%
# Assign spot-level annotations via Gaussian Mixture Models (GMMs) based on program expression.
sg.calculate_gmm_annotations(adata, ggm_key = 'ggm')

# %%
# Display annotations for all identified programs.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_anno'
sc.pl.spatial(adata, spot_size = 30, color = program_list, legend_loc = None, ncols = 6)
# Where the blue nodes indicate the spots annotated by the program, and gray nodes are unassigned.

# %%
# Summarize program-expression across the existing annotation categories as a dot plot.
sg.module_dot_plot(adata, ggm_key = 'ggm', groupby = 'cell_type', scale=True,
                   dendrogram_height = 0, fig_height = 7, fig_width = 12, axis_fontsize = 10)

# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key = 'ggm', use_smooth = False, neighbor_similarity_ratio = 0.6, result_anno = 'ggm_annotation')
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.

# %%
# Visualize the integrated annotation.
plt.rcParams["figure.figsize"] = (4.5, 6)
sc.pl.spatial(adata, spot_size = 30, color = ['ggm_annotation'], frameon = False, title = 'Integrated annotation')

# %%
# Save the annotated AnnData object.
adata.write("data/Mouse_Brain_5K_ggm_anno.h5ad")
