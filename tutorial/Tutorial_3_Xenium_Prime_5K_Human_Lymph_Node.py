# %% [markdown]
## Tutorial 3: Xenium Prime 5K: Human Lymph Node

# %% [markdown]
# <div style="margin:0; line-height:1.2">
# Analyze Human Lymph Node Spatial Transcriptomics data with SpacGPA.<br/>  
#
# Data source: https://www.10xgenomics.com/cn/datasets/preview-data-xenium-prime-gene-expression  <br/> 
#
# This is a human lymph node sample generated with Xenium Prime 5K.<br/>  
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
adata = sc.read_10x_h5('data/Xenium_5k/Human_Lymph_Node_5K/cell_feature_matrix.h5')
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
meta = pd.read_csv('data/Xenium_5k/Human_Lymph_Node_5K/cells.csv.gz')
adata.obs = meta
adata.obsm['spatial'] = adata.obs[['x_centroid','y_centroid']].values
print(adata)

# %%
# Preprocessing: log1p-transform.
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata.X.shape)

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata, project_name = "Human Lymph Node")


# %%
# Show statistically significant co-expression gene pairs.
print(ggm.SigEdges.head(5))

# %%
# For first-time use SpacGPA on a new species, download the GO annotations via sg.get_go_annotations.
# Which will also provide the ENSEMBL to gene symbol mapping file.
sg.get_GO_annoinfo(species_name = 'human')


# %%
# Identify gene programs using the MCL-Hub algorithm (set inflation to 2).
ggm.find_modules(method = 'mcl-hub', inflation = 2, convert_to_symbols = True, species = 'human')
# the parameter 'convert_to_symbols' provides gene symbols in the output programs for better interpretability.


# %%
# Inspect the top 5 identified gene programs.
print(ggm.modules_summary.head(5))


# %%
# Visualize the subnetwork of program M1 (top 30 genes by degree/connectivity for readability).
ggm.module_network_plot(module_id = 'M1', seed = 1) 
# Fix layout randomness for reproducibility via set seed.

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
# sg.download_go_annotations(species = 'human', outdir = 'data/go_annotations')
ggm.go_enrichment_analysis(species = 'human', padjust_method = "BH", pvalue_cutoff = 0.05)


# %%
# Visualize top enriched GO terms for all identified programs.
ggm.module_go_enrichment_plot(shown_modules = ggm.modules_summary['module_id'].tolist(), go_per_module = 1,
                              fig_width = 4.5)


# %%
# Visualize the M1 network with nodes highlighted by a selected GO or MP term.
print(ggm.go_enrichment.iloc[0, :6])
ggm.module_network_plot(module_id = 'M1', highlight_anno = "angiogenesis", seed = 1)
print(ggm.go_enrichment.iloc[1, :6])
ggm.module_network_plot(module_id = 'M1', highlight_anno = "cell adhesion", seed = 1)

# %%
# Print a summary of the GGM analysis.
print(ggm)


# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, "data/Human_Lymph_Node_5K.ggm.h5")


# %% [markdown]
#### Part 2: Spot annotation based on program expression ###

# %%
# Compute per-spot expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)


# %%
# Visualize the spatial distribution of the top 20 program-expression scores.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, color = program_list, cmap = 'RdYlBu_r', ncols = 5)


# %%
# Compute pairwise program similarity and plot the correlation heatmap with dendrograms.
sg.module_similarity_plot(adata, ggm_key = 'ggm', corr_method = 'pearson', heatmap_metric = 'correlation', 
                          fig_height = 8, fig_width = 9, dendrogram_height = 0.1, dendrogram_space = 0.12, return_summary = False)


# %%
# Assign spot-level annotations via Gaussian Mixture Models (GMMs) based on program expression.
sg.calculate_gmm_annotations(adata, ggm_key = 'ggm')


# %%
# Optionally smooth the annotations using spatial k-NN (on the 'spatial' embedding).
sg.smooth_annotations(adata, ggm_key = 'ggm', embedding_key = 'spatial', k_neighbors = 24)


# %%
# Display smoothed annotations for the top 20 programs.
# If smoothing is skipped, use 'M1_anno' … 'M20_anno' instead.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_anno_smooth'
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = program_list, legend_loc = None, ncols = 5)
# Where the blue nodes indicate the spots annotated by the program, and gray nodes are unassigned.


# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key = 'ggm', result_anno = 'ggm_annotation')
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.


# %%
# Visualize the integrated annotation.
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = ['ggm_annotation'], frameon = False, title = 'Integrated annotation')


# %% [markdown]
#### Part 3: Cluster spots based a dimensionality reduction of program expression ###


# %%
# Build a neighborhood graph based on program expression and perform clustering.
sc.pp.neighbors(adata, 
                use_rep='module_expression_scaled',
                n_pcs=adata.obsm['module_expression_scaled'].shape[1])
sc.tl.leiden(adata, resolution=1, key_added='leiden_ggm')
sc.tl.louvain(adata, resolution=1, key_added='louvan_ggm')

# %%
# Visualize the clustering results.
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = ['leiden_ggm'], frameon = False, title = 'Leiden clustering')
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = ['louvan_ggm'], frameon = False, title = 'Louvain clustering')

# %%
# Summarize program-expression across the leiden clusters clusters as a dot plot.
sg.module_dot_plot(adata, ggm_key = 'ggm', groupby = 'leiden_ggm', scale=True,
                   dendrogram_height = 0.1, dendrogram_space = 0.08, fig_height=6, fig_width = 8, axis_fontsize = 10)

# %%
# Save the annotated AnnData object.
adata.write("data/Human_Lymph_Node_5K_ggm_anno.h5ad")
