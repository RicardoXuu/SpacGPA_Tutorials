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
adata.obsm['spatial'] = adata.obs[['y_centroid','x_centroid']].values*[-1,-1]
print(adata)

# %%
# Preprocessing: log1p-transform.
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata.X.shape)

# %%
# Visualize the total UMI counts per spot.
plt.rcParams["figure.figsize"] = (4.5, 6)
sc.pl.spatial(adata, spot_size = 30, color = 'total_counts', frameon = False, title = 'Total UMI counts')

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata, project_name = "Mouse Brain")

# %%
# Show statistically significant co-expression gene pairs.
print(ggm.SigEdges.head(5))

# %%
# Identify gene programs using the MCL-Hub algorithm (set inflation to 2).
ggm.find_modules(method = 'mcl-hub', inflation = 2, convert_to_symbols = True, species = 'mouse')
# the parameter 'convert_to_symbols' provides gene symbols in the output programs for better interpretability.

# %%
# Inspect the top 5 identified gene programs.
print(ggm.modules_summary.head(5))

# %%
# Visualize the subnetwork of program M4 (top 30 genes by degree/connectivity for readability).
ggm.module_network_plot(module_id = 'M4', seed = 2, layout_iterations = 60) 
# Fix layout randomness for reproducibility via set seed.

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.go_enrichment_analysis(species = 'mouse', padjust_method = "BH", pvalue_cutoff = 0.05)

# %%
# Visualize top enriched GO terms for top 20 identified programs.
program_list = ggm.modules_summary['module_id'].tolist()
ggm.module_go_enrichment_plot(shown_modules = program_list[:10], go_per_module = 1)

# %%
# Mammalian Phenotype (MP) Ontology enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.mp_enrichment_analysis(species = 'mouse', padjust_method = "BH", pvalue_cutoff = 0.05)

# %%
# Visualize top enriched MP terms for top 20 identified programs.
ggm.module_mp_enrichment_plot(shown_modules = program_list[:10], mp_per_module = 1)

# %%
# Visualize the M4 network with nodes highlighted by a selected GO or MP term.
M4_GO_Enrich = ggm.go_enrichment[ggm.go_enrichment['module_id'] == 'M4']
print(M4_GO_Enrich.iloc[:3, :6])
ggm.module_network_plot(module_id = 'M4', highlight_anno = "dendrite", seed = 2, layout_iterations = 55)
M4_MP_Enrich = ggm.mp_enrichment[ggm.mp_enrichment['module_id'] == 'M4']
print(M4_MP_Enrich.iloc[:3, :5])
ggm.module_network_plot(module_id = 'M4', highlight_anno = "abnormal CNS synaptic transmission", seed = 2, layout_iterations = 55)

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
# Visualize the spatial distribution of the top 20 program-expression scores.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, spot_size = 30, color = program_list[:20], cmap = 'Reds', ncols = 5)

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
sc.pl.spatial(adata, spot_size = 30, color = program_list[:20], legend_loc = None, ncols = 5)
# Where the blue nodes indicate the spots annotated by the program, and gray nodes are unassigned.

# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key = 'ggm', use_smooth = False, neighbor_similarity_ratio = 0.6, result_anno = 'ggm_annotation')
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.

# %%
# Visualize the integrated annotation.
plt.rcParams["figure.figsize"] = (7, 7)
sc.pl.spatial(adata, spot_size = 30, color = ['ggm_annotation'], palette = adata.uns['module_colors'], frameon = False, title = 'Integrated annotation')


# %% [markdown]
#### Part 3: Cluster spots based a dimensionality reduction of program expression ###

# %%
# Build a neighborhood graph based on program expression and perform clustering.
sc.pp.neighbors(adata, 
                use_rep='module_expression_scaled',
                n_pcs=adata.obsm['module_expression_scaled'].shape[1])
sc.tl.leiden(adata, resolution=3, key_added='leiden_ggm')
sc.tl.louvain(adata, resolution=3, key_added='louvan_ggm')

# %%
# Visualize the clustering results.
plt.rcParams["figure.figsize"] = (6, 6)
sc.pl.spatial(adata, spot_size = 30, color = ['leiden_ggm'], frameon = False, title = 'Leiden clustering')
plt.rcParams["figure.figsize"] = (6, 6)
sc.pl.spatial(adata, spot_size = 30, color = ['louvan_ggm'], frameon = False, title = 'Louvain clustering')

# %%
# Summarize program-expression across the leiden clusters clusters as a dot plot.
sg.module_dot_plot(adata, ggm_key = 'ggm', groupby = 'leiden_ggm', scale=True,
                   dendrogram_height = 0.1, dendrogram_space = 0.03, fig_height=14, fig_width = 20, axis_fontsize = 10)

# %%
# Save the annotated AnnData object.
adata.write("data/Mouse_Brain_5K_ggm_anno.h5ad")
