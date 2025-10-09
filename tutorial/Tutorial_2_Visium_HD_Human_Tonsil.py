# %% [markdown]
## Tutorial 2: Visium HD: Human Tonsil

# %% [markdown]
# <div style="margin:0; line-height:1.2">
# Analyze human tonsil spatial transcriptomics data with SpacGPA.<br/>  
#
# Data source: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen  <br/> 
#
# This is a human tonsil sample generated with Visium HD.<br/>  
# <div>

# %%
# Import SpacGPA and other required packages.
import SpacGPA as sg
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", message=r".*Variable names are not unique.*")

# %%
# Set the working directory to your local path.
workdir = '..'
os.chdir(workdir)

# %% [markdown]
#### Part 1: Gene program analysis via SpacGPA ###

# %%
# For Visium HD data, we recommend using the binned outputs (e.g. 16μm binning here) for analysis.
# Before first-time use, construct the tissue_positions.csv from the tissue_positions.parquet file for easy reading.
# A demo for file conversion:
df_tissue_positions = pd.read_parquet("data/visium_HD/Human_Tonsil/binned_outputs/square_016um/spatial/tissue_positions.parquet")
df_tissue_positions.to_csv("data/visium_HD/Human_Tonsil/binned_outputs/square_016um/spatial/tissue_position.csv", index = False, header = None)
df_tissue_positions.to_csv("data/visium_HD/Human_Tonsil/binned_outputs/square_016um/spatial/tissue_positions_list.csv", index = False, header = None)

# %%
# Load spatial transcriptomics data.
adata = sc.read_visium("data/visium_HD/Human_Tonsil/binned_outputs/square_016um/",
                       count_file = "filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
print(adata)

# %%
# Preprocessing: library-size normalize and log1p-transform.
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes = 200)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata.X.shape)

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata, project_name = "Human Tonsil")

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
# Visualize the subnetwork of program M8 (top 30 genes by degree/connectivity for readability).
ggm.module_network_plot(module_id = 'M8', seed = 1, layout_iterations = 55)
# Fix layout randomness for reproducibility via set seed.

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.go_enrichment_analysis(species = 'human', padjust_method = "BH", pvalue_cutoff = 0.05)

# %%
# Visualize top enriched GO terms for all identified programs.
program_list = ggm.modules_summary['module_id'].tolist()
ggm.module_go_enrichment_plot(shown_modules = program_list[:10], go_per_module = 1)

# %%
# Visualize the M8 network with nodes highlighted by a selected GO term.
# Program M8 is associated with B cell.
M8_GO_Enrich = ggm.go_enrichment[ggm.go_enrichment['module_id'] == 'M8']
print(M8_GO_Enrich.iloc[:3, :6])
ggm.module_network_plot(module_id = 'M8', highlight_anno = "B cell receptor signaling pathway", seed = 3, layout_iterations = 55)


# %%
# Print a summary of the GGM analysis.
print(ggm)

# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, "data/Human_Tonsil_HD.ggm.h5")

# %% [markdown]
#### Part 2: Spot annotation based on program expression ###

# %%
# Compute per-spot expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)


# %%
# Visualize the spatial distribution of the top 20 program-expression scores.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = program_list[:20], cmap = 'Reds', ncols = 4)


# %%
# Compute pairwise program similarity and plot the correlation heatmap with dendrograms.
sg.module_similarity_plot(adata, ggm_key = 'ggm', corr_method = 'pearson', heatmap_metric = 'correlation', 
                          fig_height = 13, fig_width = 14, dendrogram_height = 0.1, dendrogram_space = 0.08, return_summary = False)


# %%
# Assign spot-level annotations via Gaussian Mixture Models (GMMs) based on program expression.
sg.calculate_gmm_annotations(adata, ggm_key = 'ggm')


# %%
# Optionally smooth the annotations using spatial k-NN (on the 'spatial' embedding).
sg.smooth_annotations(adata, ggm_key = 'ggm', embedding_key = 'spatial', k_neighbors = 24)


# %%
# Display smoothed annotations for top 20 programs.
# If smoothing is skipped, use 'M1_anno' … 'M20_anno' instead.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_anno_smooth'
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = program_list[:20], legend_loc = None, ncols = 4)
# Where the blue nodes indicate the spots annotated by the program, and gray nodes are unassigned.


# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key = 'ggm', result_anno = 'ggm_annotation', neighbor_similarity_ratio = 0.6)
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.


# %%
# Visualize the integrated annotation.
plt.rcParams["figure.figsize"] = (4, 6)
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = ['ggm_annotation'], frameon = False, title = 'Integrated annotation')


# %% [markdown]
#### Part 3: Cluster spots based a dimensionality reduction of program expression ###


# %%
# Build a neighborhood graph based on program expression and perform clustering.
sc.pp.neighbors(adata, 
                use_rep='module_expression_scaled',
                n_pcs=adata.obsm['module_expression_scaled'].shape[1])
sc.tl.leiden(adata, resolution=2, key_added='leiden_ggm')
sc.tl.louvain(adata, resolution=2, key_added='louvan_ggm')

# %%
# Visualize the clustering results.
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = ['leiden_ggm'], frameon = False, title = 'Leiden clustering')
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, size = 1.2, alpha_img = 0.5, bw = True, color = ['louvan_ggm'], frameon = False, title = 'Louvain clustering')

# %%
# Summarize program-expression across the leiden clusters clusters as a dot plot.
sg.module_dot_plot(adata, ggm_key = 'ggm', groupby = 'leiden_ggm', scale=True,
                   dendrogram_height = 0.15, dendrogram_space = 0.05, fig_height=8, fig_width = 14, axis_fontsize = 10)

# %%
# Save the annotated AnnData object.
adata.write("data/Human_Tonsil_HD_ggm_anno.h5ad")


# %%
adata = sc.read_h5ad("data/Human_Tonsil_HD_ggm_anno.h5ad")
ggm = sg.load_ggm("data/Human_Tonsil_HD.ggm.h5")
# %%
