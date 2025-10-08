# %% [markdown]
# # Tutorial 1: Stereo-Seq: MOSTA E16.5 E2S5

# %% [markdown]
# <div style="margin:0; line-height:1.2">
# Analyze MOSTA E16.5 E2S5 Spatial Transcriptomics data with SpacGPA.<br/>  
# 
# Data source: https://db.cngb.org/stomics/mosta/ <br/>
# 
# MOSTA E16.5 E2S5 is a mouse embryo sample at embryonic day 16.5 generated with Stereo-seq.<br/>  
# <div>

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


# %% [markdown]
# ### Part 1: Gene program analysis via SpacGPA ####

# %%
# Load spatial transcriptomics data.
adata = sc.read_h5ad("data/Stereo-seq/MOSTA/E16.5_E2S5.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata)


# %%
# Preprocessing: use raw counts from layers['count'], then library-size normalize and log1p-transform.
adata.X = adata.layers['count']
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes = 200)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata.X.shape)


# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata, project_name = "E16.5_E2S5")  


# %%
# Show statistically significant co-expression gene pairs.
print(ggm.SigEdges.head(5))


# %%
# Identify gene programs using the MCL-Hub algorithm (set inflation to 2).
ggm.find_modules(method = 'mcl-hub', inflation = 2)


# %%
# Inspect the top 5 identified gene programs.
print(ggm.modules_summary.head(5))


# %%
# Visualize the subnetwork of program M1 (top 30 genes by degree/connectivity for readability).
ggm.module_network_plot(module_id='M1', seed = 2) 
# Fix layout randomness for reproducibility via set seed.

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.go_enrichment_analysis(species = 'mouse', padjust_method = "BH", pvalue_cutoff = 0.05)


# %%
# Visualize top enriched GO terms for programs M1–M5.
ggm.module_go_enrichment_plot(shown_modules = ['M1', 'M2', 'M3', 'M4', 'M5'], go_per_module = 1)

# %%
# Mammalian Phenotype (MP) Ontology enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.mp_enrichment_analysis(species = 'mouse', padjust_method = "BH", pvalue_cutoff = 0.05)


# %%
# Visualize top enriched MP terms for programs M1–M5.
ggm.module_mp_enrichment_plot(shown_modules = ['M1', 'M2', 'M3', 'M4', 'M5'], mp_per_module = 1)

# %%
# Visualize the M1 network with nodes highlighted by a selected GO or MP term ID.
print(ggm.go_enrichment.iloc[0, :6])
ggm.module_network_plot(module_id='M1', highlight_anno = "GO:0030016", seed = 2)


# %%
# The short name of Ontology term also works when highlighting (e.g., "muscle phenotype" for MP:0005369).
print(ggm.mp_enrichment.iloc[0, :5])
ggm.module_network_plot(module_id='M1', highlight_anno = "muscle phenotype", seed = 2)


# %%
# Print a summary of the GGM analysis.
print(ggm)


# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, "data/MOSTA_E16.5_E2S5.ggm.h5")
# Then you can reload it via:
# ggm = sg.load_ggm("data/MOSTA_E16.5_E2S5.ggm.h5")


# %%
# You can also save those results (e.g., modules, GO/MP enrichment) to CSV files.
ggm.modules.to_csv('data/MOSTA_E16.5_E2S5_modules.csv')
ggm.modules_summary.to_csv('data/MOSTA_E16.5_E2S5_modules_summary.csv')
ggm.go_enrichment.to_csv('data/MOSTA_E16.5_E2S5_go_enrichment.csv')
ggm.mp_enrichment.to_csv('data/MOSTA_E16.5_E2S5_mp_enrichment.csv')
# Which can lead to further downstream analyses with other tools.

# %% [markdown]
# ### Part 2: Spot annotation based on program expression ###

# %%
# Compute per-spot expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)

# NOTE — OpenBLAS warning explanation & fix:
# If you see: "OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata",
# it means your OpenBLAS was compiled with a smaller thread cap than your machine exposes.
# This is harmless but noisy and may slightly impact performance.
# Workaround: limit BLAS threads at the VERY TOP of the pipeline, BEFORE importing any libraries:
#   import os
#   os.environ["OPENBLAS_NUM_THREADS"] = "32"   # choose a value ≤ the limit mentioned in the warning (e.g., 64)

# %%
# Visualize the spatial distribution of the top 20 program-expression scores.
plt.rcParams["figure.figsize"] = (7, 7)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, spot_size = 2, color = program_list[:20], cmap = 'RdYlBu_r', ncols = 4)


# %%
# Compute pairwise program similarity and plot the correlation heatmap with dendrograms.
sg.module_similarity_plot(adata, ggm_key = 'ggm', corr_method = 'pearson', heatmap_metric = 'correlation', 
                          fig_height = 20, fig_width = 21, dendrogram_height = 0.1, dendrogram_space = 0.05, return_summary = False)


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
sc.pl.spatial(adata, spot_size = 2, color = program_list[:20], legend_loc = None, ncols = 4)
# Where the blue nodes indicate the spots annotated by the program, and gray nodes are unassigned.


# %%
# Summarize program-expression across the existing annotation categories as a dot plot.
sg.module_dot_plot(adata, ggm_key = 'ggm', groupby = 'annotation', scale=True,
                   dendrogram_height = 0.1, dendrogram_space = 0.2, fig_width = 24, axis_fontsize = 10)


# %%
# Integrate multiple program-derived annotations into a single label set via sg.integrate_annotations.
sg.integrate_annotations(adata, ggm_key = 'ggm', neighbor_similarity_ratio = 0.6, result_anno = 'ggm_annotation')
# Here we integrate all programs as an example. You can specify a subset of programs via the 'modules_used' parameter.


# %%
# Visualize the integrated annotation and compare it to the existing annotation.
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, spot_size = 2, color = ['ggm_annotation'], frameon = False, title = 'Integrated annotation')
plt.rcParams["figure.figsize"] = (3, 6)
sc.pl.spatial(adata, spot_size = 2, color = ['annotation'], frameon = False, title = 'Original annotation')


# %%
# Save the annotated AnnData object.
adata.write("data/MOSTA_E16.5_E2S5_ggm_anno.h5ad")



