# %% [markdown]
## Tutorial 6: Atera: Breast Tumor

# %% [markdown]
# Analyze 10x Atera breast tumor spatial transcriptomics data with SpacGPA.
#
# The Atera data folder used in this tutorial contains three required files:
# cell_feature_matrix.h5, cells.csv.gz, and cell_boundaries.csv.gz.

# %%
# Import SpacGPA and other required packages.
import SpacGPA as sg
import scanpy as sc
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', message=r'.*Variable names are not unique.*')

# %%
# Set the working directory to your local path.
workdir = '..'
os.chdir(workdir)

# %% [markdown]
#### Part 1: Gene program analysis via SpacGPA ###

# %%
# Load Atera data.
adata = sg.load_atera('data/Atera/Breast_Tumor')
print(adata)

# %%
# Preprocessing: library-size normalize and log1p-transform.
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes = 200)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata.X.shape)

# %%
# Construct the co-expression network using SpacGPA (Gaussian graphical model).
ggm = sg.create_ggm(adata, project_name = 'Breast Tumor')

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
# Visualize the subnetwork of program M3.
ggm.module_network_plot(module_id = 'M3', seed = 2, layout_iterations = 60)
# Fix layout randomness for reproducibility via set seed.

# %%
# Gene Ontology (GO) enrichment analysis with BH FDR control and p-value threshold 0.05.
ggm.go_enrichment_analysis(species = 'human', padjust_method = 'BH', pvalue_cutoff = 0.05)

# %%
# Visualize top enriched GO terms for top 10 identified programs.
program_list = ggm.modules_summary['module_id'].tolist()
ggm.module_go_enrichment_plot(shown_modules = program_list[:10], go_per_module = 1)

# %%
# Visualize the M3 network with nodes highlighted by a selected GO term.
M3_GO_Enrich = ggm.go_enrichment[ggm.go_enrichment['module_id'] == 'M3']
print(M3_GO_Enrich.iloc[:3, :6])
ggm.module_network_plot(module_id = 'M3', highlight_anno = 'focal adhesion', seed = 2, layout_iterations = 60)

# %%
# Print a summary of the GGM analysis.
print(ggm)

# %%
# Save the GGM object to HDF5 for later reuse.
sg.save_ggm(ggm, 'data/Atera_Breast_Tumor.ggm.h5')

# %% [markdown]
#### Part 2: Cell annotation based on program expression ###

# %%
# Compute per-cell expression scores of each gene program.
sg.calculate_module_expression(adata, ggm)

# %%
# Visualize the spatial distribution of the top 20 program-expression scores.
plt.rcParams['figure.figsize'] = (7, 4)
program_list = ggm.modules_summary['module_id'] + '_exp'
sc.pl.spatial(adata, spot_size = 30, color = program_list[:20], cmap = 'Reds', ncols = 5)

# %%
# Visualize the expression of M1 with cell boundary polygons.
sg.pl.spatial_polygon(adata, color = 'M1_exp', figsize = (40, 20), background = 'black')

# %%
# Visualize the expression of M6 with cell boundary polygons.
sg.pl.spatial_polygon(adata, color = 'M6_exp', figsize = (40, 20), background = 'black')

# %%
# Save the annotated AnnData object.
adata.write('data/Atera_Breast_Tumor_ggm_anno.h5ad')

# %%