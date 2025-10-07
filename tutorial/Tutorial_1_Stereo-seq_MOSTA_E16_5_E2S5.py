
# %%
# Analyze MOSTA E16.5 E2S5 Spatial Transcriptomics Data using SpacGPA.
# Data source: https://db.cngb.org/stomics/mosta/
# MOSTA E16.5 E2S5 is a mouse embryo sample at embryonic day 16.5 which is sequenced by Stereo-seq technology.

# %%
# Import SpacGPA and other required packages.
import SpacGPA as sg
import scanpy as sc
import matplotlib.pyplot as plt
import os

# %%
# Set working directory as your own path.
workdir = '..'
os.chdir(workdir)

### Part 1: Gene program analysis via SpacGPA ####
# %%
# Load Spatial Transcriptomics data.
adata = sc.read_h5ad("data/Stereo-seq/MOSTA/E16.5_E2S5.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata)

# %%
# Data preprocessing.
adata.X = adata.layers['count']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
print(adata.X.shape)

# %%
# Calculate Coexpression Network using SpacGPA.
ggm = sg.create_ggm(adata,project_name = "E16.5_E2S5")  

# %%
# Show significant co-expression gene pairs.
print(ggm.SigEdges.head(5))

# %%
# Identify gene programs via MCL-Hub algorithm.
ggm.find_modules(method='mcl-hub',inflation=2)

# %%
# Show identified top 5 gene programs.
print(ggm.modules_summary.head(5))

# %%
# Visualize network of program M1 via sg.module_network_plot.
# Only top 30 genes with highest connectivity are shown for clarity.
M1_edges = ggm.get_module_edges('M1')
M1_anno = ggm.get_module_anno('M1')
sg.module_network_plot(
    nodes_edges = M1_edges,
    nodes_anno = M1_anno,
    seed=2 # Set random seed for a suitable layout
)

# %%
# GO Enrichment Analysis.
ggm.go_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
# Visualize most enrichment GO terms for programs M1-M5 via sg.module_go_enrichment_plot.
sg.module_go_enrichment_plot(ggm, top_n_modules=5, go_per_module=2)

# %%
# MP Enrichment Analysis.
ggm.mp_enrichment_analysis(species='mouse',padjust_method="BH",pvalue_cutoff=0.05)

# %%
# Visualize most enrichment MP terms for programs M1-M5 via sg.module_mp_enrichment_plot.
sg.module_mp_enrichment_plot(ggm, top_n_modules=5, mp_per_module=2)

# %%
# Visualize network of program M1 with highlighted GO or MP terms.
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
# Print GGM object to get summary of GGM analysis.
print(ggm)

# %%
# Save GGM object as h5 file for future use
sg.save_ggm(ggm, "data/MOSTA_E16.5_E2S5.ggm.h5")
# Then you can load GGM object via sg.load_ggm function
# ggm = sg.load_ggm("data/MOSTA_E16.5_E2S5.ggm.h5")



#### Part 2: Spot annotation based on program expression ####
# %%
# Calculate program expression in each spot via sg.calculate_module_expression.
sg.calculate_module_expression(adata, ggm)

# %%
# Show expression distribution of top 20 programs.
sc.pl.spatial(adata, spot_size = 2, color=['M1_exp','M2_exp','M3_exp','M4_exp','M5_exp',
                                           'M6_exp','M7_exp','M8_exp','M9_exp','M10_exp',
                                           'M11_exp','M12_exp','M13_exp','M14_exp','M15_exp',
                                           'M16_exp','M17_exp','M18_exp','M19_exp','M20_exp'], 
              cmap='RdYlBu_r',ncols=5)

# %%
# Calculate correlation between programs and visualize correlation heatmap.
mod_cor = sg.module_similarity_plot(adata,
                                    ggm_key='ggm',
                                    corr_method='pearson',
                                    heatmap_metric='correlation',   
                                    fig_height=20,
                                    fig_width=21,
                                    dendrogram_height=0.1,
                                    dendrogram_space=0.05)

# %%
# Anno spots with GMM based on program expression.
sg.calculate_gmm_annotations(adata, ggm_key='ggm')
# Smooth annotations based on spatial neighbors. (Optional)
sg.smooth_annotations(adata, ggm_key='ggm', embedding_key='spatial', k_neighbors=24)

# %%
# Show annotated spatial domains by top 20 programs, with a comparison to original annotation.
# 'annotation' is the original annotation in the anndata object.
# If you did not run sg.smooth_annotations, Use 'M1_anno' to 'M20_anno' instead.
sc.pl.spatial(adata, spot_size = 2, color=['M1_anno_smooth','M2_anno_smooth','M3_anno_smooth','M4_anno_smooth','M5_anno_smooth',
                                           #'M6_anno_smooth','M7_anno_smooth','M8_anno_smooth','M9_anno_smooth','M10_anno_smooth',
                                           #'M11_anno_smooth','M12_anno_smooth','M13_anno_smooth','M14_anno_smooth','M15_anno_smooth',
                                           #'M16_anno_smooth','M17_anno_smooth','M18_anno_smooth','M19_anno_smooth','M20_anno_smooth',
                                           'annotation'], 
              ncols=5)

# %%
# Integrate annotations.
sg.integrate_annotations(adata, ggm_key='ggm',result_anno='ggm_annotation')

# %%
# Visualize integrated annotation and compare with original annotation.
sc.pl.spatial(adata, spot_size = 2, color=['ggm_annotation'], frameon=False, title='Integrated annotation')
sc.pl.spatial(adata, spot_size = 2, color=['annotation'], frameon=False, title='Original annotation')

# %%
# Save annotated anndata object.
adata.write("data/MOSTA_E16.5_E2S5_ggm_anno.h5ad")
