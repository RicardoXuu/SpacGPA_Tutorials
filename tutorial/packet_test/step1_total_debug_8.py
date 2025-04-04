
# %%
# 一些问题修复
# 使用 CytAssist_FreshFrozen_Mouse_Brain_Rep2 数据
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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# %%
# 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
#from SpacGPA import *
import SpacGPA as sg

# %%
# 读取 ggm
start_time = time.time()
ggm = sg.load_ggm("data/ggm_gpu_32.h5")
print(f"Read ggm: {time.time() - start_time:.5f} s")
# 读取联合分析的ggm
ggm_mulit_intersection = sg.load_ggm("data/ggm_mulit_intersection.h5")
print(f"Read ggm_mulit_intersection: {time.time() - start_time:.5f} s")
ggm_mulit_union = sg.load_ggm("data/ggm_mulit_union.h5")
print(f"Read ggm_mulit_union: {time.time() - start_time:.5f} s")
print("=====================================")
print(ggm)
print("=====================================")
print(ggm_mulit_intersection)
print("=====================================")
print(ggm_mulit_union)


# %%
ggm.round_num



# %%
# 读取数据
adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                       count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)


# %%
# 更新find_best_inflation的计算策略
import numpy as np
import pandas as pd
import networkx as nx
import torch
import sys
import gc
import matplotlib.pyplot as plt

def run_mcl_qc(SigEdges, expansion=2, inflation=1.7, add_self_loops='mean', 
               max_iter=1000, tol=1e-6, pruning_threshold=1e-5, 
               run_mode=2):
    """
    Quickly perform MCL-Hub clustering on input SigEdges (DataFrame with 'GeneA', 'GeneB', 'Pcor')
    until the candidate node integration stage and return the clustering result.
    
    Parameters:
        SigEdges: DataFrame with 'GeneA', 'GeneB', and 'Pcor' columns.
        expansion: Exponent for the expansion step (default 2).
        inflation: Exponent for the inflation step (typically >1, default 1.7).
        add_self_loops: Method for adding self-loops. Options: 'mean', 'min', 'max', 'dynamic', 'none' (default 'mean').
        max_iter: Maximum iterations (default 1000).
        tol: Convergence threshold (default 1e-6).
        pruning_threshold: Elements below this threshold are set to zero (default 1e-5).
        run_mode: 0 for CPU, non-zero for GPU (default 2, uses GPU).
        
    Returns:
        clusters: A list of sets, each set contains indices of genes (according to SigEdges order) in one cluster.
    """
    try:
        # 1. Extract unique genes and create a mapping from gene to index.
        genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
        gene_index = {gene: i for i, gene in enumerate(genes)}
        n = len(genes)
        
        # Build a symmetric weighted adjacency matrix using 'Pcor' values.
        M = np.zeros((n, n), dtype=np.float32)
        for _, row in SigEdges.iterrows():
            i = gene_index[row['GeneA']]
            j = gene_index[row['GeneB']]
            M[i, j] = row['Pcor']
            M[j, i] = row['Pcor']
        
        # Construct the original graph (for candidate node integration) with nodes as indices.
        G_ori = nx.from_numpy_array(M)
        
        # 2. Add self-loops according to the specified method.
        if add_self_loops == 'mean':
            np.fill_diagonal(M, np.mean(M[M > 0]))
        elif add_self_loops == 'min':
            np.fill_diagonal(M, np.min(M[M > 0]))
        elif add_self_loops == 'max':
            np.fill_diagonal(M, np.max(M))
        elif add_self_loops == 'dynamic':
            for col_idx in range(M.shape[1]):
                nonzero_elements = M[:, col_idx][M[:, col_idx] > 0]
                if len(nonzero_elements) > 0:
                    M[col_idx, col_idx] = np.mean(nonzero_elements)
                else:
                    M[col_idx, col_idx] = np.min(M[M > 0])
        elif add_self_loops == 'none':
            pass
        else:
            raise ValueError("Invalid value for 'add_self_loops'. Choose from 'mean', 'min', 'max', 'dynamic', or 'none'.")
        
        # 3. Convert the matrix to a PyTorch tensor and set the computation device.
        device = 'cpu' if run_mode == 0 else 'cuda'
        M = torch.tensor(M, device=device)
        
        # Define a function for column normalization (each column sums to 1).
        def normalize_columns(matrix):
            col_sums = matrix.sum(dim=0)
            return matrix / col_sums
        
        # Initial normalization.
        M = normalize_columns(M)
        
        # 4. MCL iteration: expansion, inflation, normalization, pruning, and convergence check.
        for iteration in range(max_iter):
            M_prev = M.clone()
            M = torch.matrix_power(M, expansion)   # Expansion step
            M = M.pow(inflation)                     # Inflation step
            M = normalize_columns(M)
            M[M < pruning_threshold] = 0            # Pruning step
            M = normalize_columns(M)
            diff = torch.max(torch.abs(M - M_prev))
            if diff < tol:
                break
        
        # 5. Convert the final matrix back to NumPy and construct a graph for candidate integration.
        M_np = M.cpu().numpy()
        G = nx.from_numpy_array(M_np)
        del M, M_np  # Release memory
        
        # 6. Extract connected components as initial clusters.
        clusters = list(nx.connected_components(G))
        # Treat single-node clusters as candidate nodes and remove them from clusters.
        size1_clusters = [cluster for cluster in clusters if len(cluster) == 1]
        candidate_nodes = set()
        for cluster in size1_clusters:
            for node in cluster:
                candidate_nodes.add(node)
        clusters = [cluster for cluster in clusters if len(cluster) > 1]
        
        # 7. Helper function to compute the shortest path.
        def find_shortest_path(net, source, target):
            try:
                return nx.shortest_path(net, source=source, target=target)
            except nx.NetworkXNoPath:
                return None
        
        # 8. For each cluster (sorted in descending order by size), find zero-degree nodes 
        # and use candidate nodes to supplement the cluster.
        clusters_sorted = sorted(clusters, key=lambda x: len(x), reverse=True)
        for cluster in clusters_sorted:
            subgraph = G.subgraph(cluster)
            degrees = dict(subgraph.degree())
            if not degrees:
                continue
            attractor = max(degrees, key=degrees.get)  # Select the node with highest degree
            subgraph_ori = G_ori.subgraph(cluster)
            degree_zero_nodes = [node for node, deg in subgraph_ori.degree() if deg == 0]
            if degree_zero_nodes:
                net_temp = G_ori.subgraph(set(candidate_nodes) | cluster)
                for node in degree_zero_nodes:
                    path = find_shortest_path(net_temp, node, attractor)
                    if path:
                        for n in path:
                            if n in candidate_nodes:
                                candidate_nodes.remove(n)
                        cluster.update(path)
        
        # 9. For each cluster, move zero-degree nodes to the candidate set.
        for cluster in clusters:
            subgraph_ori = G_ori.subgraph(cluster)
            for node, deg in list(subgraph_ori.degree()):
                if deg == 0:
                    cluster.remove(node)
                    candidate_nodes.add(node)
        
        # 10. Map nodes to clusters for candidate assignment.
        node_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = idx
        
        # 11. For each candidate node, assign it to the best matching cluster based on neighbor counts.
        for node in list(candidate_nodes):
            neighbors = set(G_ori.neighbors(node))
            if not neighbors:
                continue
            cluster_count = {}
            for nbr in neighbors:
                if nbr in node_to_cluster:
                    cid = node_to_cluster[nbr]
                    cluster_count[cid] = cluster_count.get(cid, 0) + 1
            total_neighbors = sum(cluster_count.values())
            for cid, count in cluster_count.items():
                if count >= total_neighbors / 2:
                    clusters[cid].add(node)
                    node_to_cluster[node] = cid
                    candidate_nodes.remove(node)
                    break
        
        # Return the clustering result: each cluster is a set of node indices.
        return clusters
    
    finally:
        gc.collect()
        if run_mode != 0:
            torch.cuda.empty_cache()

def build_original_graph(SigEdges, add_self_loops='mean'):
    """
    Constructs the original network graph from SigEdges for modularity computation.
    Nodes are represented as indices.
    """
    genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
    gene_index = {gene: i for i, gene in enumerate(genes)}
    n = len(genes)
    M = np.zeros((n, n), dtype=np.float32)
    for _, row in SigEdges.iterrows():
        i = gene_index[row['GeneA']]
        j = gene_index[row['GeneB']]
        M[i, j] = row['Pcor']
        M[j, i] = row['Pcor']
    if add_self_loops == 'mean':
        np.fill_diagonal(M, np.mean(M[M > 0]))
    elif add_self_loops == 'min':
        np.fill_diagonal(M, np.min(M[M > 0]))
    elif add_self_loops == 'max':
        np.fill_diagonal(M, np.max(M))
    elif add_self_loops == 'dynamic':
        for col_idx in range(M.shape[1]):
            nonzero_elements = M[:, col_idx][M[:, col_idx] > 0]
            if len(nonzero_elements) > 0:
                M[col_idx, col_idx] = np.mean(nonzero_elements)
            else:
                M[col_idx, col_idx] = np.min(M[M > 0])
    elif add_self_loops == 'none':
        pass
    G_ori = nx.from_numpy_array(M)
    return G_ori

def find_best_inflation(SigEdges, min_inflation=1.1, max_inflation=5,
                        coarse_step=0.1, mid_step=0.05, fine_step=0.01,
                        expansion=2, add_self_loops='mean', max_iter=1000,
                        tol=1e-6, pruning_threshold=1e-5, run_mode=2,
                        phase=3, show_plot=False):
    """
    Optimize the inflation parameter for MCL clustering using SigEdges data by 
    performing a three-phase search and evaluating clustering quality using the 
    NetworkX modularity metric.

    Phases:
      - Phase 1 (Coarse Search): Starting from min_inflation, increment by coarse_step.
        If, within the first three evaluations, both the second and third modularity 
        values are lower than the first, then min_inflation is considered too high and 
        the process terminates. Otherwise, if modularity decreases in two consecutive 
        steps, the phase stops and the inflation with the highest modularity in this 
        phase (max_coarse) is selected.
      - Phase 2 (Mid-step Search): Search the range [max_coarse - coarse_step, max_coarse + coarse_step]
        with a step of mid_step to obtain an intermediate best value (max_mid).
      - Phase 3 (Fine Search): Search the range [max_mid - mid_step, max_mid + mid_step]
        with a step of fine_step to obtain the final optimal inflation value (max_fine).

    The 'phase' parameter controls execution:
      - phase = 1: Execute only Phase 1.
      - phase = 2: Execute Phases 1 and 2.
      - phase = 3: Execute all three phases.

    All evaluation points are recorded, and an optional scatter plot is produced (if show_plot=True)
    with the x-axis ticks set at mid_step intervals. The best point is marked with a red dot,
    and dashed lines are drawn at the best inflation and modularity values.

    Parameters:
      SigEdges: DataFrame containing 'GeneA', 'GeneB', and 'Pcor' columns.
      max_inflation: Maximum inflation value for the search.
      min_inflation: Minimum inflation value allowed (default 1.1).
      coarse_step: Step size for coarse search (default 0.1).
      mid_step: Step size for mid-step search (default 0.05).
      fine_step: Step size for fine search (default 0.01).
      expansion, add_self_loops, max_iter, tol, pruning_threshold, run_mode:
          Parameters for run_mcl_qc controlling the MCL clustering.
      phase: Which phase to execute up to (1, 2, or 3; default 3).
      show_plot: Whether to display a scatter plot (default False).

    Returns:
      (best_inflation, best_modularity)
    """
    # Build the original graph for modularity computation.
    G_ori = build_original_graph(SigEdges, add_self_loops=add_self_loops)
    
    # Cache to store computed modularity values to avoid redundant calculations.
    eval_cache = {}
    # Global lists to record evaluation points (used for plotting).
    global_inflations = []
    global_modularities = []
    
    def evaluate_inflation(inflation):
        if inflation in eval_cache:
            return eval_cache[inflation]
        clusters = run_mcl_qc(SigEdges, expansion=expansion, inflation=inflation, 
                                add_self_loops=add_self_loops, max_iter=max_iter, 
                                tol=tol, pruning_threshold=pruning_threshold, 
                                run_mode=run_mode)
        # Ensure all nodes are assigned (unassigned nodes become singleton communities).
        all_nodes = set(G_ori.nodes())
        clustered_nodes = set().union(*clusters)
        missing_nodes = all_nodes - clustered_nodes
        for node in missing_nodes:
            clusters.append({node})
        if len(clusters) <= 1:
            Q = 0.0
        else:
            Q = nx.algorithms.community.modularity(G_ori, clusters, weight='weight')
        eval_cache[inflation] = Q
        return Q

    best_inflation = None
    best_modularity = -np.inf

    # -------------------- Phase 1: Coarse Search (Ascending) --------------------
    print("Phase 1: Coarse search")
    inflation_val = min_inflation
    coarse_results = []
    consecutive_decrease = 0
    prev_Q = None
    first_Q = None
    second_Q = None
    third_Q = None

    while inflation_val <= max_inflation:
        Q = evaluate_inflation(inflation_val)
        coarse_results.append((inflation_val, Q))
        global_inflations.append(inflation_val)
        global_modularities.append(Q)
        print(f"Inflation = {inflation_val:.2f}, Modularity = {Q:.4f}")
        
        # Check the first three evaluations.
        if len(coarse_results) == 1:
            first_Q = Q
        elif len(coarse_results) == 2:
            second_Q = Q
        elif len(coarse_results) == 3:
            third_Q = Q
            if second_Q < first_Q and third_Q < first_Q:
                print("\nTerminating: min_inflation is set too high; initial modularity exceeds subsequent values. Please adjust min_inflation.")
                return None, None
        
        if prev_Q is not None:
            if Q < prev_Q:
                consecutive_decrease += 1
            else:
                consecutive_decrease = 0
        prev_Q = Q
        
        if consecutive_decrease >= 2:
            break
        inflation_val = round(inflation_val + coarse_step, 4)
    
    max_coarse = max(coarse_results, key=lambda x: x[1])[0]
    print(f"Coarse search best inflation: {max_coarse:.2f}")
    
    if phase == 1:
        best_inflation, best_modularity = max_coarse, evaluate_inflation(max_coarse)
        print(f"\nBest inflation (Phase 1): {best_inflation:.2f}, Modularity: {best_modularity:.4f}")
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            x_min, x_max = min(global_inflations), max(global_inflations)
            ticks = np.arange(round(x_min,2), round(x_max + mid_step,2), mid_step)
            plt.xticks(ticks)
            plt.scatter(global_inflations, global_modularities, c='blue', alpha=0.6, edgecolors='k')
            plt.scatter([best_inflation], [best_modularity], c='red', s=100, edgecolors='k', label="Best")
            plt.axvline(x=best_inflation, linestyle='--', color='lightgrey')
            plt.axhline(y=best_modularity, linestyle='--', color='lightgrey')
            plt.xlabel("Inflation")
            plt.ylabel("Modularity")
            plt.title("Inflation vs. Modularity")
            plt.grid(False)
            plt.tight_layout()
            plt.legend()
            plt.show()
        return best_inflation, best_modularity
    
    # -------------------- Phase 2: Mid-step Search --------------------
    print("\nPhase 2: Mid-step search")
    lower_bound = max(min_inflation, max_coarse - coarse_step)
    upper_bound = min(max_inflation, max_coarse + coarse_step)
    mid_results = []
    inflation_vals_mid = np.arange(lower_bound, upper_bound + mid_step/2, mid_step)
    for inflation_val in inflation_vals_mid:
        inflation_val = round(inflation_val, 4)
        Q = evaluate_inflation(inflation_val)
        mid_results.append((inflation_val, Q))
        if inflation_val not in global_inflations:
            global_inflations.append(inflation_val)
            global_modularities.append(Q)
        print(f"Inflation = {inflation_val:.2f}, Modularity = {Q:.4f}")
    
    max_mid = max(mid_results, key=lambda x: x[1])[0]
    print(f"Mid-step search best inflation: {max_mid:.2f}")
    
    if phase == 2:
        best_inflation, best_modularity = max_mid, evaluate_inflation(max_mid)
        print(f"\nBest inflation (Phase 2): {best_inflation:.2f}, Modularity: {best_modularity:.4f}")
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            x_min, x_max = min(global_inflations), max(global_inflations)
            ticks = np.arange(round(x_min,2), round(x_max + mid_step,2), mid_step)
            plt.xticks(ticks)
            plt.scatter(global_inflations, global_modularities, c='blue', alpha=0.6, edgecolors='k')
            plt.scatter([best_inflation], [best_modularity], c='red', s=100, edgecolors='k', label="Best")
            plt.axvline(x=best_inflation, linestyle='--', color='lightgrey')
            plt.axhline(y=best_modularity, linestyle='--', color='lightgrey')
            plt.xlabel("Inflation")
            plt.ylabel("Modularity")
            plt.title("Inflation vs. Modularity")
            plt.grid(False)
            plt.tight_layout()
            plt.legend()
            plt.show()
        return best_inflation, best_modularity
    
    # -------------------- Phase 3: Fine Search --------------------
    print("\nPhase 3: Fine search")
    lower_bound = max(min_inflation, max_mid - mid_step)
    upper_bound = min(max_inflation, max_mid + mid_step)
    fine_results = []
    inflation_vals_fine = np.arange(lower_bound, upper_bound + fine_step/2, fine_step)
    for inflation_val in inflation_vals_fine:
        inflation_val = round(inflation_val, 4)
        Q = evaluate_inflation(inflation_val)
        fine_results.append((inflation_val, Q))
        if inflation_val not in global_inflations:
            global_inflations.append(inflation_val)
            global_modularities.append(Q)
        print(f"Inflation = {inflation_val:.2f}, Modularity = {Q:.4f}")
    
    best_inflation, best_modularity = max(fine_results, key=lambda x: x[1])
    print(f"\nBest inflation (Phase 3): {best_inflation:.2f}, Modularity: {best_modularity:.4f}")
    
    # Plotting (Optional)
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        x_min, x_max = min(global_inflations), max(global_inflations)
        ticks = np.arange(round(x_min,2), round(x_max + mid_step,2), mid_step)
        plt.xticks(ticks)
        plt.scatter(global_inflations, global_modularities, c='blue', alpha=0.6, edgecolors='k')
        plt.scatter([best_inflation], [best_modularity], c='red', s=100, edgecolors='k', label="Best")
        plt.axvline(x=best_inflation, linestyle='--', color='lightgrey')
        plt.axhline(y=best_modularity, linestyle='--', color='lightgrey')
        plt.xlabel("Inflation")
        plt.ylabel("Modularity")
        plt.title("Inflation vs. Modularity")
        plt.grid(False)
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    return best_inflation, best_modularity


# %%
start_time = time.time()
best_inf_1, best_mod_1 = sg.find_best_inflation(ggm, max_inflation=2.5)
print(f"Time1: {time.time() - start_time:.5f} s")

# %%
start_time = time.time()
best_inf_2, best_mod_2 = find_best_inflation(ggm.SigEdges, min_inflation=1.1, phase=3, show_plot=True)
print(f"Time2: {time.time() - start_time:.5f} s")

# %%
ggm.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=best_inf_1, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm.modules_summary)


# %%
ggm.SigEdges
# %%



# %%
sc.pp.filter_genes(adata,min_cells=10)
print(adata.X.shape)

# %%
ggm_1 = sg.create_ggm(adata,
                    project_name = "CytAssist_FreshFrozen_Mouse_Brain_Rep2", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.05,
                    auto_adjust=True,
                    auto_find_modules=True,
                    )  
print(ggm_1.SigEdges)

# %%
ggm_2 = sg.create_ggm(adata,
                    project_name = "CytAssist_FreshFrozen_Mouse_Brain_Rep2", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.05,
                    auto_adjust=True,
                    auto_find_modules=True,
                    )  
print(ggm_2.SigEdges)

# %%
ggm_3 = sg.create_ggm(adata,
                    project_name = "CytAssist_FreshFrozen_Mouse_Brain_Rep2", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.05,
                    auto_adjust=True,
                    auto_find_modules=True,
                    )  
print(ggm_3.SigEdges)

# %%
ggm_4 = sg.create_ggm(adata,
                    project_name = "CytAssist_FreshFrozen_Mouse_Brain_Rep2", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.05,
                    auto_adjust=True,
                    auto_find_modules=True,
                    )  
print(ggm_4.SigEdges)

# %%
ggm.modules_summary

# %%
ggm_1.modules_summary

# %%
ggm_2.modules_summary

# %%
ggm_3.modules_summary

# %%
ggm_4.modules_summary

# %%
ggm_1
# %%
ggm_2
# %%
ggm_3           
# %%
ggm_4
# %%

# %%


# %%
# 重新读取数据
adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                       count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)

# %%
sg.annotate_with_ggm(adata, ggm_1,
                     ggm_key='ggm_1')
# %%
sg.annotate_with_ggm(adata, ggm_2,
                     ggm_key='ggm_2')
# %%
sg.annotate_with_ggm(adata, ggm_3,
                     ggm_key='ggm_3')
# %%
sg.annotate_with_ggm(adata, ggm_4,
                     ggm_key='ggm_4')

# %%

# %%

# %%
ggm_1.adjust_cutoff(pcor_threshold=0.075)
best_inf_1, _ = sg.find_best_inflation(ggm_1, min_inflation=1.1, phase=3, show_plot=True)
ggm_1.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=best_inf_1, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm_1.modules_summary.shape)
sg.annotate_with_ggm(adata, ggm_1,
                     ggm_key='ggm_1')
sg.integrate_annotations(adata,
                        ggm_key='ggm_1',
                        result_anno='annotation_1',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_1", show=True)

# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm_2',
                        result_anno='annotation_2',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_2", show=True)

# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm_3',
                        result_anno='annotation_3',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_3", show=True)

# %%
sg.integrate_annotations(adata,
                        ggm_key='ggm_4',
                        result_anno='annotation_4',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )       
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation_4",
              na_color="black", show=True)
# %%









# %%

# %%
# 读取空转数据
adata = sc.read_h5ad("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/MOSTA/E16.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata.X.shape)

sc.pp.filter_cells(adata, min_genes=1000)
print(adata.X.shape)

sc.pp.filter_genes(adata, min_cells=10)
print(adata.X.shape)

# %%
ggm_1 = sg.create_ggm(adata,
                    project_name = "E16.5_E1S1", 
                    run_mode=2, 
                    double_precision=False,
                    use_chunking=True,
                    chunk_size=10000,
                    stop_threshold=0,
                    FDR_control=True,
                    FDR_threshold=0.01,
                    auto_adjust=True,
                    auto_find_modules=True,
                    )  
print(ggm_1.SigEdges)

# %%
# 重新读取数据
adata = sc.read_h5ad("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/MOSTA/E16.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(adata.X.shape)

# %%
#ggm_1.adjust_cutoff(pcor_threshold=0.02)
best_inf_1, _ = sg.find_best_inflation(ggm_1, min_inflation=1.1, phase=3, show_plot=True)
ggm_1.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=best_inf_1, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm_1.modules_summary.shape)

# %%
sg.annotate_with_ggm(adata, ggm_1,
                     ggm_key='ggm_1')
sg.integrate_annotations(adata,
                        ggm_key='ggm_1',
                        result_anno='annotation_1',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
sc.pl.spatial(adata, spot_size=1.2, title= "", frameon = False, color="annotation_1", show=True)
# %%
