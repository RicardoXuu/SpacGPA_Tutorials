
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
# 读取数据
adata = sc.read_visium("/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1/data/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2",
                       count_file="CytAssist_FreshFrozen_Mouse_Brain_Rep2_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.var_names = adata.var['gene_ids']
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.X.shape)


# %%
# 开发寻找最优 inflation 的函
import numpy as np
import pandas as pd
import networkx as nx
import torch
import sys
import gc

def run_mcl_qc(SigEdges, expansion=2, inflation=1.7, add_self_loops='mean', 
               max_iter=1000, tol=1e-6, pruning_threshold=1e-5, 
               run_mode=2):
    """
    基于输入的 SigEdges（包含 'GeneA', 'GeneB', 'Pcor' 列）执行 MCL 聚类，
    仅执行到候选节点归类（即“找最匹配的簇”）阶段，并返回聚类结果。
    
    参数：
        SigEdges: 包含 'GeneA', 'GeneB', 'Pcor' 列的 DataFrame。
        expansion: 扩展步骤的指数（默认2）。
        inflation: 膨胀步骤的指数（通常 > 1，默认1.7）。
        add_self_loops: 添加自环的方法，可选 'mean', 'min', 'max', 'dynamic', 'none'（默认'mean'）。
        max_iter: 最大迭代次数（默认1000）。
        tol: 收敛阈值（默认1e-6）。
        pruning_threshold: 小于该值的矩阵元素置零（默认1e-5）。
        run_mode: 0表示CPU，非0表示GPU（默认2，即使用GPU）。
        
    返回：
        clusters: 一个列表，每个元素是一个集合，表示一个簇中包含的节点索引（基于 SigEdges 中基因的顺序）。
    """
    try:
        # 1. 提取所有唯一基因，并构建基因与索引的映射
        genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
        gene_index = {gene: i for i, gene in enumerate(genes)}
        n = len(genes)
        
        # 构建对称加权邻接矩阵（权重取自 'Pcor' 列）
        M = np.zeros((n, n), dtype=np.float32)
        for _, row in SigEdges.iterrows():
            i = gene_index[row['GeneA']]
            j = gene_index[row['GeneB']]
            M[i, j] = row['Pcor']
            M[j, i] = row['Pcor']
        
        # 构建原始网络（用于候选节点归类），节点采用索引表示
        G_ori = nx.from_numpy_array(M)
        
        # 2. 添加自环，根据参数选取方法
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
            raise ValueError("Invalid value for 'add_self_loops'. Please choose from 'mean', 'min', 'max', 'dynamic', or 'none'.")
        
        # 3. 转换为 PyTorch 张量，并设置计算设备
        device = 'cpu' if run_mode == 0 else 'cuda'
        M = torch.tensor(M, device=device)
        
        # 定义列归一化函数（每列和为1）
        def normalize_columns(matrix):
            col_sums = matrix.sum(dim=0)
            return matrix / col_sums
        
        # 初始归一化
        M = normalize_columns(M)
        
        # 4. MCL 迭代：扩展、膨胀、归一化、剪枝，并检查收敛
        for iteration in range(max_iter):
            M_prev = M.clone()
            
            # 扩展：矩阵幂运算
            M = torch.matrix_power(M, expansion)
            
            # 膨胀：每个元素取 inflation 次方
            M = M.pow(inflation)
            
            # 归一化
            M = normalize_columns(M)
            
            # 剪枝：将小于阈值的元素置零
            M[M < pruning_threshold] = 0
            
            # 再次归一化
            M = normalize_columns(M)
            
            # 检查收敛：若最大变化小于 tol，则退出迭代
            diff = torch.max(torch.abs(M - M_prev))
            if diff < tol:
                break
        
        # 5. 将最终矩阵转换回 NumPy，并构建最终图（用于候选节点归类）
        M_np = M.cpu().numpy()
        G = nx.from_numpy_array(M_np)
        del M, M_np  # 释放内存
        
        # 6. 提取连通分量作为初步聚类簇
        clusters = list(nx.connected_components(G))
        
        # 将仅包含单个节点的簇作为候选节点，并剔除出簇集合
        size1_clusters = [cluster for cluster in clusters if len(cluster) == 1]
        candidate_nodes = set()
        for cluster in size1_clusters:
            for node in cluster:
                candidate_nodes.add(node)
        clusters = [cluster for cluster in clusters if len(cluster) > 1]
        
        # 7. 定义辅助函数：计算最短路径
        def find_shortest_path(net, source, target):
            try:
                return nx.shortest_path(net, source=source, target=target)
            except nx.NetworkXNoPath:
                return None
        
        # 8. 对每个簇（按簇大小降序）寻找簇内度为0的节点，利用候选节点补充
        clusters_sorted = sorted(clusters, key=lambda x: len(x), reverse=True)
        for cluster in clusters_sorted:
            subgraph = G.subgraph(cluster)
            degrees = dict(subgraph.degree())
            if not degrees:
                continue
            # 选择簇中度最大的节点作为 attractor
            attractor = max(degrees, key=degrees.get)
            # 在原始图中查找该簇内度为 0 的节点
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
        
        # 9. 对每个簇再次检查，将簇内度为 0 的节点移入候选节点集合
        for cluster in clusters:
            subgraph_ori = G_ori.subgraph(cluster)
            for node, deg in list(subgraph_ori.degree()):
                if deg == 0:
                    cluster.remove(node)
                    candidate_nodes.add(node)
        
        # 10. 构建节点到簇的映射，方便候选节点归类
        node_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = idx
        
        # 11. 对候选节点，根据邻居所在簇频次将其分配到最匹配的簇中
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
        
        # 返回聚类结果：每个簇为一个节点索引集合
        return clusters
    
    finally:
        gc.collect()
        if run_mode != 0:
            torch.cuda.empty_cache()

def build_original_graph(SigEdges, add_self_loops='mean'):
    """
    根据 SigEdges 构建原始网络图，用于 modularity 计算，节点以索引表示。
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

def find_best_inflation(SigEdges, max_inflation, min_inflation=1.1,
                        coarse_step=0.1, mid_step=0.05, fine_step=0.01,
                        expansion=2, add_self_loops='mean', max_iter=1000,
                        tol=1e-6, pruning_threshold=1e-5, run_mode=2):
    """
    根据 SigEdges 数据，利用 run_mcl_qc 进行 MCL 聚类，
    分三阶段搜索不同 inflation 参数下的聚类结果，并基于 NetworkX 的 modularity 评价聚类质量，
    返回最佳 inflation 参数及对应 modularity 值。
    
    参数：
      SigEdges: 包含 'GeneA', 'GeneB', 'Pcor' 列的 DataFrame。
      max_inflation: 搜索时使用的最大 inflation 参数值。
      min_inflation: 搜索时允许的最小 inflation 参数值（默认1.1）。
      coarse_step: 阶段1（粗略搜索）步长，默认0.1。
      mid_step: 阶段2（中步长搜索）步长，默认0.05。
      fine_step: 阶段3（精细搜索）步长，默认0.01。
      expansion, add_self_loops, max_iter, tol, pruning_threshold, run_mode:
          均传递给 run_mcl_qc，控制 MCL 聚类过程。
    
    返回：
      (best_inflation, best_modularity)
    """
    # 构建原始网络图，用于 modularity 计算
    G_ori = build_original_graph(SigEdges, add_self_loops=add_self_loops)
    
    best_inflation = None
    best_modularity = -np.inf
    
    def evaluate_inflation(inflation):
        # 使用 run_mcl_qc 进行聚类，返回聚类结果（各簇为节点索引集合）
        clusters = run_mcl_qc(SigEdges, expansion=expansion, inflation=inflation, 
                              add_self_loops=add_self_loops, max_iter=max_iter, 
                              tol=tol, pruning_threshold=pruning_threshold, 
                              run_mode=run_mode)
        # 确保聚类划分覆盖所有节点
        all_nodes = set(G_ori.nodes())
        clustered_nodes = set().union(*clusters)
        missing_nodes = all_nodes - clustered_nodes
        for node in missing_nodes:
            clusters.append({node})
        # 如果只有一个社区，modularity 定义为 0
        if len(clusters) <= 1:
            return 0.0
        Q = nx.algorithms.community.modularity(G_ori, clusters, weight='weight')
        return Q

    # 阶段1：粗略搜索
    print("【阶段1：粗略搜索】")
    inflation_val = max_inflation
    while inflation_val >= min_inflation:
        Q = evaluate_inflation(inflation_val)
        print(f"inflation = {inflation_val:.2f}, modularity = {Q:.4f}")
        if Q > best_modularity:
            best_modularity = Q
            best_inflation = inflation_val
        inflation_val = round(inflation_val - coarse_step, 4)
    
    # 阶段2：中步长搜索，在 best_inflation ± coarse_step 范围内
    print("\n【阶段2：中步长搜索】")
    lower_bound = max(best_inflation - coarse_step, min_inflation)
    upper_bound = min(best_inflation + coarse_step, max_inflation)
    inflation_val = upper_bound
    while inflation_val >= lower_bound:
        Q = evaluate_inflation(inflation_val)
        print(f"inflation = {inflation_val:.2f}, modularity = {Q:.4f}")
        if Q > best_modularity:
            best_modularity = Q
            best_inflation = inflation_val
        inflation_val = round(inflation_val - mid_step, 4)
    
    # 阶段3：精细搜索，在 best_inflation ± mid_step 范围内
    print("\n【阶段3：精细搜索】")
    lower_bound = max(best_inflation - mid_step, min_inflation)
    upper_bound = min(best_inflation + mid_step, max_inflation)
    inflation_val = upper_bound
    while inflation_val >= lower_bound:
        Q = evaluate_inflation(inflation_val)
        print(f"inflation = {inflation_val:.2f}, modularity = {Q:.4f}")
        if Q > best_modularity:
            best_modularity = Q
            best_inflation = inflation_val
        inflation_val = round(inflation_val - fine_step, 4)
    
    print(f"\n最佳 inflation 值: {best_inflation:.2f}, 对应 modularity: {best_modularity:.4f}")
    return best_inflation, best_modularity

# 示例调用：
# 假设 SigEdges 是一个包含 'GeneA', 'GeneB', 'Pcor' 列的 DataFrame，
# 例如：
# SigEdges = pd.read_csv("your_SigEdges.csv")
# best_inf, best_mod = find_best_inflation(SigEdges, max_inflation=2.5)

# %%
best_inf, best_mod = find_best_inflation(ggm_mulit_union.SigEdges, max_inflation=2.5)
# %%

# %%
ggm.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=1.38, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm.modules_summary)



# %%
ggm_mulit_intersection.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=1.37, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm_mulit_intersection.modules_summary)

# %%
ggm_mulit_union.find_modules(methods='mcl-hub', 
                        expansion=2, inflation=1.35, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=False, 
                        convert_to_symbols=True, species='mouse')
print(ggm_mulit_union.modules_summary)

# %%
ggm

# %%
ggm_mulit_union.modules_summary['size'][0:20]
# %%

# %%
# 计算模块表达值
start_time = time.time()
sg.calculate_module_expression(adata, 
                               ggm_obj=ggm,
                               ggm_key='ggm', 
                               top_genes=30,
                               weighted=True,
                               calculate_moran=False,
                               embedding_key='spatial',
                               k_neighbors=6,
                               add_go_anno=5)  
print(f"Time1: {time.time() - start_time:.5f} s")


# %%
# 计算GMM注释
start_time = time.time()
sg.calculate_gmm_annotations(adata, 
                            ggm_key='ggm',
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            random_state=42,
                            calculate_moran=True,
                            embedding_key='spatial',
                            k_neighbors=6
                            )
print(f"Time: {time.time() - start_time:.5f} s")
print(adata.uns['module_stats'])


# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M1_anno", show=True)

# %%
# 平滑注释
start_time = time.time()
sg.smooth_annotations(adata, 
                        ggm_key='ggm',
                        embedding_key='spatial',
                        k_neighbors=18,
                        min_annotated_neighbors=2
                        )
print(f"Time: {time.time() - start_time:.5f} s")    

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="M1_anno_smooth", show=True)

# %%
sg.classify_modules(adata, 
                 ggm_key='ggm',
                 #ref_anno='graph_cluster',
                 ref_cluster_method='leiden', 
                 #ref_cluster_method='none',  
                 ref_cluster_resolution=0.5, 
                 skew_threshold=2,
                 top1pct_threshold=2,
                 Moran_I_threshold=0.6,
                 min_dominant_cluster_fraction=0.3,
                 anno_overlap_threshold=0.4)


# %%
adata.uns['module_filtering']['type_tag'].value_counts()

# %%
start_time = time.time()
sg.integrate_annotations(adata,
                        ggm_key='ggm',
                        #modules_used=adata.uns['module_filtering'][adata.uns['module_filtering']['is_identity'] == True]['module_id'],
                        #modules_used=None,
                        #modules_used=adata.uns[mod_stats_key][adata.uns[mod_stats_key]['module_moran_I'] > 0.7]['module_id'],
                        #modules_preferred=adata.uns[mod_stats_key][adata.uns[mod_stats_key]['module_moran_I'] > 0.9]['module_id'],
                        #modules_used = adata.uns['module_info']['module_id'].unique()[0:20], 
                        #modules_used=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        #modules_used=['M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20'],
                        #modules_used={'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'},
                        #modules_excluded=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10'],
                        result_anno='annotation',
                        use_smooth=True,
                        embedding_key='spatial',
                        k_neighbors=24,
                        neighbor_similarity_ratio=0,
                        )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, title= "", frameon = False, color="annotation", show=True)
# %%



# %%
for module in adata.uns['module_stats']['module_id'].unique():
    print(module)
    print(adata.uns['module_info'][adata.uns['module_info']['module_id']==module]['module_moran_I'].unique())
    sc.pl.spatial(adata, color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"], 
                  color_map="Reds", alpha_img = 0.5, size = 1.6, frameon = False, show=True,
                  save=f"/CytAssist_FreshFrozen_Mouse_Brain_Rep2_{module}_ggm_anno.pdf")



# %%
# 逐个可视化各个模块的注释结果
anno_modules = adata.uns['module_stats']['module_id']
pdf_file = "figures/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_all_modules_Anno.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
image_files = []
for module in anno_modules:
    plt.figure()    
    sc.pl.spatial(adata, alpha_img = 0.5, size = 1.6, frameon = False, color_map="Reds", 
                  color=[f"{module}_exp",f"{module}_anno",f"{module}_anno_smooth"],show=False)
    show_png_file = f"figures/visium/CytAssist_FreshFrozen_Mouse_Brain_Rep2_{module}_Anno.png"
    plt.savefig(show_png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    image_files.append(show_png_file)

for image_file in image_files:
    img = Image.open(image_file)
    c.setPageSize((img.width, img.height))
    c.drawImage(image_file, 0, 0, width=img.width, height=img.height)
    c.showPage()
c.save()    

# %%