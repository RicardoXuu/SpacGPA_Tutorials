
# %%
# 尝试开发多切片联合分析的函数，使用 MOSTA E16.5_E1 数据集的5张切片数据进行测试
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
from SpacGPA import *

# %%
adata_1 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S1.MOSTA.h5ad')
adata_2 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S2.MOSTA.h5ad')
adata_3 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S3.MOSTA.h5ad')
adata_4 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S4.MOSTA.h5ad')
adata_5 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S5.MOSTA.h5ad')

# %%
# 批量过滤细胞和基因
adata_list = [adata_1, adata_2, adata_3, adata_4, adata_5]
for adata in adata_list:
    print(adata.X.shape)
    sc.pp.filter_cells(adata, min_genes=1000)
    sc.pp.filter_genes(adata, min_cells=10)
    print(adata.X.shape)
    print('------------------')

# %%
# 构建合并函数
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
import time
import gc

class Multi_ST_GGM:   
    def __init__(self, adata_list, round_num=None, sample_name=None, dataset_name='na',
                 cut_off_pcor=0.03, cut_off_coex_cell=10, selected_num=2000, 
                 use_chunking=True, chunk_size=1000, 
                 seed=98, run_mode=2, double_precision=False, stop_threshold=0,
                 FDR_control=False, FDR_threshold=0.0001, auto_adjust=False):
        """
        Instruction:        
        Please Normalize The Expression Matrix Before Running; 

        Parameters: 
        adata_list : list of AnnData objects.
        round_num : number of iterations.
        gene_name : an array of gene names. 
        sample_name : an array of spot ids.
        dataset_name : optional, default as 'na'.
        cut_off_pcor : optional, default as 0.03.
        cut_off_coex_cell : optional, default as 10.
        selected_num : optional, default as 2000.
        seed : optional, default as 98.
        run_mode : optional, default as 1.
                   0 - All computations on CPU.
                   1 - Preprocessing on CPU, partial correlation calculations on GPU.
                   2 - All computations on GPU.
        double_precision : optional, default as False.
        use_chunking : optional, default as True. Whether to enable chunk-based computations.
        chunk_size : optional, default as 1000. Controls memory usage for large datasets.   
        stop_threshold: threshold for stopping the loop if the sum of `valid_elements_count` over 100 iterations is less than this value.
        Returns:

        """

        self.matrix = None
        self.RoundNumber = round_num
        self.gene_num = None
        self.samples_num = None
        self.gene_name = None
        self.sample_name = sample_name
        self.data_name = dataset_name

        self.cut_off_pcor = cut_off_pcor
        self.cut_off_coex_cell = cut_off_coex_cell
        self.selected_num = selected_num
        self.seed_used = seed
        self.run_mode = run_mode  # Added run_mode
        self.double_precision = double_precision  # Added double_precision
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.stop_threshold = stop_threshold
        self.FDR_control = FDR_control
        self.FDR_threshold = FDR_threshold
        self.auto_adjust = auto_adjust

        self.coexpressed_cell_num = None
        self.pcor_all = None
        self.pcor_sampling_num = None
        self.rho_all = None
        self.SigEdges = None

        x, gene_name, sample_name = self._validate_and_extract_input(adata_list)
        self.matrix = sp.csr_matrix(x) 
        self.gene_name = gene_name
        self.sample_name = sample_name 
        
        self._initialize_variables(x)

        round_num_default = estimate_rounds(self.gene_num, self.selected_num, 100)
        if round_num is None:
            round_num = round_num_default
            self.RoundNumber = round_num_default 
        
        if run_mode == 0:
            print("\nRunning entirely on CPU.")
        elif run_mode == 1:
            print("\nRunning preprocessing on CPU and computing on GPU (if available).")
        elif run_mode == 2:
            print("\nRunning entirely on GPU (if available).")
        else:
            raise ValueError("Invalid run_mode. Use 0 for CPU, 1 for hybrid, or 2 for GPU.")

        # Determine dtype based on parameters
        if double_precision:
            print("Using double precision (float64).")
        else:
            print("Using single precision (float32).")

        # Adjust chunk_size if not using chunking
        if  use_chunking:
            print(f"Using chunk size of {chunk_size} for efficient computation")  

        self.RoundNumber, self.coexpressed_cell_num, self.pcor_all, self.pcor_sampling_num, self.rho_all, self.SigEdges = calculate_pcors_pytorch(
            x, 
            round_num, 
            self.gene_name, 
            self.data_name, 
            cut_off_pcor=cut_off_pcor, 
            cut_off_coex_cell=cut_off_coex_cell, 
            selected_num=selected_num, 
            seed=seed,
            run_mode=run_mode, 
            double_precision=double_precision, 
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            stop_threshold=stop_threshold)
        
        if FDR_control:
            print("\nPerforming FDR control...")
            self.fdr_control()
            print(f"Current Pcor: {self.cut_off_pcor:.3f}")
            fdr_df = self.fdr.fdr
            fdr_filtered = fdr_df.loc[fdr_df['FDR'] <= self.FDR_threshold]
            if fdr_filtered.shape[0] == 0:
                print(f"No Pcors found with FDR <= {self.FDR_threshold}.") 
            else: 
                min_pcor_row = fdr_filtered.iloc[0]
                min_pcor = min_pcor_row['Pcor']
                print(f"Minimum Pcor with FDR <= {self.FDR_threshold}: {min_pcor:.3f}")
                if auto_adjust:
                    print("Adjusting cutoff based on FDR results...")
                    self.adjust_cutoff(pcor_threshold=round(min_pcor, 3),
                                        coex_cell_threshold=self.cut_off_coex_cell) 

           
        
    def _validate_and_extract_input(self, adata_list):
        print("Please Normalize The Expression Matrices Before Running!")
        print("Loading Data from AnnData list.")
        
        if not isinstance(adata_list, list) or len(adata_list) == 0:
            raise ValueError("adata_list must be a non-empty list of AnnData objects.")

        # 检查每个对象
        for adata in adata_list:
            if not hasattr(adata, "X") or not hasattr(adata, "var_names") or adata.X is None or adata.var_names is None:
                raise ValueError("Each AnnData object must have X and var_names.")
            if not np.issubdtype(adata.X.dtype, np.number):
                raise ValueError("Expression data must be numeric.")

        # 获取所有adata的公共基因，保持第一个adata的基因顺序
        common_genes = list(adata_list[0].var_names)
        for adata in adata_list[1:]:
            common_genes = [gene for gene in common_genes if gene in adata.var_names]
            
        if len(common_genes) == 0:
            raise ValueError("No overlapping genes found among the provided AnnData objects.")

        print(f"Found {len(common_genes)} common genes across datasets.")

        # 对每个adata仅保留公共基因，并组合表达矩阵及样本名称
        matrix_list = []
        sample_names_list = []
        for i, adata in enumerate(adata_list):
            subset_adata = adata[:, common_genes]
            matrix_list.append(sp.csr_matrix(subset_adata.X))
            # 为样本名添加数据集前缀，避免重复（例如 "Dataset1_"）
            prefix = f"Dataset{i+1}_"
            prefixed_names = np.array([prefix + str(name) for name in subset_adata.obs_names])
            sample_names_list.append(prefixed_names)
            
        combined_matrix = sp.vstack(matrix_list)
        combined_sample_names = np.concatenate(sample_names_list)
            
        return combined_matrix, np.array(common_genes), combined_sample_names


    def _initialize_variables(self, x):
        self.samples_num = x.shape[0]
        self.gene_num = x.shape[1]

    def adjust_cutoff(self, pcor_threshold=0.03, coex_cell_threshold=10):
        """
        Instruction:

        Parameters:

        Returns:

        """   
    
        cut_off_pcor = pcor_threshold
        cut_off_coex_cell = coex_cell_threshold        
        cellnum = np.diagonal(self.coexpressed_cell_num)
        
        self.cut_off_pcor = cut_off_pcor
        self.cut_off_coex_cell = cut_off_coex_cell

        idx = np.where((self.pcor_all >= cut_off_pcor)
                    & (self.pcor_all < 1)
                    & (self.coexpressed_cell_num >= cut_off_coex_cell))
        e1 = list(self.gene_name[idx[0]])
        e1n = list(cellnum[idx[0]].astype(int))
        e2 = list(self.gene_name[idx[1]])
        e2n = list(cellnum[idx[1]].astype(int))
        e3 = list(self.pcor_all[idx])
        e4 = list(self.pcor_sampling_num[idx].astype(int))
        e5 = list(self.rho_all[idx])
        e6 = list(self.coexpressed_cell_num[idx].astype(int))
        e7 = [self.data_name] * len(e1)
        self.SigEdges = pd.DataFrame({'GeneA': e1, 'GeneB': e2, 'Pcor': e3, 'SamplingTime': e4,
                                        'r': e5, 'Cell_num_A': e1n, 'Cell_num_B': e2n,
                                        'Cell_num_coexpressed': e6, 'Dataset': e7})

    def fdr_control(self, permutation_fraction=1.0):
        """
        Perform FDR control by permuting gene columns and calculating the necessary statistics.

        Parameters:
        - permutation_fraction: Fraction of genes to permute.

        Returns:
        - fdr: The FDR results object.
        """
        run_mode = self.run_mode
        device = 'cuda' if run_mode != 0 else 'cpu'

        # Create result storage
        print("Randomly redistribute the expression distribution of input genes...")
        fdr = FDRResults_Pytorch()
        fdr.permutation_fraction = permutation_fraction
        new_gene_name = self.gene_name.copy()

        # Randomly select columns to permute
        gene_num = self.gene_num
        samples_num = self.samples_num
        permutation_num = round(gene_num * permutation_fraction)
        
        np.random.seed(1)
        perm_cols = np.random.choice(gene_num, size=permutation_num, replace=False)
        perm_cols_set = set(perm_cols)

        # Convert to CSC format for modification
        permutation_x = self.matrix.tocsc()
        indices = permutation_x.indices
        data = permutation_x.data
        indptr = permutation_x.indptr

        # Process each column
        for c in range(gene_num):
            c_start = indptr[c]
            c_end = indptr[c + 1]
            row_idx = indices[c_start:c_end]
            col_vals = data[c_start:c_end]

            # If the column needs to be permuted
            if c in perm_cols_set:
                # Convert to dense format, keeping non-zero elements
                dense_col = np.zeros(samples_num, dtype=np.float32)
                dense_col[row_idx] = col_vals

                # Shuffle using PyTorch, ensuring it's on CPU first
                dense_col = torch.from_numpy(dense_col).to(device)
                shuffle_idx = torch.randperm(samples_num, device=device)
                dense_col = dense_col[shuffle_idx]

                # Update the sparse matrix data and indices
                new_row_idx = np.nonzero(dense_col.cpu()).squeeze()  # Get non-zero row indices
                new_col_vals = dense_col.cpu()[new_row_idx]

                # Update gene names
                new_gene_name[c] = 'P_' + new_gene_name[c]

                # Restore shuffled data back to the sparse matrix
                permutation_x.data[c_start:c_end] = new_col_vals
                permutation_x.indices[c_start:c_end] = new_row_idx

        # Convert back to CSR format
        permutation_x_csr = permutation_x.tocsr()

        # Store results
        fdr.permutation_matrix = permutation_x_csr
        fdr.permutation_gene_name = new_gene_name

        # Calculate partial correlations and other statistics
        print("\nCalculate correlation between genes after redistribution...")
        round_num = self.RoundNumber
        cut_off_pcor = self.cut_off_pcor
        cut_off_coex_cell = self.cut_off_coex_cell
        selected_num = self.selected_num

        seed = int(time.time() * 1000) % 4294967296
        
        _, fdr.permutation_coexpressed_cell_num, fdr.permutation_pcor_all, fdr.permutation_pcor_sampling_num, fdr.permutation_rho_all, fdr.permutation_SigEdges = calculate_pcors_pytorch(
            permutation_x_csr, 
            round_num=round_num, 
            gene_name=new_gene_name, 
            data_name=self.data_name, 
            cut_off_pcor=cut_off_pcor, 
            cut_off_coex_cell=cut_off_coex_cell, 
            selected_num=selected_num, 
            seed=seed, 
            run_mode=self.run_mode,
            double_precision=self.double_precision,
            use_chunking=self.use_chunking,
            chunk_size=self.chunk_size,
            stop_threshold=0
        )

        del permutation_x_csr, new_gene_name
        gc.collect()

        # Calculate the statistics (num_ori_sig, num_permutated_sig, pct1, pct2)
        fdr_stat = torch.zeros((91, 5), dtype=torch.float32, device=device)
        idx_np = torch.ones((self.gene_num, self.gene_num), dtype=torch.bool, device=device)
        idx_p = torch.ones_like(idx_np, dtype=torch.bool, device=device)
        
        # Precompute permutation matrix mask
        idx_np[:, perm_cols] = False
        idx_np[perm_cols, :] = False
        idx_p = ~idx_np
        idx_p = idx_p.cpu().numpy()
        del idx_np
        gc.collect()
        torch.cuda.empty_cache()

        # Precompute number of permutations and the remaining edges
        num_np = (self.gene_num - permutation_num) * (self.gene_num - permutation_num - 1) / 2
        num_permutated = self.gene_num * (self.gene_num - 1) / 2 - num_np

        # Convert pcor_all and coexpressed_cell_num to tensors once before the loop
        print("\nSummarizing the FDR Statistics...")
        pcor_all_tensor = torch.tensor(self.pcor_all, device=device, dtype=torch.float32)
        coexpressed_cell_num_tensor = torch.tensor(self.coexpressed_cell_num, device=device, dtype=torch.int32)

        # First loop: Calculate num_ori_sig and store
        for i in range(10, 101):
            # Calculate idx_ori for original condition
            idx_ori = (pcor_all_tensor >= i / 1000) & (coexpressed_cell_num_tensor >= self.cut_off_coex_cell)
            idx_ori = idx_ori.cpu().numpy() 
            # Calculate num_ori_sig
            num_ori_sig = np.sum(idx_ori).item()            
            # Store the result in fdr_stat for later use
            fdr_stat[i - 10, 0] = i / 1000
            fdr_stat[i - 10, 1] = num_ori_sig

            del idx_ori, num_ori_sig
            gc.collect()
            torch.cuda.empty_cache()

        del pcor_all_tensor, coexpressed_cell_num_tensor
        gc.collect()
        torch.cuda.empty_cache()

        # Convert permutation_pcor_all and permutation_coexpressed_cell_num to tensors before the second loop
        perm_pcor_all_tensor = torch.tensor(fdr.permutation_pcor_all, device=device, dtype=torch.float32)
        perm_coexpressed_cell_num_tensor = torch.tensor(fdr.permutation_coexpressed_cell_num, device=device, dtype=torch.int32)

        # Second loop: Calculate num_permutated_sig and store
        for i in range(10, 101):
            # Calculate idx_permutated for permuted condition
            idx_permutated = (perm_pcor_all_tensor >= i / 1000) & (perm_coexpressed_cell_num_tensor >= self.cut_off_coex_cell)
            idx_permutated = idx_permutated.cpu().numpy()
            # Calculate num_permutated_sig
            num_permutated_sig = np.sum(idx_permutated & idx_p).item()
            # Store the result in fdr_stat for later use
            fdr_stat[i - 10, 3] = num_permutated_sig

            del idx_permutated, num_permutated_sig
            gc.collect()
            torch.cuda.empty_cache()

        del perm_pcor_all_tensor, perm_coexpressed_cell_num_tensor
        gc.collect()
        torch.cuda.empty_cache()

        # Third loop: Calculate pct1 and pct2
        for i in range(10, 101):
            # Calculate pct1 and pct2 using the stored values
            num_ori_sig = fdr_stat[i - 10, 1]
            num_permutated_sig = fdr_stat[i - 10, 3]
            pct1 = num_permutated_sig / num_permutated
            pct2 = self.gene_num * (self.gene_num - 1) / 2 * pct1 / num_ori_sig
            if pct2 > 1:
                pct2 = 1

            # Store pct2 and pct1 in fdr_stat
            fdr_stat[i - 10, 2] = pct2
            fdr_stat[i - 10, 4] = pct1

        # Convert to DataFrame for easier inspection
        fdr_df = pd.DataFrame(fdr_stat.cpu().numpy(), columns=['Pcor', 'SigEdgeNum', 'FDR', 'SigPermutatedEdgeNum', 'SigPermutatedEdgeProportion'])
        fdr_df['SigEdgeNum'] = fdr_df['SigEdgeNum'].astype(int)
        fdr_df['SigPermutatedEdgeNum'] = fdr_df['SigPermutatedEdgeNum'].astype(int)
        fdr.fdr = fdr_df
        self.fdr = fdr

        print("FDR control completed.")

    def find_modules(self, methods='mcl', 
                expansion=2, inflation=1.7, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                resolution=1.0,
                scheme=7, threads=1,
                min_module_size=10, topology_filtering=True,
                convert_to_symbols=False, species='human'):
        """
        Find modules using the specified method.

        Parameters:
        - methods: The method to use for module detection. Options are 'mcl', 'louvain' or 'mcl_original'.
        - mcl parameters:
            - expansion: The mcl expansion parameter.
            - inflation: The mcl inflation parameter.
            - max_iter: The maximum number of iterations for mcl.
            - tol: The convergence threshold for mcl.
            - pruning_threshold: The mcl pruning threshold.
        - louvain parameters:
            - resolution: The resolution parameter for Louvain.
        - mcl_original parameters:
            - inflation: The mcl inflation parameter.
            - scheme: The mcl scheme parameter.
            - threads: The number of threads to use for mcl.
        - min_module_size: The minimum number of genes required for a module to be retained.
        - topology_filtering: Whether to apply topology filtering for each module.
        - convert_to_symbols: Whether to convert gene IDs to gene symbols.
        - species: The species for gene ID conversion.
        """
        if methods == 'mcl':
            print("\nFind modules using MCL...")
            print(f"Current Pcor: {self.cut_off_pcor}")
            print(f"Total significantly co-expressed gene pairs: {len(self.SigEdges)}")
            inflation = inflation
            expansion = expansion
            max_iter = max_iter
            tol = tol
            pruning_threshold = pruning_threshold
            module_df = run_mcl(self.SigEdges, 
                                inflation=inflation, expansion=expansion, max_iter=max_iter, tol=tol, pruning_threshold=pruning_threshold,
                                min_module_size=min_module_size, topology_filtering=topology_filtering,
                                convert_to_symbols=convert_to_symbols, species=species)
            self.modules = module_df.copy()
        elif methods == 'louvain':
            print("\nFind modules using Louvain...")
            print(f"Current Pcor: {self.cut_off_pcor}")
            print(f"Total significantly co-expressed gene pairs: {len(self.SigEdges)}")
            resolution = resolution
            module_df = run_louvain(self.SigEdges, 
                                    resolution=resolution,
                                    min_module_size=min_module_size, topology_filtering=topology_filtering,
                                    convert_to_symbols=convert_to_symbols, species=species)
            self.modules = module_df.copy()                          
        elif methods == 'mcl_original':
            print("\nFind modules using MCL original...")
            print(f"Current Pcor: {self.cut_off_pcor}")
            print(f"Total significantly co-expressed gene pairs: {len(self.SigEdges)}")
            module_df = run_mcl_original(self.SigEdges,
                                        inflation=inflation, scheme=scheme, threads=threads,
                                        min_module_size=min_module_size, topology_filtering=topology_filtering,
                                        convert_to_symbols=convert_to_symbols, species=species) 
            self.modules = module_df.copy()   
        else:
            raise ValueError("Invalid method. Use 'mcl', 'louvain', or 'mcl_original'.")
        
        if convert_to_symbols:
            modules_symbol = module_df.copy()
            self.modules_symbol = modules_symbol[['module_id','symbol','degree','rank']].rename(columns={'symbol':'gene'})
            grouped = module_df.groupby('module_id')
            self.modules_summary = grouped.agg(
                size =('gene','size'),
                num_genes_degree_ge_2=('degree', lambda x: (x >= 2).sum()),
                all_symbols=('symbol',', '.join),
                all_genes=('gene', ', '.join)
            ).reset_index()
        else:
            grouped = module_df.groupby('module_id')
            self.modules_summary = grouped.agg(
                size =('gene','size'),
                num_genes_degree_ge_2=('degree', lambda x: (x >= 2).sum()),
                all_genes=('gene', ', '.join)
            ).reset_index()


# %%
# 测试
start_time = time.time()
ggm_combined = Multi_ST_GGM(adata_list,
                            round_num=20000, 
                            dataset_name = "MOSTA_E16.5", 
                            selected_num=2000, 
                            cut_off_pcor=0.03,
                            run_mode=2, 
                            double_precision=False,
                            use_chunking=True,
                            chunk_size=5000,
                            stop_threshold=0,
                            FDR_control=False,
                            FDR_threshold=0.05,
                            auto_adjust=False
                            )
print(f"Time: {time.time() - start_time:.5f} s")

# %%
start_time = time.time()
ggm_combined.fdr_control()
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm_combined.fdr.fdr[ggm_combined.fdr.fdr['FDR'] <= 0.05])

# %%
cut_pcor = ggm_combined.fdr.fdr[ggm_combined.fdr.fdr['FDR'] <= 0.05]['Pcor'].min()
if cut_pcor < 0.03:
    cut_pcor = 0.03
print("Adjust cutoff pcor:", cut_pcor)
ggm_combined.adjust_cutoff(pcor_threshold=cut_pcor)

# %%
ggm_combined.find_modules(methods='mcl',
                        expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                        min_module_size=10, topology_filtering=True, 
                        convert_to_symbols=False, species='mouse')
print(ggm_combined.modules_summary)

# %%
start_time = time.time()
go_enrichment_analysis(ggm_combined, 
                       padjust_method='BH',
                       pvalue_cutoff=0.05,
                       species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm_combined.go_enrichment)

# %%
start_time = time.time()
mp_enrichment_analysis(ggm_combined,
                       padjust_method='BH',
                       pvalue_cutoff=0.05,
                       species='mouse')
print(f"Time: {time.time() - start_time:.5f} s")
print(ggm_combined.mp_enrichment)


# %%
# 逐个计算各张切片的模块
# %%
ggm_1 = ST_GGM_Pytorch(adata_1,
                        round_num=20000, 
                        dataset_name = "E1S1", 
                        selected_num=2000, 
                        cut_off_pcor=0.03,
                        run_mode=2, 
                        double_precision=False,
                        use_chunking=True,
                        chunk_size=5000,
                        stop_threshold=0,
                        FDR_control=False,
                        FDR_threshold=0.05,
                        auto_adjust=False
                        )
# %%
ggm_1.find_modules(methods='mcl',
                    expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                    min_module_size=10, topology_filtering=True, 
                    convert_to_symbols=False, species='mouse')


# %%
ggm_2 = ST_GGM_Pytorch(adata_2,
                        round_num=20000, 
                        dataset_name = "E1S2", 
                        selected_num=2000, 
                        cut_off_pcor=0.03,
                        run_mode=2, 
                        double_precision=False,
                        use_chunking=True,
                        chunk_size=5000,
                        stop_threshold=0,
                        FDR_control=False,
                        FDR_threshold=0.05,
                        auto_adjust=False
                        )
# %%
ggm_2.find_modules(methods='mcl',
                    expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                    min_module_size=10, topology_filtering=True, 
                    convert_to_symbols=False, species='mouse')

# %%    
ggm_3 = ST_GGM_Pytorch(adata_3,
                        round_num=20000, 
                        dataset_name = "E1S3", 
                        selected_num=2000, 
                        cut_off_pcor=0.03,
                        run_mode=2, 
                        double_precision=False,
                        use_chunking=True,
                        chunk_size=5000,
                        stop_threshold=0,
                        FDR_control=False,
                        FDR_threshold=0.05,
                        auto_adjust=False
                        )  
 # %%
ggm_3.find_modules(methods='mcl',
                    expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                    min_module_size=10, topology_filtering=True, 
                    convert_to_symbols=False, species='mouse')

# %%   
ggm_4 = ST_GGM_Pytorch(adata_4,
                        round_num=20000, 
                        dataset_name = "E1S4", 
                        selected_num=2000, 
                        cut_off_pcor=0.03,
                        run_mode=2, 
                        double_precision=False,
                        use_chunking=True,
                        chunk_size=5000,
                        stop_threshold=0,
                        FDR_control=False,
                        FDR_threshold=0.05,
                        auto_adjust=False
                        )   
# %%
ggm_4.find_modules(methods='mcl',
                    expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                    min_module_size=10, topology_filtering=True, 
                    convert_to_symbols=False, species='mouse')

# %%
ggm_5 = ST_GGM_Pytorch(adata_5,
                        round_num=20000, 
                        dataset_name = "E1S5", 
                        selected_num=2000, 
                        cut_off_pcor=0.03,
                        run_mode=2, 
                        double_precision=False,
                        use_chunking=True,
                        chunk_size=5000,
                        stop_threshold=0,
                        FDR_control=False,
                        FDR_threshold=0.05,
                        auto_adjust=False
                        )
# %%
ggm_5.find_modules(methods='mcl',
                    expansion=2, inflation=2, max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                    min_module_size=10, topology_filtering=True, 
                    convert_to_symbols=False, species='mouse')


# %%
# 使用合并数据集的模块结果注释各张切片
del adata_1, adata_2, adata_3, adata_4, adata_5
gc.collect()

# %%
# 重新读取数据，补全细胞
adata_1 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S1.MOSTA.h5ad')
adata_2 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S2.MOSTA.h5ad')
adata_3 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S3.MOSTA.h5ad')
adata_4 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S4.MOSTA.h5ad')
adata_5 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S5.MOSTA.h5ad')



# %%
for adata in [adata_1, adata_2, adata_3, adata_4, adata_5]:
    calculate_module_expression(adata, ggm_combined, 
                                top_genes=30,
                                weighted=True)
    calculate_gmm_annotations(adata, 
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            random_state=42)
    smooth_annotations(adata,
                    embedding_key='spatial',
                    k_neighbors=24,
                    min_annotated_neighbors=2
                    )
    integrate_annotations(adata,
                  result_anno='annotation_combined',
                  embedding_key='spatial',
                  k_neighbors=24,
                  use_smooth=True,
                  neighbor_majority_frac=0.90
                  )
    
# %%
# 可视化注释结果
sc.pl.spatial(adata_1, spot_size=1.2, title= "E1S1 Combined", frameon = False, color="annotation_combined", 
                save="/E16_5_E1S1_annotation_combined.pdf",show=True)
sc.pl.spatial(adata_2, spot_size=1.2, title= "E1S2 Combined", frameon = False, color="annotation_combined",
                save="/E16_5_E1S2_annotation_combined.pdf",show=True)
sc.pl.spatial(adata_3, spot_size=1.2, title= "E1S3 Combined", frameon = False, color="annotation_combined",
                save="/E16_5_E1S3_annotation_combined.pdf",show=True)
sc.pl.spatial(adata_4, spot_size=1.2, title= "E1S4 Combined", frameon = False, color="annotation_combined",
                save="/E16_5_E1S4_annotation_combined.pdf",show=True)
sc.pl.spatial(adata_5, spot_size=1.2, title= "E1S5 Combined", frameon = False, color="annotation_combined",
                save="/E16_5_E1S5_annotation_combined.pdf",show=True)
     
# %%
# 保存数据
adata_1.write('data/E16.5_E1S1.MOSTA_Combined_Anno.h5ad')
adata_2.write('data/E16.5_E1S2.MOSTA_Combined_Anno.h5ad')
adata_3.write('data/E16.5_E1S3.MOSTA_Combined_Anno.h5ad')
adata_4.write('data/E16.5_E1S4.MOSTA_Combined_Anno.h5ad')
adata_5.write('data/E16.5_E1S5.MOSTA_Combined_Anno.h5ad')

# %%
# 使用各自自己的模块结果注释各张切片
del adata_1, adata_2, adata_3, adata_4, adata_5
gc.collect()
# %%
# 重新读取数据，补全细胞
adata_1 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S1.MOSTA.h5ad')
adata_2 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S2.MOSTA.h5ad')
adata_3 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S3.MOSTA.h5ad')
adata_4 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S4.MOSTA.h5ad')
adata_5 = sc.read_h5ad('/dta/ypxu/ST_GGM/Raw_Datasets/MOSTA/E16.5_E1S5.MOSTA.h5ad')

# %%
for adata,ggm in zip([adata_1, adata_2, adata_3, adata_4, adata_5], [ggm_1, ggm_2, ggm_3, ggm_4, ggm_5]):
    calculate_module_expression(adata, ggm, 
                                top_genes=30,
                                weighted=True)
    calculate_gmm_annotations(adata, 
                            max_iter=200,
                            prob_threshold=0.99,
                            min_samples=10,
                            n_components=3,
                            enable_fallback=True,
                            random_state=42)
    smooth_annotations(adata,
                    embedding_key='spatial',
                    k_neighbors=24,
                    min_annotated_neighbors=2
                    )
    integrate_annotations(adata,
                  result_anno='annotation_self',
                  embedding_key='spatial',
                  k_neighbors=24,
                  use_smooth=True,
                  neighbor_majority_frac=0.90
                  )
# %%
# 可视化注释结果
sc.pl.spatial(adata_1, spot_size=1.2, title= "E1S1 Self", frameon = False, color="annotation_self",
                save="/E16_5_E1S1_annotation_self.pdf",show=True)
sc.pl.spatial(adata_2, spot_size=1.2, title= "E1S2 Self", frameon = False, color="annotation_self",
                save="/E16_5_E1S2_annotation_self.pdf",show=True)
sc.pl.spatial(adata_3, spot_size=1.2, title= "E1S3 Self", frameon = False, color="annotation_self",
                save="/E16_5_E1S3_annotation_self.pdf",show=True)
sc.pl.spatial(adata_4, spot_size=1.2, title= "E1S4 Self", frameon = False, color="annotation_self",
                save="/E16_5_E1S4_annotation_self.pdf",show=True)
sc.pl.spatial(adata_5, spot_size=1.2, title= "E1S5 Self", frameon = False, color="annotation_self",
                save="/E16_5_E1S5_annotation_self.pdf",show=True)


# %%
# 保存数据
adata_1.write('data/E16.5_E1S1.MOSTA_Self_Anno.h5ad')
adata_2.write('data/E16.5_E1S2.MOSTA_Self_Anno.h5ad')
adata_3.write('data/E16.5_E1S3.MOSTA_Self_Anno.h5ad')
adata_4.write('data/E16.5_E1S4.MOSTA_Self_Anno.h5ad')
adata_5.write('data/E16.5_E1S5.MOSTA_Self_Anno.h5ad')
del adata_1, adata_2, adata_3, adata_4, adata_5

# %%


