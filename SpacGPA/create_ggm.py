import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
import time
import gc

from .calculate_gpu import calculate_pcors_pytorch
from .calculate_gpu import estimate_rounds
from .find_modules import run_mcl, run_louvain, run_mcl_original

class FDRResults_Pytorch:

    def __init__(self):
        """
        Instruction:

        Parameters:

        Returns:

        """   

        self.permutation_fraction = None 
        self.permutation_matrix = None
        self.permutation_gene_name = None
        self.permutation_coexpressed_cell_num = None
        self.permutation_pcor_all = None
        self.permutation_pcor_sampling_num = None
        self.permutation_rho_all = None
        self.permutation_SigEdges = None
        self.fdr = None


class ST_GGM_Pytorch:   
    def __init__(self, x, round_num=None, gene_name=None, sample_name=None, dataset_name='na',
                 cut_off_pcor=0.03, cut_off_coex_cell=10, selected_num=2000, 
                 use_chunking=True, chunk_size=1000, 
                 seed=98, run_mode=2, double_precision=False, stop_threshold=0,
                 FDR_control=False, FDR_threshold=0.0001, auto_adjust=False):
        """
        Instruction:        
        Please Normalize The Expression Matrix Before Running; 

        Parameters: 
        x : an expression matrix or AnnData object; spot in row, gene in column.
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
        self.gene_name = gene_name
        self.samples_num = None
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

        x, gene_name, sample_name = self._validate_and_extract_input(x, gene_name, sample_name)
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

           
        
    def _validate_and_extract_input(self, x, gene_name, sample_name):
        print("Please Normalize The Expression Matrix Before Running!")
        print("Loading Data.")
        if isinstance(x, AnnData):
            if x.X is None or x.var_names is None:
                raise ValueError("AnnData object must have X and var_names.")
            if not np.issubdtype(x.X.dtype, np.number):
                raise ValueError("Expression data must be numeric.")
            x_matrix = x.X
            if isinstance(x_matrix, np.matrix):
                x_matrix = sp.csr_matrix(np.asarray(x))
            elif isinstance(x, np.ndarray):
                x_matrix = sp.csr_matrix(x)
            gene_name = np.array(x.var_names)
            sample_name = np.array(x.obs_names)
        elif isinstance(x, np.matrix):
            x_matrix = sp.csr_matrix(np.asarray(x))
        elif isinstance(x, np.ndarray): 
            x_matrix = sp.csr_matrix(x)
        elif sp.issparse(x):
            x_matrix = x_matrix    
        else:
            raise ValueError("x must be a 2D cell x gene matrix(accepted formats include scipy sparse CSR matrix, numpy ndarray) or an AnnData object.")
        
        if x_matrix.ndim != 2:
            raise ValueError("x must be a 2D cell x gene matrix.(accepted formats include scipy sparse CSR matrix, numpy ndarray)")
        
        if gene_name is None or len(gene_name) != x_matrix.shape[1]:
            raise ValueError("Length of gene_name must match the number of columns in x.")
        
        if sample_name is None or len(sample_name) != x_matrix.shape[0]:
            raise ValueError("Length of sample_name must match the number of rows in x.")
        
        return x_matrix, gene_name, sample_name

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