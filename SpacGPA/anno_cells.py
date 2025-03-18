
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import itertools
import warnings


# calculate_module_expression 
def calculate_module_expression(adata, ggm_obj, top_genes=30, weighted=True):
    """
    Calculate and store module expression in adata based on input modules.

    Parameters:
        adata: AnnData object containing gene expression data.
        ggm_obj: GGM object containing module information or a DataFrame with columns 'module_id', 'gene', 'degree', and 'rank'.
        top_genes: Number of top genes to use for module expression calculation.
        weighted: Whether to calculate weighted average expression based on gene degree.

    """
    if isinstance(ggm_obj, pd.DataFrame):
        module_df = ggm_obj
    else:
        if ggm_obj.modules is None:
            raise ValueError("No modules found in the GGM object. Please run `find_modules` first.")
        module_df = ggm_obj.modules
    
    print(f"\nCalculating module expression using top {top_genes} genes...")
    
    # 1. Filter out genes with rank larger than top_genes
    module_df = module_df[module_df['rank'] <= top_genes].copy()

    # 2. Calculate the weights of each gene in the input module
    if weighted:
        print("Calculating gene weights based on degree...")
        module_df['weight']  = module_df.groupby('module_id')['degree'].transform( lambda x: x / x.sum() )
        #module_df['weight']  = module_df.groupby('module_id')['degree'].transform( lambda x: x / x.sum() * x.size )
    else:
        print("Using unweighted gene expression...")
        module_df['weight'] = module_df.groupby('module_id')['degree'].transform( lambda x: 1 / x.size ) 

    # 3. Filter the input modules to keep only genes that exist in adata
    genes_in_adata = adata.var_names
    module_df = module_df[module_df['gene'].isin(genes_in_adata)]
    module_df.index = range(module_df.shape[0])
    
    # Check if module_info already exists in adata.uns
    if 'module_info' not in adata.uns:
        print("Storing module information in adata.uns['module_info']...")
        adata.uns['module_info'] = module_df
    else:
        print("NOTE: module_info already exists in adata.uns, overwriting...")
        adata.uns['module_info'] = module_df
    
    # 4. Construct a transformation matrix
    # Create a mapping from gene to index in adata
    gene_to_index = {gene: i for i, gene in enumerate(adata.var_names)}
    # Create a mapping from module ID to index in the transformation matrix
    module_ids = module_df['module_id'].unique()
    module_to_index = {module: i for i, module in enumerate(module_ids)}

    # Initialize the transformation matrix
    n_genes = len(adata.var_names)
    n_modules = len(module_ids)
    transformation_matrix = sp.lil_matrix((n_genes, n_modules), dtype=np.float32)

    # Fill the transformation matrix with weights
    for _, row in module_df.iterrows():
        gene_idx = gene_to_index[row['gene']]
        module_idx = module_to_index[row['module_id']]
        transformation_matrix[gene_idx, module_idx] = row['weight']

    # Convert to CSR format for efficient multiplication
    transformation_matrix = transformation_matrix.tocsr()

    # 5. Multiply adata by the transformation matrix to obtain the weighted-average-expression matrix
    weighted_expression = adata.X.dot(transformation_matrix)

    # Convert to dense array if necessary
    if sp.issparse(weighted_expression):
        weighted_expression = weighted_expression.toarray()

    # 6. Store the weighted-average-expression in both obsm and obs of the original adata object
    # Store in obsm (as a single matrix)
    adata.obsm['module_expression'] = weighted_expression
    
    # Scale the weighted expression
    scaler = StandardScaler()
    adata.obsm['module_expression_scaled'] = scaler.fit_transform(weighted_expression)
    
    # Create a DataFrame for the weighted expression
    weighted_expression_df = pd.DataFrame(
        weighted_expression,
        index=adata.obs_names,
        columns=[f'{module}_exp' for module in module_ids]
    )
    
    # Store in obs (one column per module)
    obs_df = pd.concat([adata.obs, weighted_expression_df], axis=1)
    obs_df = obs_df.loc[:, ~obs_df.columns.duplicated(keep='last')]
    adata.obs = obs_df.copy()
    
    print(f"\nTotal {n_modules} modules' average expression calculated and stored in adata.obs and adata.obsm")


# calculate_gmm_annotation
def calculate_gmm_annotations(adata, 
                              modules_list = None,
                              max_iter=200,
                              prob_threshold=0.99,
                              min_samples=10,
                              n_components=3,
                              enable_fallback=True,
                              random_state=42
                              ):
    """
    Gaussian Mixture Model annotation (with threshold check for 3 components).
    
    Parameters:
      adata: AnnData object.
      modules_list: List of module IDs.
      max_iter: Maximum iterations.
      prob_threshold: Probability threshold for high expression.
      min_samples: Minimum valid sample count.
      n_components: Initial number of GMM components.
      enable_fallback: Whether to enable fallback to fewer components.
      random_state: Random seed.
      
    Returns:
      adata: Updated AnnData object with:
        - obs: Integrated annotation data (columns prefixed with "Anno_").
        - uns['module_stats']: Raw statistics records.
    """
    # Input validation
    if "module_expression" not in adata.obsm:
        raise ValueError("module_expression not found in adata.obsm")
    if "module_info" not in adata.uns:
        raise ValueError("module_info not found in adata.uns")
    
    module_expr_matrix = adata.obsm["module_expression"]
    module_expr_matrix = pd.DataFrame(module_expr_matrix, index=adata.obs.index)
    if len(adata.uns['module_info']['module_id'].unique()) != module_expr_matrix.shape[1]:
        raise ValueError("module_info and module_expression dimensions do not match")
    else:
        module_expr_matrix.columns = adata.uns['module_info']['module_id'].unique()

    # Check module list
    if modules_list is None:
        modules_list = adata.uns['module_info']['module_id'].unique()

    valid_modules = [mid for mid in modules_list if mid in module_expr_matrix.columns]
    
    if not valid_modules:
        raise ValueError("Ensure that the input module IDs exist in adata.uns['module_info']")
    
    existing_columns = [f"{mid}_anno" for mid in modules_list if f"{mid}_anno" in adata.obs]
    if existing_columns:
        print(f"Removing existing annotation columns: {existing_columns}")
        adata.obs.drop(columns=existing_columns, inplace=True)

    # Initialize annotation matrix
    anno_cols = [f"{mid}" for mid in modules_list]
    annotations = pd.DataFrame(
        np.zeros((adata.obs.shape[0], len(anno_cols)), dtype=int),
        index=adata.obs.index,
        columns=anno_cols
    )
    stats_records = []
    
    # Process each module
    for module_id in valid_modules:
        stats = {
            'module_id': module_id,
            'status': 'success',
            'n_components': n_components,
            'final_components': n_components,
            'threshold': np.nan,
            'anno_one': 0,
            'anno_zero': 0,
            'components': [],
            'error_info': 'None'
        }
        
        try:
            expr_values = module_expr_matrix[module_id].values
            non_zero_mask = expr_values != 0
            non_zero_expr = expr_values[non_zero_mask]
            
            # Basic checks
            if len(non_zero_expr) == 0:
                raise ValueError("all_zero_expression")
            if len(non_zero_expr) < min_samples:
                raise ValueError(f"insufficient_samples ({len(non_zero_expr)}<{min_samples})")
            if np.var(non_zero_expr) < 1e-6:
                raise ValueError("zero_variance")

            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=random_state,
                max_iter=max_iter
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                gmm.fit(non_zero_expr.reshape(-1, 1))

            # Select high-expression component
            means = gmm.means_.flatten()
            main_component = np.argmax(means)
            main_mean = means[main_component]
            
            # Compute probabilities and generate annotation
            probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
            anno_non_zero = (probs >= prob_threshold).astype(int)
            
            if np.sum(anno_non_zero) == 0:
                raise ValueError("no_positive_cells")
            
            # Compute expression threshold
            positive_expr = non_zero_expr[anno_non_zero == 1]
            threshold = np.min(positive_expr)
            
            # For 3-component model, check threshold validity
            if n_components >= 3 and threshold > main_mean:
                raise ValueError(f"threshold {threshold:.2f} > μ ({main_mean:.2f})")
            
            # Store module annotation
            anno_col = f"{module_id}"
            annotations.loc[non_zero_mask, anno_col] = anno_non_zero
            
            # Update statistics
            stats.update({
                'threshold': threshold,
                'anno_one': int(anno_non_zero.sum()),
                'anno_zero': int(len(anno_non_zero) - anno_non_zero.sum()),
                'components': [
                    {   'component': i,
                        'mean': float(gmm.means_[i][0]),
                        'var': float(gmm.covariances_[i][0][0]),
                        'weight': float(gmm.weights_[i])
                    } 
                    for i in range(n_components)
                ],
                'main_component': int(main_component)
            })

        except Exception as e:
            stats.update({
                'status': 'failed',
                'error_info': str(e),
                'components': [],
                'threshold': np.nan,
                'anno_one': 0,
                'anno_zero': expr_values.size
            })
            
            # Fallback strategy
            if enable_fallback and n_components > 2:
                try:
                    # Try 2-component model (without threshold check)
                    gmm = GaussianMixture(
                        n_components=2,
                        random_state=random_state,
                        max_iter=max_iter
                    )
                    gmm.fit(non_zero_expr.reshape(-1, 1))
                    
                    # Select high-expression component
                    means = gmm.means_.flatten()
                    main_component = np.argmax(means)
                    
                    probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
                    #anno_non_zero = (probs >= prob_threshold).astype(int)
                    anno_non_zero = (probs >= ((1 - (1 - prob_threshold) * 1e-2))).astype(int)
                    
                    if anno_non_zero.sum() > 0:
                        positive_expr = non_zero_expr[anno_non_zero == 1]
                        threshold = np.min(positive_expr)
                        
                        stats.update({
                            'status': 'success',
                            'final_components': 2,
                            'threshold': threshold,
                            'anno_one': int(anno_non_zero.sum()),
                            'anno_zero': int(len(anno_non_zero) - anno_non_zero.sum()),
                            'components': [
                                {   'component': 0,
                                    'mean': float(gmm.means_[0][0]),
                                    'var': float(gmm.covariances_[0][0][0]),
                                    'weight': float(gmm.weights_[0])
                                },
                                {   'component': 1,
                                    'mean': float(gmm.means_[1][0]),
                                    'var': float(gmm.covariances_[1][0][0]),
                                    'weight': float(gmm.weights_[1])
                                }
                            ],
                            'main_component': int(main_component)
                        })
                        annotations.loc[non_zero_mask, f"{module_id}"] = anno_non_zero
                except Exception as fallback_e:
                    stats['error_info'] += f"; Fallback failed: {str(fallback_e)}"

        finally:
            if stats['status'] == 'success':
                print(f"{module_id} processed, {stats['status']}, anno cells : {stats['anno_one']}")
            else:
                print(f"{module_id} processed, {stats['status']}")

            if stats.get('components'):
                stats['components'] = str(stats['components'])
            else:
                stats['components'] = 'None'
            stats_records.append(stats)

    # Store annotations in adata.obs        
    annotations.columns = [f"{col}_anno" for col in annotations.columns]
    adata.obs = pd.concat([adata.obs, annotations], axis=1)
    
    # Store statistics in adata.uns
    stats_records_df = pd.DataFrame(stats_records)
    stats_records_df = pd.DataFrame(stats_records)
    if 'module_stats' in adata.uns:
        existing_stats = adata.uns['module_stats']
        # For each module, update existing records with new data
        for mid in stats_records_df['module_id'].unique():
            new_row = stats_records_df.loc[stats_records_df['module_id'] == mid].iloc[0]
            mask = existing_stats['module_id'] == mid
            if mask.any():
                num_rows = mask.sum()
                new_update_df = pd.DataFrame([new_row] * num_rows, index=existing_stats.loc[mask].index)
                existing_stats.loc[mask] = new_update_df
            else:
                existing_stats = pd.concat([existing_stats, pd.DataFrame([new_row])], ignore_index=True)
        existing_stats.dropna(how='all', inplace=True)
        adata.uns['module_stats'] = existing_stats
    else:
        adata.uns['module_stats'] = stats_records_df


# smooth_annotations
def smooth_annotations(adata, module_list=None, embedding_key='spatial', k_neighbors=24, min_annotated_neighbors=1):
    """
    Smooth spatial annotations by processing each module's annotation.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
      module_list (list): List of modules to smooth; if None, all modules are used.
      embedding_key (str): Key in adata.obsm for spatial coordinates (default 'spatial').
      k_neighbors (int): Number of KNN neighbors (default 24); may need adjustment based on technology and cell density.
      min_annotated_neighbors (int): Minimum number of neighbors with annotation 1 required to retain the annotation (default 1).
    """
    # Check input: ensure the embedding key exists in adata.obsm
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Please ensure the coordinate exists.")
    
    # If module_list is not provided, get all modules from adata.uns['module_stats']
    if module_list is None:
        module_list = adata.uns['module_stats']['module_id'].unique()

    # Remove existing smoothed annotation columns if they exist
    existing_columns = [f"{mid}_anno_smooth" for mid in module_list if f"{mid}_anno_smooth" in adata.obs]
    if existing_columns:
        print(f"Removing existing smooth annotation columns: {existing_columns}")
        adata.obs.drop(columns=existing_columns, inplace=True)

    # Extract spatial coordinates and the annotation columns to be smoothed
    embedding_coords = adata.obsm[embedding_key]
    module_annotations = adata.obs.loc[:, [f"{mid}_anno" for mid in module_list]]
    n_cells, _ = module_annotations.shape

    # Compute KNN neighbors based on the embedding coordinates
    print(f"\nCalculating {k_neighbors} nearest neighbors for each cell based on {embedding_key} embedding...\n")
    k = k_neighbors + 1  # include self
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embedding_coords)
    _, indices = nbrs.kneighbors(embedding_coords)  # indices: KNN indices for each cell

    # Initialize the smoothed annotation matrix
    smooth_annotations = np.zeros_like(module_annotations, dtype=int)

    # Smooth each module's annotation
    for mid in module_annotations.columns:
        module_values = module_annotations[mid].values  # current module's annotation values
        smooth_values = np.zeros(n_cells, dtype=int)  # initialize smoothed values

        for i in range(n_cells):
            if module_values[i] == 1:  # only process cells with annotation 1
                neighbor_values = module_values[indices[i, 1:]]  # KNN neighbor annotations (excluding self)
                if np.sum(neighbor_values) >= min_annotated_neighbors:  # if at least min_annotated_neighbors neighbors are 1
                    smooth_values[i] = 1

        # Store the smoothed values in the matrix
        smooth_annotations[:, module_annotations.columns.get_loc(mid)] = smooth_values
        print(f"{mid.replace('_anno', '')} processed. removed cells: {np.sum(module_values) - np.sum(smooth_values)}, remain cells: {np.sum(smooth_values)}")

    # Save the smoothed annotations to adata.obs
    smooth_annotations = pd.DataFrame(smooth_annotations,
                                      index=adata.obs_names,
                                      columns=[f"{mid}_smooth" for mid in module_annotations.columns])
    adata.obs = pd.concat([adata.obs, smooth_annotations], axis=1)

    print("\nAnnotation smoothing completed. Results stored in adata.obs.\n")


# integrate_annotations
def integrate_annotations(adata,
                          module_list=None,
                          keep_modules=None,
                          result_anno='annotation',
                          embedding_key='spatial',
                          k_neighbors=24,
                          use_smooth=True,
                          neighbor_majority_frac=0.90
                          ):
                      
    """
    Integrate cell annotations from multiple programs using the following logic:
      1) Optionally use smoothed annotations (controlled by use_smooth);
      2) Automatically compute k_neighbors nearest neighbors;
      3) If a cell is annotated by multiple programs:
         3.1 If >= neighbor_majority_frac (adjustable) of its neighbors belong to one program, select that program;
         3.2 Otherwise, decide based on expression scores, the higher the value, the higher the priority.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing module annotations.
      module_list (list): List of modules to integrate; if None, all modules are used.
      keep_modules (list): List of prioritized modules; if a cell is annotated by these, only consider the intersection.
      result_anno (str): Column name for the integrated annotation (default 'annotation').
      embedding_key (str): Key in adata.obsm for KNN coordinates (default 'spatial').
      k_neighbors (int): Number of KNN neighbors (default 24); may need adjustment based on technology and cell density.
      use_smooth (bool): Whether to use smoothed annotations (default True).
      neighbor_majority_frac (float): If a module's annotation accounts for >= this fraction among neighbors, it is directly selected (default 0.90).
    """
    # Check input: ensure embedding_key exists in adata.obsm.
    if  embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Please ensure the coordinate exists.")
    
    # Check if the integrated annotation column already exists; if so, remove it.
    if adata.obs.get(result_anno) is not None:
        print(f"NOTE: The '{result_anno}' already exists in adata.obs, which will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)
    
    # Check and extract annotation columns.
    if module_list is None:
        module_list = adata.uns['module_stats']['module_id'].unique()
 
    if use_smooth:
        # Identify modules missing smoothed annotation columns.
        missing_smooth = [mid for mid in module_list if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'Smooth anno': {missing_smooth}. Using 'anno' instead.")
        # Use smoothed columns if available; otherwise, use the original '_anno' columns.
        existing_columns = []
        for mid in module_list:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in module_list if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(module_list):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    
    # 1) Compute nearest neighbors.
    print(f"\nCalculating {k_neighbors} nearest neighbors for each cell based on {embedding_key} embedding...\n")
    embedding_coords = adata.obsm[embedding_key]
    k = k_neighbors + 1  # include self
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embedding_coords)
    _, indices = nbrs.kneighbors(embedding_coords)

    n_obs = adata.n_obs
    combined_annotation = [None] * n_obs  # Store final integrated annotation for each cell.

    # 2) Pre-calculate expression scores based on ranked expression.
    expr_score = {}
    anno_dict = {}
    for mid in module_list:
        module_col = f"{mid}_exp"
        if module_col not in adata.obs.columns:
            raise KeyError(f"'{module_col}' not found in adata.obs.")        
        rank_vals = adata.obs[module_col].rank(method='dense', ascending=False).astype(int)
        expr_score[mid] = rank_vals.values
    
    # 3) Build annotation mask.
    for mid in module_list:
        if f"{mid}_anno" in existing_columns:
            anno_col = f"{mid}_anno"
        else:
            anno_col = f"{mid}_anno_smooth"    
        anno_dict[mid] = adata.obs[anno_col].values
    
    unclear_cells = 0
    unclear_cells_neighbor = 0
    unclear_cells_rank = 0
    # 3) Integrate annotations for each cell.
    for i in range(n_obs):
        # Find which modules annotate the cell.
        annotated_modules = [p for p in module_list if anno_dict[p][i]]
        if len(annotated_modules) > 1:
            unclear_cells += 1

        # If keep_modules is provided, take the intersection (prioritize these modules).
        if keep_modules is not None:
            intersection = [p for p in annotated_modules if p in keep_modules]
            if intersection:
                annotated_modules = intersection
        
        if len(annotated_modules) == 0:
            combined_annotation[i] = "None"
            continue
        if len(annotated_modules) == 1:
            combined_annotation[i] = annotated_modules[0]
            continue
        
        # Multiple annotations => Step 1: neighbor voting.
        neighbor_idx = indices[i, 1:]  # exclude self
        n_nb = len(neighbor_idx)

        # Count annotations among neighbors for each module.
        neighbor_counts = {}
        for mid in annotated_modules:
            neighbor_anno = anno_dict[mid][neighbor_idx]
            neighbor_counts[mid] = np.sum(neighbor_anno)

        # Check if any module meets the neighbor majority fraction.
        annotated_modules_re = []
        for mid in annotated_modules:
            frac = neighbor_counts[mid] / n_nb
            if frac >= neighbor_majority_frac:
                annotated_modules_re.append(mid)
        
        if len(annotated_modules_re) == 1:
            combined_annotation[i] = annotated_modules_re[0]
            unclear_cells_neighbor += 1
            continue
        elif len(annotated_modules_re) > 1:
            annotated_modules = annotated_modules_re    

        # Step 2: If neighbor voting is indecisive, use expression score.
        best_mid = None
        best_score = n_obs + 1  # initialize with a large value 
        for mid in annotated_modules:
            sc = expr_score[mid][i]
            if sc < best_score:
                best_score = sc
                best_mid = mid
            
        combined_annotation[i] = best_mid
        unclear_cells_rank += 1
    
    # Print statistics about cells with multiple annotations.
    print(f"\n{unclear_cells} of total {n_obs} cells have multiple annotations. Among them, ")
    print(f"    {unclear_cells_neighbor} cells are resolved by neighbors.")
    print(f"    {unclear_cells_rank} cells are resolved by expression score.")

    # Update integrated annotation in adata.obs.
    annotation_df = pd.Series(combined_annotation, index=adata.obs.index, name=result_anno)
    adata.obs[result_anno] = annotation_df


# calculate_module_overlap
def calculate_module_overlap(adata, module_list, use_smooth=True):
    """
    Compute the overlap ratios between modules based on annotation columns in adata.obs.
    
    The returned DataFrame has the following seven columns:
      - module_a: ID of module A
      - module_b: ID of module B
      - overlap_ratio_a: Overlap count divided by the cell count of module A
      - overlap_ratio_b: Overlap count divided by the cell count of module B
      - overlap_ratio_union: Overlap count divided by the cell count of the union of module A and module B
      - count_a: Number of cells annotated by module A
      - count_b: Number of cells annotated by module B
    
    Pairs with zero overlap are omitted. The final DataFrame is sorted by module_a (ascending)
    and then by overlap_ratio_a (descending).

    Parameters:
      adata (AnnData): AnnData object. The annotation columns should exist in adata.obs with names 
                       "{module_id}_anno" or "{module_id}_anno_smooth".
      module_list (list): List of module IDs to compute overlaps.
      use_smooth (bool): Whether to use smoothed annotation columns (default True).

    Returns:
      pd.DataFrame: DataFrame containing the overlap statistics.
    """
    # If module_list is not provided, get modules from uns['module_stats']
    if module_list is None:
        module_list = adata.uns['module_stats']['module_id'].unique()
    
    # Determine which annotation column to use for each module.
    if use_smooth:
        missing_smooth = [mid for mid in module_list if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'smoothed anno': {missing_smooth}. Using 'anno' instead.")
        existing_columns = []
        for mid in module_list:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in module_list if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(module_list):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    # Build a dictionary mapping module IDs to their annotation (0/1) arrays.
    anno_dict = {}
    for prog in module_list:
        if f"{prog}_anno" in existing_columns:
            anno_col = f"{prog}_anno"
        else:
            anno_col = f"{prog}_anno_smooth"
        anno_dict[prog] = adata.obs[anno_col].values
    
    overlap_records = []
    # Iterate over all pairs of modules.
    for mod_a, mod_b in itertools.combinations(module_list, 2):
        a = anno_dict[mod_a]
        b = anno_dict[mod_b]
        count_a = np.sum(a)
        count_b = np.sum(b)
        overlap = np.sum((a == 1) & (b == 1))
        if overlap == 0:
            continue
        
        ratio_a = overlap / count_a if count_a > 0 else np.nan
        ratio_b = overlap / count_b if count_b > 0 else np.nan
        union_count = count_a + count_b - overlap
        ratio_union = overlap / union_count if union_count > 0 else np.nan
        
        overlap_records.append({
            "module_a": mod_a,
            "module_b": mod_b,
            "overlap_ratio_a": ratio_a,
            "overlap_ratio_b": ratio_b,
            "overlap_ratio_union": ratio_union,
            "count_a": count_a,
            "count_b": count_b
        })
    
    df = pd.DataFrame(overlap_records)
    df.sort_values(by=["module_a", "overlap_ratio_a"], ascending=[True, False], inplace=True)
    df.index = range(df.shape[0])
    return df




############################################################################################################
# Old functions
# integrate_annotations_old
def integrate_annotations_old(adata, module_list=None, result_anno = "annotation", use_smooth=True):
    """
    Integrate module annotations into a single cell annotation.
    
    Parameters:
        adata (anndata.AnnData): AnnData object containing module annotations.
        module_list (list): List of module IDs to integrate; if None, all modules are used.
        result_anno (str): Column name for the integrated annotation (default 'annotation').
        use_smooth (bool): Whether to use smoothed annotations (default True).
    """
    # Check if the integrated annotation column already exists
    if adata.obs.get(result_anno) is not None:
        print(f"NOTE: The '{result_anno}' already exists in adata.obs, which will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)

    # Determine and extract annotation columns
    if module_list is None:
        module_list = adata.uns['module_stats']['module_id'].unique()

    if use_smooth:
        # Identify modules missing smoothed annotation columns
        missing_smooth = [mid for mid in module_list if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'Smooth anno': {missing_smooth}. Using 'anno' instead.")
        # For modules with smoothed columns, use them; otherwise, use the original '_anno' columns
        existing_columns = []
        for mid in module_list:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in module_list if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(module_list):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    
    # Extract module annotations
    module_annotations = adata.obs.loc[:, [mid for mid in existing_columns]]

    # Compute the number of annotated cells for each module
    module_counts = module_annotations.sum(axis=0)

    # Sort modules by the number of annotated cells in ascending order
    sorted_modules = module_counts.sort_values().index

    # Initialize an empty Series to store the final cell annotations
    cell_annotations = pd.Series(index=module_annotations.index, dtype=object)

    # Annotate cells iteratively
    for module in sorted_modules:
        module_name = module.replace("_anno_smooth", "")
        module_name = module_name.replace("_anno", "")
        # Find cells with annotation value 1 for the current module
        cells_to_annotate = module_annotations.index[module_annotations[module] == 1]

        # Only annotate cells that have not been annotated yet
        cells_to_annotate = cells_to_annotate[cell_annotations[cells_to_annotate].isna()]

        # Set these cells to the current module name
        cell_annotations[cells_to_annotate] = module_name

    # Add the final cell annotations to adata.obs
    adata.obs[result_anno] = cell_annotations

    print(f"Cell annotation completed. Results stored in adata.obs['{result_anno}'].")

