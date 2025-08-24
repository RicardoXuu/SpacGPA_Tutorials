
# %%
# 修正关于GO和MP富集分析中校正P值排序问题以及Symbol列问题。
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

# %% 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
import SpacGPA as sg

# %%
# 读取GGM，r3版本
start_time = time.time()
ggm = sg.load_ggm("/dta/ypxu/SpacGPA/Article_Info/Codes_For_SpacGPA_Reproducibility/data/Fig3/ggm_data/MOSTA_E16.5_E1S1_r3.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取GGM，r3版本
start_time = time.time()
ggm = sg.load_ggm("/dta/ypxu/SpacGPA/Article_Info/Codes_For_SpacGPA_Reproducibility/data/Fig3/ggm_data/Mouse_Pup_5K_r3.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")




# %%
ggm.SigEdges


# %%
# 更新后的测试
start_time = time.time()
sg.go_enrichment_analysis(ggm, species="mouse", padjust_method="BH", pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
re_go = ggm.go_enrichment
sg.mp_enrichment_analysis(ggm, species="mouse", padjust_method="BH", pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
re_mp = ggm.mp_enrichment

# %%
re_mp















# %%
# 新版函数测试，解决了校正P值和富集分析中的Symbol列缺失问题
import numpy as np
import pandas as pd
import os
import sys
import gzip
import time
import mygene
from scipy.stats import hypergeom

# Set the base directory and Ref directory
BASE_DIR = "/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1/SpacGPA/"
DATA_DIR = os.path.join(BASE_DIR, "Ref_for_Enrichment")

# go_enrichment_analysis
def go_enrichment_analysis(self,
                           species="mouse",
                           padjust_method="BH",
                           pvalue_cutoff=0.05
                           ):
    """
    Perform GO enrichment analysis using module information from a ggm object.

    Parameters:
        self: An object containing module information (ggm.modules) and background genes (ggm.gene_name).
        species: Species for the GO annotation file. Default is "mouse". 
                 * make sure the species is consistent with the GO annotation file before using this function.
                 * one can create a new GO annotation file for a different species named "species.GO.annotation.txt.gz" and "species.gene.symbl.txt.gz".
        padjust_method: Method for adjusting p-values. choose from "BH" or "Bonferroni". Default is "BH".
        pvalue_cutoff: P-value cutoff for selecting significant GO terms. Default is 0.05.
    """    
    # Check if the GO annotation files exist
    go_name_file = f"{DATA_DIR}/GO.names.txt.gz"
    go_annotation_file = f"{DATA_DIR}/{species}.GO.annotation.txt.gz"
    gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
    if os.path.exists(go_annotation_file) and os.path.exists(go_name_file) and os.path.exists(gene_symbl_file):
        pass
    else:
        raise ValueError("GO annotation files not found.")
    
    # Define a function for adjusting p-values
    def p_adjust(pvalues, method="BH"):
        """
        Adjust p-values with Benjamini-Hochberg (BH) or Bonferroni.
        - Returns adjusted p-values aligned to the original order.
        - Preserves NaNs.
        """
        p = np.asarray(pvalues, dtype=float)
        out = np.full_like(p, np.nan, dtype=float)

        # operate only on finite values
        mask = np.isfinite(p)
        if not np.any(mask):
            return out

        pv = p[mask]
        m = pv.size

        method = method.upper()
        if method == "BH":
            # 1) sort ascending
            order = np.argsort(pv)
            pv_sorted = pv[order]

            # 2) raw BH factor
            ranks = np.arange(1, m + 1, dtype=float)
            adj = pv_sorted * m / ranks

            # 3) step-up (enforce monotonicity from right to left)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            adj = np.clip(adj, 0.0, 1.0)

            # 4) place back to original positions of the finite subset
            adj_back = np.empty_like(adj)
            adj_back[order] = adj
            out[mask] = adj_back

        elif method == "BONFERRONI":
            out[mask] = np.clip(pv * m, 0.0, 1.0)
        else:
            raise ValueError("Unsupported method. Choose 'BH' or 'Bonferroni'.")

        return out
    
    print(f"\nReading GO term information for |{species}|...")
    # Extract module information from ggm.modules
    modules = self.modules[['gene', 'module_id']].drop_duplicates()
    bk_genes = self.gene_name.tolist()  # Background genes
    total_gene_num = len(bk_genes)

    # Process GO annotation file
    allgo = pd.read_csv(go_annotation_file, sep="\t", header=None).drop_duplicates()
    allgo = allgo[allgo[0].isin(bk_genes)]  # Filter by background genes
    bk_go_count = allgo[1].value_counts()
    bk_go_count = bk_go_count[bk_go_count < 2500]  # Filter GO terms with < 2500 annotations
    allgo = allgo[allgo[1].isin(bk_go_count.index)]

    # Read in GO names
    go_name = pd.read_csv(go_name_file, sep="\t", header=None, comment="#", quoting=3)
    go_name.set_index(0, inplace=True)

    # Read in gene symbols
    gene_table = pd.read_csv(gene_symbl_file, sep="\t", header=None, comment="#", quoting=3)
    gene_symbl = gene_table.set_index(0)[1].to_dict()
    all_genes = list(set(modules['gene'].tolist() + bk_genes))
    missing_genes = set(all_genes) - set(gene_symbl.keys())
    gene_symbl.update({gene: gene for gene in missing_genes})

    # GO enrichment analysis
    all_table = []  # List to collect all results
    module_name = modules['module_id'].unique().tolist()
    print(f"\nStart GO enrichment analysis ...")
    for i in module_name:
        selected_genes = modules[modules['module_id'] == i]['gene'].tolist()
        module_size_val = len(selected_genes)
        
        # Get the GO terms associated with the selected genes
        module_go = allgo[allgo[0].isin(selected_genes)]
        module_go_count = module_go[1].value_counts()

        go_ids = module_go_count.index.tolist()
        if len(go_ids) > 0:
            # Pre-compute in_c, in_g for all GO terms
            in_c_list = [module_go_count[j] for j in go_ids]  # GO count in module
            in_g_list = [bk_go_count[j] for j in go_ids]  # GO count in background

            # Batch compute p-values for all GO terms
            p_values = hypergeom.sf(np.array(in_c_list) - 1, total_gene_num, np.array(in_g_list), module_size_val)
            # Adjust p-values 
            p_values_adjusted = p_adjust(p_values, method=padjust_method)
            num_significant = (p_values_adjusted <= pvalue_cutoff).sum()
            sys.stdout.write(f"\rFound {num_significant} significant enriched GO terms in {i}       ")
            sys.stdout.flush()
            del num_significant 

            if len(p_values_adjusted[p_values_adjusted <= pvalue_cutoff]) > 0:    
                # Filter out GO terms with pValueAdjusted greater than 0.05
                valid_go_ids = [go_ids[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values = [p_values[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values_adjusted = [p_values_adjusted[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_c = [in_c_list[idx] for idx in range(len(in_c_list)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_g = [in_g_list[idx] for idx in range(len(in_g_list)) if p_values_adjusted[idx] <= pvalue_cutoff]

                # Prepare the new rows for the valid GO terms
                new_rows = []
                for idx, go_id in enumerate(valid_go_ids):
                    genes_with_go = "/".join(sorted(module_go[module_go[1] == go_id][0].tolist()))
                    symbols_with_go = "/".join([gene_symbl.get(g, g) for g in genes_with_go.split("/")])
                    new_row = {
                        "module_id": i,
                        "module_size": module_size_val,
                        "go_rank": 1,  # Rank will be assigned later
                        "go_id": go_id,
                        "go_category": go_name.loc[go_id, 1],
                        "go_term": go_name.loc[go_id, 2],
                        "module_go_count": valid_in_c[idx],
                        "genome_go_count": valid_in_g[idx],
                        "total_gene_number": total_gene_num,
                        "genes_with_go_in_module": genes_with_go,
                        "symbols_with_go_in_module": symbols_with_go,
                        "pValue": valid_p_values[idx],
                        "pValueAdjusted": valid_p_values_adjusted[idx]
                    }
                    new_rows.append(new_row)

                # Create the module table from the new rows
                module_table = pd.DataFrame(new_rows)
                module_table = module_table.sort_values(by="pValue")
                module_table["go_rank"] = range(1, len(module_table) + 1)

                # Append the current module table to all_table
                all_table.append(module_table)
    
    # Merge all module tables into a single DataFrame
    if all_table:
        all_table = pd.concat(all_table, ignore_index=True)
        all_table.reset_index(drop=True, inplace=True)
        if all_table['genes_with_go_in_module'].equals(all_table['symbols_with_go_in_module']):
            all_table = all_table.drop(columns=['symbols_with_go_in_module'])
        self.go_enrichment = all_table
        print(f"\nGO enrichment analysis completed. Found {len(all_table)} significant enriched GO terms total.")
    else:
        print("No significant GO term found.")


# %%
start_time = time.time()
go_enrichment_analysis(ggm, species="mouse", padjust_method="Bonferroni", pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
new_go = ggm.go_enrichment
# %%

# %%
start_time = time.time()
sg.go_enrichment_analysis(ggm, species="mouse", padjust_method="BH", pvalue_cutoff=0.05)
print(f"Time: {time.time() - start_time:.5f} s")
old_go = ggm.go_enrichment

# %%
new_go['genes_with_go_in_module'][0].split("/")

# %%
new_go['symbols_with_go_in_module'][0].split("/")

# %%
print(old_go)
