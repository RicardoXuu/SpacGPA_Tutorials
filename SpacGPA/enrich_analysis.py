
import numpy as np
import pandas as pd
import os
import sys
import gzip
import time
import mygene
from scipy.stats import hypergeom

# Set the base directory and Ref directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        """Adjust p-values using Benjamini-Hochberg method.
           method: Adjust p-values using different methods, including Benjamini-Hochberg, Bonferroni, Holm, and FDR.
        """
        pvalues = np.array(pvalues)
        n = len(pvalues)
        if method == "BH":
            # Benjamini-Hochberg procedure for controlling False Discovery Rate (FDR)
            ranked_pvalues = np.argsort(np.argsort(pvalues))
            adjusted_pvalues = pvalues * n / (ranked_pvalues + 1)
            adjusted_pvalues = np.minimum(adjusted_pvalues, 1)
        elif method == "Bonferroni":
            # Bonferroni correction: simple method, divide p-value by the number of tests
            adjusted_pvalues = pvalues * n
            adjusted_pvalues = np.minimum(adjusted_pvalues, 1)  # Ensure p-value doesn't exceed 1
        else:
            raise ValueError("Unsupported method,Please choose from 'BH' or 'Bonferroni'.")
        return adjusted_pvalues
    
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
            # Adjust p-values using BH method
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
        self.go_enrichment = all_table
        print(f"\nGO enrichment analysis completed. Found {len(all_table)} significant enriched GO terms total.")
    else:
        print("No significant GO term found.")



# mp_enrichment_analysis
def mp_enrichment_analysis(self,
                           species="mouse",
                           padjust_method="BH",
                           pvalue_cutoff=0.05):
    """
    Perform MP enrichment analysis using module information from a ggm object.

    Parameters:
        self: An object containing module information (ggm.modules) and background genes (ggm.gene_name).
        species: Species for the MP annotation file. Default is "mouse". 
        padjust_method: Method for adjusting p-values. Choose from "BH" or "Bonferroni". Default is "BH".
        pvalue_cutoff: P-value cutoff for selecting significant MP terms. Default is 0.05.
    """

    # Check if the MP annotation files exist
    mp_name_file = f"{DATA_DIR}/MP.names.txt.gz"
    mp_annotation_file = f"{DATA_DIR}/{species}.MP.annotation.txt.gz"
    gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
    if os.path.exists(mp_annotation_file) and os.path.exists(mp_name_file) and os.path.exists(gene_symbl_file):
        pass
    else:
        raise ValueError("MP annotation files not found.")
    
    # Define a function for adjusting p-values
    def p_adjust(pvalues, method="BH"):
        """Adjust p-values using Benjamini-Hochberg or Bonferroni method."""
        pvalues = np.array(pvalues)
        n = len(pvalues)
        if method == "BH":
            ranked_pvalues = np.argsort(np.argsort(pvalues))
            adjusted_pvalues = pvalues * n / (ranked_pvalues + 1)
            adjusted_pvalues = np.minimum(adjusted_pvalues, 1)
        elif method == "Bonferroni":
            adjusted_pvalues = pvalues * n
            adjusted_pvalues = np.minimum(adjusted_pvalues, 1)  # Ensure p-value doesn't exceed 1
        else:
            raise ValueError("Unsupported method, Please choose from 'BH' or 'Bonferroni'.")
        return adjusted_pvalues
    
    print(f"\nReading MP term information for |{species}|...") 
    # Extract cluster information from ggm.modules
    clusters = self.modules[['gene', 'module_id']].drop_duplicates()
    bk_genes = self.gene_name.tolist()  # Background genes
    total_gene_num = len(bk_genes)

    # Process MP annotation file
    allmp = pd.read_csv(mp_annotation_file, sep="\t", header=None).drop_duplicates()
    allmp = allmp[allmp[0].isin(bk_genes)]  # Filter by background genes
    bk_mp_count = allmp[1].value_counts()
    bk_mp_count = bk_mp_count[bk_mp_count < 2500]  # Filter MP terms with < 2500 annotations
    allmp = allmp[allmp[1].isin(bk_mp_count.index)]

    # Read in MP names
    mp_name = pd.read_csv(mp_name_file, sep="\t", header=None, comment="#", quoting=3)
    mp_name.set_index(0, inplace=True)

    # Read in gene symbols
    gene_table = pd.read_csv(gene_symbl_file, sep="\t", header=None, comment="#", quoting=3)
    gene_symbl = gene_table.set_index(0)[1].to_dict()
    all_genes = list(set(clusters['gene'].tolist() + bk_genes))
    missing_genes = set(all_genes) - set(gene_symbl.keys())
    gene_symbl.update({gene: gene for gene in missing_genes})

    # MP enrichment analysis
    all_table = []  # List to collect all results
    module_name = clusters['module_id'].unique().tolist()
    print(f"\nStart MP enrichment analysis ...")
    
    for i in module_name:
        selected_genes = clusters[clusters['module_id'] == i]['gene'].tolist()
        module_size_val = len(selected_genes)

        # Get the MP terms associated with the selected genes
        module_mp = allmp[allmp[0].isin(selected_genes)]
        module_mp_count = module_mp[1].value_counts()

        mp_ids = module_mp_count.index.tolist()

        if len(mp_ids) > 0:
            # Pre-compute in_c, in_g for all MP terms
            in_c_list = [module_mp_count[j] for j in mp_ids]  # MP count in module
            in_g_list = [bk_mp_count[j] for j in mp_ids]  # MP count in background

            # Batch compute p-values for all MP terms
            p_values = hypergeom.sf(np.array(in_c_list) - 1, total_gene_num, np.array(in_g_list), module_size_val)
            # Adjust p-values using selected method
            p_values_adjusted = p_adjust(p_values, method=padjust_method)
            num_significant = (p_values_adjusted <= pvalue_cutoff).sum()
            sys.stdout.write(f"\rFound {num_significant} significant enriched MP terms in {i}       ")
            sys.stdout.flush()

            if len(p_values_adjusted[p_values_adjusted <= pvalue_cutoff]) > 0:    
                # Filter out MP terms with pValueAdjusted greater than 0.05
                valid_mp_ids = [mp_ids[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values = [p_values[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values_adjusted = [p_values_adjusted[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_c = [in_c_list[idx] for idx in range(len(in_c_list)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_g = [in_g_list[idx] for idx in range(len(in_g_list)) if p_values_adjusted[idx] <= pvalue_cutoff]

                # Prepare the new rows for the valid MP terms
                new_rows = []
                for idx, mp_id in enumerate(valid_mp_ids):
                    genes_with_mp = "/".join(sorted(module_mp[module_mp[1] == mp_id][0].tolist()))
                    new_row = {
                        "module_id": i,
                        "module_size": module_size_val,
                        "mp_rank": 1,  # Rank will be assigned later
                        "mp_id": mp_id,
                        "mp_term": mp_name.loc[mp_id, 1],
                        "mp_description": mp_name.loc[mp_id, 2],
                        "module_mp_count": valid_in_c[idx],
                        "genome_mp_count": valid_in_g[idx],
                        "total_gene_number": total_gene_num,
                        "genes_with_mp_in_module": genes_with_mp,
                        "pValue": valid_p_values[idx],
                        "pValueAdjusted": valid_p_values_adjusted[idx]
                    }
                    new_rows.append(new_row)

                # Create the module table from the new rows
                module_table = pd.DataFrame(new_rows)
                module_table = module_table.sort_values(by="pValue")
                module_table["mp_rank"] = range(1, len(module_table) + 1)

                # Append the current module table to all_table
                all_table.append(module_table)

    # Merge all module tables into a single DataFrame
    if all_table:
        all_table = pd.concat(all_table, ignore_index=True)
        all_table.reset_index(drop=True, inplace=True)
        self.mp_enrichment = all_table
        print(f"\nMP enrichment analysis completed. Found {len(all_table)} significant enriched MP terms total.")
    else:
        print("No significant MP term found.")


# get_GO_annoinfo
def get_GO_annoinfo(species_name=None,
                    species_taxonomy_id=None):
    """
    Download GO annotation and gene symbol mapping for the given species from MyGene.info and save them into compressed files.
    If higher quality GO annotation information is desired, it may be necessary to construct the relevant files manually.
    
    Parameters:
        species_name: set a common name for the species, which will be used to name the output files 
                      and as an anchor for the enrichment analysis function.
                      (e.g. species_name="mouse" for Mus musculus)
            - These common names can be used instead of passing species_taxonomy_id:
                'human'	        for Homo sapiens	
                'mouse'	        for Mus musculus	
                'rat'	        for Rattus norvegicus	
                'fruitfly'	    for Drosophila melanogaster	
                'nematode'	    for Caenorhabditis elegans	
                'zebrafish'	    for Danio rerio	GRCz10 
                'thale-cress'   for Arabidopsis thaliana
                'frog'	        for Xenopus tropicalis	
                'pig'	        for Sus scrofa
                (see https://docs.mygene.info/projects/mygene-py/en/ver_2.3.0/index.html for details.)
            - For other species, you must pass the NCBI taxonomy ID to parameters 'species_taxonomy_id'.

        species_taxonomy_id: NCBI taxonomy ID for the species. 
                             (e.g. species_name='rice', species_taxonomy_id='39947' for Oryza sativa Japonica Group) 
                             (see https://www.ncbi.nlm.nih.gov/taxonomy for details.)

    """
    # Make sure the species name is set
    if species_name is None :
        raise ValueError("Please provide a common name for the species you are searching for.")
    if species_taxonomy_id is None and species_name not in ["human", "mouse", "rat", "fruitfly", "nematode", "zebrafish", "thale-cress", "frog", "pig"]:
        raise ValueError(f"Please provide the NCBI taxonomy ID of {species_name}.")

    # Make sure the GO names file exists \
    go_names_file = f"{DATA_DIR}/GO.names.txt.gz"
    out_annotation_file = f"{DATA_DIR}/{species_name}.GO.annotation.txt.gz"
    out_gene_symbl_file = f"{DATA_DIR}/{species_name}.gene.symbl.txt.gz"

    if os.path.exists(go_names_file):
        pass
    else:
        raise ValueError("GO names file not found.")
    
    # If the GO Anno file and gene symbol file already exist, rebuild them
    if os.path.exists(out_annotation_file) and os.path.exists(out_gene_symbl_file):
        print(f"\nNOTE! The GO annotation files for |{species_name}| will be rebuilt")
        os.remove(out_annotation_file)
        os.remove(out_gene_symbl_file)

    # Set up MyGeneInfo and query for genes with GO annotations
    mg = mygene.MyGeneInfo()
    query = "go:*"
    fields = "ensembl,symbol,go"
    print(f"\nQuerying MyGene.info for {species_name} genes with GO annotations ...")
    time.sleep(1)
    if species_taxonomy_id is not None:
        res0 = mg.query(query, species=species_taxonomy_id, fields=fields, size=0)
    elif species_taxonomy_id is None:
        res0 = mg.query(query, species=species_name, fields=fields, size=0)
    total = res0.get("total", 0)
    print(f"Total genes with GO annotation (reported): {total}")
    
    # Fetch all genes with GO annotations
    print(f"\nFetching all genes with GO annotation ...")
    if species_taxonomy_id is not None:
        res = mg.query(q=query, species=species_taxonomy_id, fields=fields, fetch_all=True)
    elif species_taxonomy_id is None:
        res = mg.query(q=query, species=species_name, fields=fields, fetch_all=True)
    hits = []
    for gene in res:
        hits.append(gene)    
    
    # Collect (gene, GO term) pairs for Ensembl and Symbol separately
    ensembl_annotation_pairs = set()
    symbol_annotation_pairs = set()
    gene2symbol = {}  # Mapping from Ensembl ID to gene symbol

    for hit in hits:
        # Try to get the Ensembl gene ID
        ensembl_info = hit.get("ensembl")
        gene_id = None
        if isinstance(ensembl_info, list):
            gene_id = ensembl_info[0].get("gene")
        elif isinstance(ensembl_info, dict):
            gene_id = ensembl_info.get("gene")
        # Get the gene symbol
        symbol = hit.get("symbol")
        # If both Ensembl and symbol exist, record both annotation types
        if gene_id and symbol:
            ensembl_annotation_pairs.add((gene_id, None))  # GO term will be filled later
            symbol_annotation_pairs.add((symbol, None))
            gene2symbol[gene_id] = symbol
        elif gene_id:
            ensembl_annotation_pairs.add((gene_id, None))
            gene2symbol[gene_id] = gene_id  # Use gene_id if symbol is missing
        elif symbol:
            symbol_annotation_pairs.add((symbol, None))
            gene2symbol[symbol] = symbol
        
        # Process GO annotations
        go_data = hit.get("go")
        if go_data and isinstance(go_data, dict):
            for cat in ["BP", "MF", "CC"]:
                terms = go_data.get(cat)
                if terms:
                    if isinstance(terms, list):
                        for term in terms:
                            go_term = term.get("id")
                            if go_term:
                                if gene_id:
                                    ensembl_annotation_pairs.add((gene_id, go_term))
                                if symbol:
                                    symbol_annotation_pairs.add((symbol, go_term))
                    elif isinstance(terms, dict):
                        go_term = terms.get("id")
                        if go_term:
                            if gene_id:
                                ensembl_annotation_pairs.add((gene_id, go_term))
                            if symbol:
                                symbol_annotation_pairs.add((symbol, go_term))
    
    # Build mapping (Ensembl -> Symbol) and inverse mapping (Symbol -> Ensembl)
    symbol_to_ensembl = {}
    for ensembl_id, sym in gene2symbol.items():
        if sym not in symbol_to_ensembl:
            symbol_to_ensembl[sym] = ensembl_id

    # Fill in missing counterpart information for annotations
    ensembl_annotation_pairs_filled = set()
    for gene_id, go_term in ensembl_annotation_pairs:
        if go_term is None:
            continue
        sym = gene2symbol.get(gene_id, gene_id)
        ensembl_annotation_pairs_filled.add((gene_id, go_term, sym))
    symbol_annotation_pairs_filled = set()
    for sym, go_term in symbol_annotation_pairs:
        if go_term is None:
            continue
        ensembl_id = symbol_to_ensembl.get(sym, sym)
        symbol_annotation_pairs_filled.add((sym, go_term, ensembl_id))
    
    # Read valid GO terms from GO.names.txt.gz
    valid_go_terms = set()
    with gzip.open(go_names_file, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                valid_go_terms.add(parts[0])
    
    # Filter out annotation pairs with GO terms not present in GO.names.txt.gz
    ensembl_annotation_pairs_final = {pair for pair in ensembl_annotation_pairs_filled if pair[1] in valid_go_terms}
    symbol_annotation_pairs_final = {pair for pair in symbol_annotation_pairs_filled if pair[1] in valid_go_terms}
    
    # Write annotation file: first write Ensembl-based annotations, then Symbol-based annotations
    with gzip.open(out_annotation_file, "wt") as f:
        # Write Ensembl-based annotations: format "EnsemblID<TAB>GO term<TAB>GeneSymbol"
        for gene_id, go_term, _ in sorted(ensembl_annotation_pairs_final):
            f.write(f"{gene_id}\t{go_term}\n")
        # Write Symbol-based annotations: format "GeneSymbol<TAB>GO term<TAB>EnsemblID"
        for sym, go_term, _ in sorted(symbol_annotation_pairs_final):
            f.write(f"{sym}\t{go_term}\n")
    
    # Write gene symbol mapping file
    with gzip.open(out_gene_symbl_file, "wt") as f:
        for gene_id, sym in sorted(gene2symbol.items()):
            f.write(f"{gene_id}\t{sym}\n")

    print(f"Files written:\n  {out_annotation_file}\n  {out_gene_symbl_file}")
