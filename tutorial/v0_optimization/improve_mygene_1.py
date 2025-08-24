
# %%
# 修正mygene调用问题
import torch
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import mygene
import subprocess
import tempfile
import sys
import time
import gc
import os


# run_mcl
import torch
import numpy as np
import pandas as pd
import networkx as nx
import sys
import mygene
import time

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
convert_to_symbols = True
module_df = ggm.modules.copy()
module_df = module_df.drop(columns=['symbol'])
species = 'mouse' 

# %%
# 改进代码
import os, re, gzip, io
import pandas as pd

# Set the base directory and Ref directory
BASE_DIR = "/dta/ypxu/SpacGPA/Dev_Version/SpacGPA_dev_1/SpacGPA/"
DATA_DIR = os.path.join(BASE_DIR, "Ref_for_Enrichment")

def ensure_gene_symbol_table(species, gene_symbl_file) -> dict:
    """
    Ensure a local Ensembl->symbol lookup exists for the given species.
    1) If file exists, load and return mapping.
    2) Else, fetch ALL genes (with Ensembl IDs) for the species from MyGene, save to gzip TSV, then return.
       File format (no header): <ensembl_gene_id>\t<symbol>
    """
    # 1) cache first
    if os.path.exists(gene_symbl_file):
        df = pd.read_csv(
            gene_symbl_file, sep="\t", header=None, names=["ensembl", "symbol"],
            dtype=str, comment="#", quoting=3
        ).dropna(subset=["ensembl"]).drop_duplicates(subset=["ensembl"], keep="first")
        return df.set_index("ensembl")["symbol"].to_dict()

    print(f"\n[MyGene] Building ALL-genes Ensembl→symbol table for species='{species}' ...")
    time.sleep(1)

    import mygene
    mg = mygene.MyGeneInfo()
    mapping = {}

    # fold hits into mapping (first-win; ensembl can be dict or list)
    def add_hits(hits):
        for doc in hits:
            if not isinstance(doc, dict):
                continue
            sym = doc.get("symbol")
            ens = doc.get("ensembl")
            if isinstance(ens, dict):
                gid = ens.get("gene")
                if isinstance(gid, str) and gid not in mapping:
                    mapping[gid] = sym if isinstance(sym, str) and sym else gid
            elif isinstance(ens, list):
                for e in ens:
                    if isinstance(e, dict):
                        gid = e.get("gene")
                        if isinstance(gid, str) and gid not in mapping:
                            mapping[gid] = sym if isinstance(sym, str) and sym else gid

    def pull_all(q_string: str):
        # try fetch_all (streaming)
        try:
            cursor = mg.query(q=q_string, species=species,
                              fields="symbol,ensembl.gene",
                              size=1000, fetch_all=True)
            got = False
            for chunk in cursor:
                if isinstance(chunk, dict) and "hits" in chunk:
                    hits = chunk["hits"]
                elif isinstance(chunk, list):
                    hits = chunk
                else:
                    hits = [chunk]
                if hits:
                    got = True
                    add_hits(hits)
            if got:
                return
        except Exception:
            pass
        # fallback: paginated
        try:
            from_ = 0
            size = 1000
            while True:
                page = mg.query(q=q_string, species=species,
                                fields="symbol,ensembl.gene",
                                size=size, from_=from_)
                hits = page.get("hits", [])
                if not hits:
                    break
                add_hits(hits)
                from_ += size
                if from_ > 2_000_000:  # hard cap
                    break
        except Exception:
            pass

    # 2) build mapping
    pull_all("_exists_:ensembl.gene")
    if not mapping:
        pull_all("ensembl.gene:*")

    if not mapping:
        raise RuntimeError(f"[MyGene] Failed to build ALL-genes mapping for species='{species}'.")

    # 3) save cache (two columns, no header)
    os.makedirs(os.path.dirname(gene_symbl_file), exist_ok=True)
    with gzip.open(gene_symbl_file, "wt", encoding="utf-8") as gz:
        for gid, sym in mapping.items():
            gz.write(f"{gid}\t{sym}\n")
    print(f"[MyGene] Saved mapping to: {gene_symbl_file}  (n={len(mapping)})")

    return mapping

# %%
# 测试
species = 'human'
if convert_to_symbols:
    print("\nConverting Ensembl IDs to gene symbols...")
    gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
    ensembl_to_symbol = ensure_gene_symbol_table(species, gene_symbl_file)
    # Add the 'Symbol' column to the DataFrame.
    module_df['symbol'] = module_df['gene'].map(ensembl_to_symbol)

# %%
module_df













# %%
# 原版函数
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