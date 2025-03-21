
import numpy as np
import pandas as pd

# get_module_edges 
def get_module_edges(self, module_id):
    """
    Extract edges within a module.
    Parameters:
        module_id: The ID of the module to extract.
    Returns:
        module_edges: The edges within the module.
    """
    module_list = self.modules['module_id'].unique()
    if module_id not in module_list:
        raise ValueError("Module ID not found.")
    if self.modules is None:
        raise ValueError("Please run find_modules first.")
    genes_in_module = self.modules.loc[self.modules["module_id"] == module_id, "gene"].unique()

    mask = self.SigEdges["GeneA"].isin(genes_in_module) & self.SigEdges["GeneB"].isin(genes_in_module)
    module_edges = self.SigEdges[mask].copy()
    module_edges.index = range(len(module_edges))
    module_edges.insert(0, "module_id", module_id)
    
    return module_edges

# get_module_anno 
def get_module_anno(self, module_id, add_enrich_info=True, top_n=None, term_id=None):
    """
    Get the annotation information of a specific module.
    Parameters:
        self: GGM object
        module_id: str, a module id in the modules_summary
        add_enrich_info: bool, whether to add GO and MP enrichment information to the module annotation
        top_n: int, the top n GO or MP terms to add to the module annotation, default as None
            use when add_enrich_info is True and too many GO or MP terms are enriched in the module
        term_id: a list of GO or MP term ids to add to the module annotation, default as None
            use for specific GO or MP terms to add to the module annotation
    """
    if term_id is not None and top_n is not None:
        raise ValueError("term_id and top_n cannot be specified at the same time.")
    
    if self.modules is None:
        raise ValueError("No modules found. run find_modules function first.")
    
    if module_id not in self.modules['module_id'].values:
        raise ValueError(f"{module_id} not found in modules.")

    module_anno = self.modules[self.modules['module_id'] == module_id].copy()
    if add_enrich_info:
        if self.go_enrichment is not None:
            go_df = self.go_enrichment[self.go_enrichment['module_id'] == module_id].copy()
            if go_df.empty:
                print(f"No significant enrichment GO term found for {module_id}.")
            else:
                if top_n is not None:
                    go_df = go_df.head(top_n)
                for _, row in go_df.iterrows():
                    go_id = row['go_id']
                    go_term = row['go_term']
                    gene_list = row['genes_with_go_in_module'].split("/")
                    module_anno[go_id] = module_anno['gene'].apply(lambda g: go_term if g in gene_list else None)

        if self.mp_enrichment is not None:
            mp_df = self.mp_enrichment[self.mp_enrichment['module_id'] == module_id].copy()
            if mp_df.empty:
                print(f"No significant enrichment MP term found for {module_id}.")
            else:
                if top_n is not None:
                    mp_df = mp_df.head(top_n)
                for _, row in mp_df.iterrows():
                    mp_id = row['mp_id']
                    mp_term = row['mp_term']
                    gene_list = row['genes_with_mp_in_module'].split("/")
                    module_anno[mp_id] = module_anno['gene'].apply(lambda g: mp_term if g in gene_list else None)
    if term_id is not None:
        save_id = np.concatenate((go_df['go_id'].values, mp_df['mp_id'].values))
        remove_id = [x for x in save_id if x not in term_id]
        keep_id = [x for x in term_id if x in save_id]
        wrong_id = [x for x in term_id if x not in save_id]
        if len(keep_id) == 0:
            print(f"Make sure the term_id listed are in the enrichment information.")
        else:    
            if len(wrong_id) > 0:
                print(f"The term_id {wrong_id} not in the enrichment information.")
            module_anno = module_anno.drop(columns=remove_id)  

    return module_anno