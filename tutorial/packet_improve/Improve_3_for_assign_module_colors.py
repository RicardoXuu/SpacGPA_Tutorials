# %%
# 尝试设计函数来为模块分配颜色
# 使用MOATA_E16.5_E1S1 数据进行分析
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
# 读取GGM
start_time = time.time()
ggm = sg.load_ggm("data/MOSTA_E16.5_E1S1.ggm.h5")
print(f"Time: {time.time() - start_time:.5f} s")

# %%
# 读取数据
adata = sc.read_h5ad("data/MOSTA_E16.5_E1S1_ggm_anno.h5ad")


# %%
adata.uns['module_colors']

# %%
# 设置配色方案
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors
from scanpy.plotting.palettes import default_20, vega_10, vega_20

def assign_module_colors(adata, ggm_key='ggm', seed=1):
    """
    Create and store a consistent color mapping for gene modules.

    Depending on the number of modules, this function selects from
    Scanpy/Vega discrete palettes (up to 100 modules) or XKCD_COLORS
    (above 100 modules). If `seed` is non-zero, colors are shuffled
    reproducibly; if zero, the order is deterministic.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing module metadata under
        adata.uns['ggm_keys'][ggm_key].
    ggm_key : str, default 'ggm'
        Key in adata.uns['ggm_keys'] that defines module stats and
        where to store the resulting color dictionary.
    seed : int, default 1
        Random seed for reproducible shuffling. A value of 0 means
        no randomization.

    Returns
    -------
    dict
        Mapping from module IDs to hex color codes or named colors.
    """
    # Retrieve GGM metadata from adata.uns
    ggm_meta = adata.uns.get('ggm_keys', {}).get(ggm_key)
    if ggm_meta is None:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_info_key = ggm_meta.get('module_info')
    mod_col_val   = ggm_meta.get('module_colors')

    # Load module statistics and extract module IDs
    module_info = adata.uns.get(mod_info_key)
    if module_info is None:
        raise ValueError(f"Module Info not found in adata.uns['{mod_info_key}']")
    module_ids = module_info['module_id'].unique()

    n_all = len(module_ids)
    if n_all == 0:
        return {}
    n_modules = min(n_all, 806)

    # Initialize random number generator if needed
    rng = np.random.RandomState(seed) if seed != 0 else None
    
    # Select base color palette according to module count
    if n_modules <= 10:
        colors = vega_10[:n_modules]
    elif n_modules <= 20:
        colors = default_20[:n_modules]
    elif n_modules <= 40:
        # combine two 20-color palettes to reach up to 40
        colors = (default_20 + vega_20)[:n_modules]
    elif n_modules <= 60:
        # combine three 20-color palettes to reach up to 60
        tab20b = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20b').colors]
        colors = (default_20 + vega_20 + tab20b)[:n_modules]
    elif n_modules <= 100:
        tab20b  = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20b').colors]
        tab20c  = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20c').colors]
        set3    = [mpl.colors.to_hex(c) for c in plt.get_cmap('Set3').colors]
        pastel2 = [mpl.colors.to_hex(c) for c in plt.get_cmap('Pastel2').colors]
        palette_combo = default_20 + vega_20 + tab20b + tab20c + set3 + pastel2
        colors = palette_combo[:n_modules]
    else:
        # More than 100 modules: sample from filtered XKCD_COLORS by HSV order
        xkcd_colors = mcolors.XKCD_COLORS
        to_remove = ['gray','grey','black','white','light',
                     'lawngreen','silver','gainsboro','snow',
                     'mintcream','ivory','fuchsia','cyan']
        filtered = {
            name: col for name, col in xkcd_colors.items()
            if not any(key in name for key in to_remove)
        }
        sorted_hsv = sorted(
            ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(col)[:3])), name))
            for name, col in filtered.items()
        )
        sorted_names = [name for hsv, name in sorted_hsv]
        if rng is not None:
            if len(sorted_names) >= n_modules:
                colors = list(rng.choice(sorted_names, size=n_modules, replace=False))
            else:
                base = sorted_names.copy()
                rem = n_modules - len(base)
                vir = plt.cm.viridis
                extra = [mpl.colors.to_hex(vir(i/(rem-1))) for i in range(rem)]
                colors = base + extra
        else:
            L = len(sorted_names)
            step = L / n_modules
            colors = [sorted_names[int(i * step)] for i in range(n_modules)]

    # Handle modules beyond 806 by repeating or sampling
    if n_all > 806:
        extra_count = n_all - 806
        if rng is not None:
            extra = list(rng.choice(colors, size=extra_count, replace=True))
        else:
            extra = [colors[i % len(colors)] for i in range(extra_count)]
        colors += extra

    # Shuffle full color list if RNG provided
    if rng is not None:
        perm = rng.permutation(len(colors))
        colors = [colors[i] for i in perm]

    # Build the module-to-color dictionary and store in adata.uns
    color_dict = {module_ids[i]: colors[i] for i in range(n_all)}
    if isinstance(mod_col_val, str):
        adata.uns.setdefault(mod_col_val, {}).update(color_dict)
    else:
        raise ValueError(f"Invalid module color key in adata.uns['{mod_col_val}']: {mod_col_val}")




# %%
# 生成配色方案
assign_module_colors(adata, ggm_key='ggm', seed=4)  
# %%
adata.uns['module_colors']
# %%
