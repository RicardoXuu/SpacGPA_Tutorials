
# %%
# 一些问题修复
import numpy as np
import pandas as pd
import random
import time
import torch
import scanpy as sc
import anndata
import os
import gc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# %%
# 切换工作目录
os.getcwd()
workdir = '/dta/ypxu/ST_GGM/VS_Code/ST_GGM_dev_1'
os.chdir(workdir)
os.getcwd()

# %%
from ST_GGM_dev_1 import *




# %%
# GGM计算相关的问题
# 问题1，ggm计算的部分改为，create_ggm。

# %%
# 问题2，GO和MP函数内置到ggm。

# %%
# 问题3，select_number的自动化选择：
#        5000个基因到20000个基因，取 1/10的基因数目，平均采样100次。
#        大于20000，取2000个基因，平均采样100次。
#        小于5000，取500个基因，平均采样100次。
#        小于500，用全部的基因，只算1次。

# %%
# 问题4，Pcor的阈值选择。优先考虑FDR, 细胞数目越多，可以接受越小的Pcor阈值。

# %%
# 问题5，Chuking_size的默认阈值的设置改为较大一点的值。

# %%
# 问题6，放射状模块的去除设计不够严谨，需要进一步优化。

# %%
# 问题7，设计函数，提取指定模块的edges用于绘图。可以添加参数，考虑是否同时提取模块的GO和MP注释结果。

# %%
# 问题8，优化参数命名

# %%
# 问题9，设计储存函数，保存ggm的结果。










# %%
# 细胞注释相关的问题
# 问题1，关于计算平均表达值。当使用新的ggm结果注释已经存在module expression的adata时，会报错

# %%
# 问题2，关于计算平均表达值。添加可选参数，计算模块内每个基因的莫兰指数。

# %%
# 问题3，关于模块注释的全部函数，添加反选参数，用来反向排除模块。

# %%
# 问题4，关于模块注释的全部函数, 细胞按模块的注释结果改为category类型。而不是现在的0，1，int类型。并注意，之后在涉及到使用这些数据的时候还要换回int类型。

# %%
# 问题5，关于高斯混合分布，设计activity模块的排除标准。尽量不使用先验知识，

# %%
# 问题6，关于高斯混合分布，阈值和主成分数目的关系优化。

# %%
# 问题7，关于高斯混合分布，除了使用高斯混合分布，也考虑表达值的排序。
#       对于一个模块，只有那些表达水平大于模块最大表达水平（或者为了防止一些离散的点，可以考虑前20个或者30个细胞的平均值作为模块最大表达水平）的一定比例的细胞才被认为是注释为该模块的

# %%
# 问题8，关于平滑处理，在使用的时候，无法仅处理部分模块。

# %%
# 问题9，关于合并注释，优化keep modules的参数。

# %%
# 问题10，关于合并注释，尝试引入模块的整体莫兰指数，来评估模块的空间分布。如果一个模块的莫兰指数很高，则优先考虑该模块的细胞的可信度。

# %%
# 问题11，关于合并注释，尝试结合louvain或者leiden的聚类结果，在每个聚类之内使用模块来精准注释。

# %%
# 问题12，关于合并注释，注释结果中，字符串None改为空值的None。

# %%
# 问题13，关于合并注释，在adata的uns中添加一个配色方案，为每个模块指定配色，特别是模块过多的时候。

# %%
# 问题14，关于合并注释，neighbor_majority_frac参数似乎会导致activity模块的权重过高。考虑将其设置为大于1的值。