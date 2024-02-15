#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:34:46 2023

@author: benwib

Script which defines some user defined colormaps adapted from hclwizard
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

cmap_user = {}
# -------------------------
# ---- Dark Mint
# -------------------------
cmap_name = 'DarkMint'
colors = ("#0E3F5C", "#10445F", "#124862", "#144D65", "#175268", "#1A566B", "#1D5B6E", "#206071", "#246574",
          "#276978", "#2B6E7B", "#2F737E", "#337881", "#387D84", "#3C8187", "#41868A", "#458B8D", "#4A9090",
          "#4F9593", "#549A96", "#599F99", "#5FA49C", "#64A89F", "#6AADA2", "#70B2A5", "#75B7A8", "#7BBCAB",
          "#81C1AE", "#88C6B1", "#8ECBB4", "#94D0B7", "#9BD5BA", "#A1D9BD", "#A8DEC0", "#AEE3C3", "#B5E8C6",
          "#BCEDCA", "#C3F2CD", "#CAF7D1", "#D1FBD4")
cmap_user[cmap_name] = LinearSegmentedColormap.from_list(cmap_name, colors)


# -------------------------
# ---- Blues
# -------------------------
cmap_name = 'Blues'
colors = ("#5669B4", "#838EC4", "#ACB2D4", "#D1D3E3", "#EEEEEE")
cmap_user[cmap_name] = LinearSegmentedColormap.from_list(cmap_name, colors)


# -------------------------
# ---- Purple-Yellow
# -------------------------
cmap_name = 'PurpleYellow'
colors = ("#80146E", "#478EC1", "#7ED5B8", "#F5F2D8")
cmap_user[cmap_name] = LinearSegmentedColormap.from_list(cmap_name, colors)
