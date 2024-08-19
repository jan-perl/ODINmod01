# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import glob

import pandas as pd
import numpy as np
import os as os
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

import geopandas
import contextily as cx
plt.rcParams['figure.figsize'] = [10, 6]

# +
#read rundifun data , inspect and convert
# -

ilst =glob.glob("../dnload2211/Rudifun*/Rudifun*/Rudifun_Netto_Buurt_*.shp")

ilst

itot = pd.concat(map(geopandas.read_file,ilst))

itot

itot.plot(column="MXI_22",legend=True, cmap='OrRd')

itot.to_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 

ilst =glob.glob("../dnload2211/Rudifun*/Rudifun*/Rudifun_Netto_Bouwblok_*.shp")

ilst

itot = pd.concat(map(geopandas.read_file,ilst))

itot.to_pickle("../intermediate/rudifun_Netto_Bouwblok_o.pkl") 

itot.columns


