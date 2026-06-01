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

# +
#Vergelijk valideer active mode modellen op originele data

# +
#eerste deel: alle code voor MainUseSelFactorV='FactorVGen'
#steeds in makkelijk aanroepbare eenheden
#eind eerste deel: datzelfde in een subroutine incl checks
#tweede deel : kijk ook naar ongefilterde en gefitte data
# -

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import re
import time
import glob

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt
from matplotlib import colors 
import matplotlib.ticker as ticker

import RUDIbas

from importlib import reload  # Python 3.4+
if True:
        foo = reload(RUDIbas)

myname='actrawval'
suprtests= myname in RUDIbas.suprtests 
suprdata= myname in RUDIbas.suprdata
#suprtests=True
print ('Suprtests',suprtests)

assertdbg=RUDIbas.assertdbg

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"

#voor gemeentegrenzen; kost hier wel heel veel geheugen voor. Kijken hoe dit te vermijden
import ODiN2readpkl

from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from sklearn import linear_model
import seaborn

import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

RUDIbas.suprtests = RUDIbas.suprtests+['cbspc4plot']
import cbspc4plot


cbspc4data= cbspc4plot.cbspc4data
if 0==1:
    pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
    pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid=cbspc4plot.pc4inwgrid

#rudifunset, heb originele data niet nodig, alleen grid
#Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
#Rf_net_buurt.reset_index(inplace=True,drop=True)
#gemaakt in ROfietsbalans2
rudifungrid= cbspc4plot.rudifungrid

getcachedgrids= cbspc4plot.getcachedgrids
addbasemkmsch= cbspc4plot.addbasemkmsch
pc4inwgcache = getcachedgrids(pc4inwgrid)
rudifungcache = getcachedgrids(rudifungrid)


RUDIbas.suprtests = RUDIbas.suprtests+['actmodval']
import actmodval

# +
#nu ook ruwe data lezen, en labelling en plot routines
# -

RUDIbas.suprtests = RUDIbas.suprtests+['ODiN1stexplore']
import ODiN1stexplore



print ("Finished")


