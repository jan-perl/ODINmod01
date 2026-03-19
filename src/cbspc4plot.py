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
#dit stuk code zat vooraan bij veel py bestanden aks voorbeeld van plot. nu apart gezet

# +
#beschrijf kenmerken
#volledig additief
# binnen cirkels
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

myname='cbspc4plot'
suprtests= myname in RUDIbas.suprtests 
suprdata= myname in RUDIbas.suprdata
#suprtests=True
print ('Suprtests',suprtests)

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"

#voor gemeentegrenzen; kost hier wel heel veel geheugen voor. Kijken hoe dit te vermijden
import ODiN2readpkl

from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
from sklearn import linear_model
import seaborn

import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

stryear='2020'
cbspc4data =pd.read_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl")
cbspc4data= cbspc4data.sort_values(by=['postcode4']).reset_index()

cbspc4data['oppervlak'] = cbspc4data.area
cbspc4data['aantal_inwoners'] = np.where(cbspc4data['aantal_inwoners'] <0,0,
                                         cbspc4data['aantal_inwoners'] )

cbspc4data.dtypes

#providers = cx.providers.flatten()
#providers
prov0=cx.providers.nlmaps.grijs.copy()
print( cbspc4data.crs)
print (prov0)
plot_crs=3857
#data_crs="epsg:28992"
if 1==1:
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['min_zoom']= 0
    prov0['max_zoom'] =12
    print (prov0)

if (not suprtests):
    pland= cbspc4data.plot(alpha=0.4)
    cx.add_basemap(pland, source= prov0,crs=cbspc4data.crs)

#alternatief: gebruik webcoordinaten in plot
if (not suprtests):
    cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
    phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot(alpha=0.4)
    cx.add_basemap(phtn, source= prov0)


# +
#en nu netjes, met schaal in km
def plaxkm(x, pos=None):
      return '%.0f'%(x/1000.)

def addbasemkmsch(ax,mapsrc):
    cx.add_basemap(ax,source= mapsrc,crs="epsg:28992")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))

if (not suprtests):
    fig, ax = plt.subplots(figsize=(6, 4))
    cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
    phtn = cbspc4datahtn.plot(ax=ax,alpha=0.4)
    addbasemkmsch(ax,prov0)
# -

if (not suprtests):
    cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']==3995)]
    phtn = cbspc4datahtn.plot()
    cx.add_basemap(phtn, source= prov0,crs=cbspc4data.crs)

if (not suprdata):
    pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
    pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
    pc4inwgrid= rasterio.open(pc4tifname)

#rudifunset, heb originele data niet nodig, alleen grid
#Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
#Rf_net_buurt.reset_index(inplace=True,drop=True)
#gemaakt in ROfietsbalans2
if (not suprdata):
    rudifuntifname=calcgdir+'/oriTN2-NL.tif'
    rudifungrid= rasterio.open(rudifuntifname)


def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
if (not suprdata):
    pc4inwgcache = getcachedgrids(pc4inwgrid)
    rudifungcache = getcachedgrids(rudifungrid)

print("Finished")


