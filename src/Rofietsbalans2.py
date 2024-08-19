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

import rasterio
import rasterio.plot
from rasterio.transform import Affine  
from rasterio import features

os.system("pip install rasterstats")

import rasterstats

# +
#https://stackoverflow.com/questions/72452203/how-to-make-moving-average-using-geopandas-nearest-neighbors
#import libpysal
#from https://gis.stackexchange.com/questions/421556/performance-problem-with-getting-average-pixel-values-within-buffered-circles
#import qgis

# +
#read rundifun data , inspect and convert
# -

Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 

Rf_net_buurt.columns

Rf_net_buurt

Rf_net_buurt['area_geo'] = Rf_net_buurt.area
Rf_net_buurt['area_geo_diff'] = Rf_net_buurt['Shape_Area'] - Rf_net_buurt['area_geo']
Rf_net_buurt['C_FSI22'] = Rf_net_buurt['bruto_22'] /Rf_net_buurt['area_geo']
Rf_net_buurt['C_FSI22diff'] = Rf_net_buurt['FSI_22'] -Rf_net_buurt['C_FSI22']
Rf_net_buurt['C_GSI22'] = Rf_net_buurt['pand_opp22'] /Rf_net_buurt['Shape_Area']
Rf_net_buurt['C_GSI22diff'] = Rf_net_buurt['GSI_22'] -Rf_net_buurt['C_GSI22']

Rf_net_buurt['O_MXI22T'] = Rf_net_buurt['bvo_woo_22'] +Rf_net_buurt['bvo_log_22']
Rf_net_buurt['O_MXI22N'] = Rf_net_buurt['bruto_22'] - (Rf_net_buurt['bvo_ove_22'] +Rf_net_buurt['opp_sch_22'] )
Rf_net_buurt['R_MXI22T'] = Rf_net_buurt['O_MXI22T'] /Rf_net_buurt['area_geo']  
Rf_net_buurt['R_MXI22N'] = Rf_net_buurt['O_MXI22N'] /Rf_net_buurt['area_geo']  
Rf_net_buurt['C_MXI22']  = Rf_net_buurt['O_MXI22T'] /Rf_net_buurt['O_MXI22N']
Rf_net_buurt['C_MXI22diff'] = Rf_net_buurt['MXI_22'] -Rf_net_buurt['C_MXI22']

Rf_net_buurt[ abs (Rf_net_buurt['area_geo_diff'] ) > 100 ][
      ['Shape_Area','area_geo','area_geo_diff']]

Rf_net_buurt[ abs (Rf_net_buurt['C_MXI22diff'] ) > 1e-2 ][
      ['Shape_Area','area_geo','area_geo_diff','C_MXI22diff','C_MXI22','MXI_22']]

prov_mxitots = Rf_net_buurt.groupby(['PV_NAAM']).agg('sum')[['O_MXI22T','O_MXI22N']]
#prov_mxitots = Rf_net_buurt.dissolve(by='PV_NAAM', aggfunc='sum')
prov_mxitots['PV_MXI22']= prov_mxitots['O_MXI22T'] /prov_mxitots['O_MXI22N']
prov_mxitots 

prov_mxitots[['PV_MXI22']].plot()

itotUtr= Rf_net_buurt[Rf_net_buurt['PV_NAAM']=='Utrecht'].copy()

itotUtr.plot(column="MXI_22",legend=True, cmap='OrRd')


# +
#from https://rasterio.readthedocs.io/en/latest/quickstart.html
def mkexampsetnl():    
    x = np.linspace(0,280000, 240)
    y = np.linspace(300000, 625000, 180)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-2 * np.log(2) * ((X - 160000) ** 2 + (Y - 500000) ** 2) / 100000 ** 2)
    Z2 = np.exp(-3 * np.log(2) * ((X - 100000) ** 2 + (Y - 400000) ** 2) / 200000 ** 2)
    Z = 10.0 * (Z2 - Z1)
    return {'X':X,'Y':Y,'Z':Z }
xi= mkexampsetnl()
#print(xi)
fig, ax = plt.subplots(1, 1) 
# plots contour lines 
ax.contour(xi['X'], xi['Y'], xi['Z']) 
  
ax.set_title('Contour Plot') 
ax.set_xlabel('feature_x') 
ax.set_ylabel('feature_y') 
  
plt.show() 

# -

xi['Z'].dtype


# +
def createNLgrid(res,fn,countin):
#resulution
#coords    "[{300000, 0}, {625000, 280000})"; 
    
    transformx = Affine.translation(0 - res / 2, 625000  - res / 2) * Affine.scale(res, -res)
    transformx
    new_dataset = rasterio.open(
         fn,
         'w',
         driver='GTiff',
         height=(625000 - 300000)/res,
         width=(280000- 0)/res,
         count=countin,
         dtype=xi['Z'].dtype,
         crs='EPSG:28992',
         transform=transformx,
         nodata = np.nan
#          nodata = 0
     )
    return new_dataset

gridNL100 = createNLgrid(100,'../intermediate/land.tif',1)
# -

gridNL100.write(xi['Z'],1)

# +
#mag niet: plt.imshow(gridNL100.read(1), cmap='pink')
# -

gridNL100.close()

dataset = rasterio.open('../intermediate/land.tif')

dataset.width

dataset.height

{i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}

dataset.bounds

dataset.crs

plt.imshow(dataset.read(1), cmap='pink')

rasterio.plot.show(dataset, cmap='pink')

itotUtr[ "center"]= itotUtr.representative_point()
coord_list = [(x, y) for x, y in zip(itotUtr[ "center"].x, itotUtr[ "center"].y)]

coord_list

# +
#attribute<ratio>  G_MXI22T    (NLgrid/rdc_100m) := Results/inbuurt1/R_MXI22T[gconv];
#attribute<ratio>  G_MXI22N    (NLgrid/rdc_100m) := Results/inbuurt1/R_MXI22N[gconv];
#attribute<ratio>  G1_MXI22    (NLgrid/rdc_100m) := Results/inbuurt1/C_MXI22[gconv];
#attribute<ratio>  G2_MXI22    (NLgrid/rdc_100m) := G_MXI22T / G_MXI22N;
#attribute<ratio>  D_MXI22     (NLgrid/rdc_100m) := G1_MXI22 - G2_MXI22;
# -

gridNL100b = createNLgrid(100,'../intermediate/landUmxi.tif',2)


#all_touched = True zorgt dat bij 100 meter grid ca 1.300 of  1.484 van waardes te
# hoog terug komt  in check met if fillrela
# en dat 49 van de 971 waarden in Utrecht bij uitlezen waarde van 
#all_touched = False zorgt dat bij 100 meter grid ca 1.0025467901140264, 1.0006930004604095 van waardes te
# hoog terug komt  in check met if fillrela
# en dat 0 van de 971 waarden in Utrecht bij uitlezen waarde anders worden
def makerasterdef (df,col,grid):
    shapesusd = ((geom,value) for geom, value in zip( df.geometry, df[col]))
    imageout = rasterio.features.rasterize(
            shapes=shapesusd,
            merge_alg=rasterio.features.MergeAlg.replace,
#            all_touched=True,
            default_value=np.nan,
            out_shape=grid.shape,
            transform=grid.transform)
    return imageout
imageMXIo= makerasterdef(itotUtr,'MXI_22',gridNL100b)


def makerasterpts (df,col,grid):
    shapesusd = ((geom,value) for geom, value in zip( df["center"], df[col]))
    imageout = rasterio.features.rasterize(
            shapes=shapesusd,            
            merge_alg=rasterio.features.MergeAlg.add,
            default_value=0,
            out_shape=grid.shape,
            transform=grid.transform)
    return imageout
imageMXIp= makerasterpts(itotUtr,'MXI_22',gridNL100b)

gridNL100b.write(imageMXIo,1)
gridNL100b.write(imageMXIp,2)
gridNL100b.close()

dataset2 = rasterio.open('../intermediate/landUmxi.tif')


# +
def setaxhtn(ax):
    ax.set_xlim(left=137000, right=143000)
    ax.set_ylim(bottom=444000, top=452000)
    
def setaxutr(ax):
    ax.set_xlim(left=113000, right=180000)
    ax.set_ylim(bottom=480000, top=430000)


# -

fig, ax = plt.subplots()
rasterio.plot.show(dataset2, cmap='OrRd',ax=ax)
setaxhtn(ax)

#read back points, using rasterio directly
coord_sample = [(x, y) for x, y in zip(itotUtr[ "center"].x, itotUtr[ "center"].y)]
itotUtr["mxifromgrida"] = [x[0] for x in dataset2.sample(coord_list,indexes=1)]
itotUtr["mxifromgridp"] = [x[0] for x in dataset2.sample(coord_list,indexes=2)]

itotUtr["mxifromgridp"]


def misstats (c1,c2):
    itotUtr["griddif"]= abs(itotUtr[c1] - itotUtr[c2])
    mis=itotUtr[np.isnan(itotUtr[c1])]
    print (( len(itotUtr["griddif"]),len(mis),
    len(itotUtr[itotUtr["griddif"]>1e-6]) ))
    return (itotUtr[itotUtr["griddif"]>1e-6])
misstats("mxifromgrida","MXI_22")            

misstats("mxifromgridp","MXI_22") 

itotUtr[abs(itotUtr["mxifromgrida"] - 0.94999998807907) <1e-6]


#now some kernel and transformations
#note: for gaussian: ony can do 2 1d convolutions
def kernelfietspara():
    maxfietsdist =5000
    maxkernrng = 1* maxfietsdist
    gridstep =100
    FWHMcnv  = 2.0 * np.sqrt (2.0 * np.log (2.0));
    #print(FWHMcnv)
    gaussdenom   = (maxfietsdist/FWHMcnv) ;
    #print(gaussdenom)
    x = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    y = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    X, Y = np.meshgrid(x, y)
    d1 = np.sqrt(X*X+Y*Y)
    Z1 = np.exp(-d1*d1 / (2*gaussdenom* gaussdenom))
    Z1= Z1/ np.sum(Z1)
    K1D= np.exp(-x*x / (2*gaussdenom* gaussdenom))
    K1D= K1D/ np.sum(K1D)
    Z2D= np.outer( K1D, (K1D))
#    print(np.sum(Z1))
#    print('sum')
    Z = Z1
    return {'X':X,'Y':Y,'Z':Z,'K1D':K1D,'Z2D':Z2D }
fietskern1= kernelfietspara()
dcalc=(fietskern1['Z2D']-fietskern1['Z'])
print(np.max(np.abs(dcalc)) )


def cfunc(sl):
    return np.convolve(sl,fietskern1['K1D'])
def convfietsimg(img):
    return np.apply_along_axis(cfunc,1, np.apply_along_axis(cfunc,0, img) )


fillrela=True

# +
#open en close binnen 1 cel om verwarring te voorkomen

# +
from rasterio.enums import ColorInterp
gridori = createNLgrid(100,'../intermediate/oriTN.tif',7)
if fillrela:
    image1 = makerasterdef(itotUtr,'R_MXI22T',gridori)
    image2 = makerasterdef(itotUtr,'R_MXI22N',gridori)
else:
    image1 = makerasterpts(itotUtr,'O_MXI22T',gridori)
    image2 = makerasterpts(itotUtr,'O_MXI22N',gridori)

image4= convfietsimg(image1 ) 
image5= convfietsimg(image2) 
image6= convfietsimg( imageMXIo) 
image7= image4/image5

gridori.write(image1,1)
gridori.write(image2,2)
gridori.write(imageMXIo,3)
gridori.write(image4,4)
#gridori.write_colorinterp(4,  ColorInterp.red)
gridori.write(image5,5)
#gridori.write_colorinterp(5,  ColorInterp.blue)
gridori.write(image6,6)
gridori.write(image7,7)
gridori.close()

# +
#nu wat analyses

# +
if fillrela:
    gridm=100
else:
    gridm=1
mixtirat = np.sum(image1)*gridm*gridm /sum(itotUtr['O_MXI22T']) 
mixnirat = np.sum(image2)*gridm*gridm /sum(itotUtr['O_MXI22N']) 
mixtfrat = np.sum(image4)*gridm*gridm /sum(itotUtr['O_MXI22T']) 
mixnfrat = np.sum(image5)*gridm*gridm /sum(itotUtr['O_MXI22N']) 

print ((mixtirat,mixnirat,mixtfrat,mixnfrat) )
# -

dataset3 = rasterio.open('../intermediate/oriTN.tif')

dataset3.tags(1)

fig, ax = plt.subplots()
rasterio.plot.show((dataset3,5), cmap='OrRd',ax=ax)
setaxhtn(ax)


# +
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def plotrb(dset,c1,c2):
    # Normalize band DN
    red_norm = np.sqrt(normalize(dset.read(c1)))/2+.5
    blue_norm = np.sqrt(normalize(dset.read(c2)))/2+.5
    # Stack bands
    nrg = np.dstack((red_norm, (red_norm+blue_norm)/2, blue_norm))
    # View the color composite
    plt.imshow(nrg)  
#heel NL niet geschaald
plotrb(dataset3,4,5)    
# -

fig, ax = plt.subplots()
rasterio.plot.show((dataset3,1),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((dataset3,2),cmap='Blues',ax=ax,alpha=0.5)
setaxhtn(ax)

fig, ax = plt.subplots()
rasterio.plot.show((dataset3,4),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((dataset3,5),cmap='Blues',ax=ax,alpha=0.5)
setaxhtn(ax)

rasterio.plot.show_hist(
     (dataset3,1), bins=50, lw=0.0, stacked=False, alpha=0.3,
     histtype='stepfilled', title="Histogram")

# +
#nu terug transformeren
# -

stats1=rasterstats.zonal_stats( itotUtr[ "geometry"], dataset3.read(1),
                              affine=dataset3.transform, stats='sum mean')
stats2=rasterstats.zonal_stats( itotUtr[ "geometry"], dataset3.read(2),
                              affine=dataset3.transform, stats='sum mean')

stats1

list(s['sum'] for s in stats1)

# +
#fillrela=True werkt best mooi via mean (DSD_MXI22T 1.8e0/8.94e7)  en dan oppervlakten gebruiken
#en het telt sowieso tot de goede totalen op
#fillrela=False werkt beter via sum  (DSD_MXI22T 9.2e5/8.94e7) dan mean
if fillrela:
    itotUtr['GSD_MXI22T'] = np.fromiter((s['sum'] for s in stats1),float) *gridm*gridm
    itotUtr['GSD_MXI22N'] = np.fromiter((s['sum'] for s in stats2),float) *gridm*gridm
    itotUtr['GSM_MXI22T'] = np.fromiter((s['mean'] for s in stats1),float) *itotUtr['area_geo']  
    itotUtr['GSM_MXI22N'] = np.fromiter((s['mean'] for s in stats2),float) *itotUtr['area_geo']  
    itotUtr['DSD_MXI22T'] = abs(itotUtr['GSD_MXI22T'] -itotUtr['O_MXI22T'] ) 
    itotUtr['DSM_MXI22T'] = abs(itotUtr['GSM_MXI22T'] -itotUtr['O_MXI22T'] )
else:
    itotUtr['GSD_MXI22T'] = np.fromiter((s['sum'] for s in stats1),float) 
    itotUtr['GSD_MXI22N'] = np.fromiter((s['sum'] for s in stats2),float) 
    itotUtr['GSM_MXI22T'] = np.fromiter((s['mean'] for s in stats1),float) * \
            itotUtr['area_geo']  /100/100
    itotUtr['GSM_MXI22N'] = np.fromiter((s['mean'] for s in stats2),float) * \
             itotUtr['area_geo']   /100/100
    itotUtr['DSD_MXI22T'] = abs(itotUtr['GSD_MXI22T'] -itotUtr['O_MXI22T'] ) 
    itotUtr['DSM_MXI22T'] = abs(itotUtr['GSM_MXI22T'] -itotUtr['O_MXI22T'] )

sr1=itotUtr[['GSD_MXI22T','GSD_MXI22T','GSM_MXI22T','GSM_MXI22T','O_MXI22T',
             'DSD_MXI22T','DSM_MXI22T'] ]
np.sum(sr1 )
# -


