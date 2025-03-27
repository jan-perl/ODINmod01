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
#checks voor algortimes, op deelverzameling Utrecht
# -

import glob

import pandas as pd
import numpy as np
import os as os
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

# +
#os.system("pip install geopandas contextily rasterio rasterstats")
# -

import geopandas
import contextily as cx
plt.rcParams['figure.figsize'] = [10, 6]

import rasterio
import rasterio.plot
from rasterio.transform import Affine  
from rasterio import features

import rasterstats
import rasteruts1

calcgdir="../intermediate/calcgrids"

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

os.system("pip install itables")

Rf_net_buurt['area_geo'] = Rf_net_buurt.area
Rf_net_buurt['center']= Rf_net_buurt.representative_point()
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

#buurten excl ongelukkig kleine laatste buurt
itotUtr= Rf_net_buurt[(Rf_net_buurt['PV_NAAM']=='Utrecht') & 
                      (Rf_net_buurt['BU_CODE']!='BU19613901')].copy().reset_index()
itotUtr

itotUtr.plot(column="MXI_22",legend=True, cmap='OrRd')



# +
#from https://rasterio.readthedocs.io/en/latest/quickstart.html
def mkexampsetnl():    
    x = np.linspace(0,280000, 240)
    y = np.linspace(300000, 625000, 180)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-2 * np.log(2) * ((X - 160000) ** 2 + (Y - 500000) ** 2) / 100000 ** 2)
    Z2 = np.exp(-3 * np.log(2) * ((X - 100000) ** 2 + (Y - 400000) ** 2) / 200000 ** 2)
    Z = np.float32(10.0 * (Z2 - Z1))
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



lasttifname=calcgdir+'/land.tif'
gridNL100 = rasteruts1.createNLgrid(100,lasttifname,1,'')

gridNL100.write(xi['Z'],1)
#mag niet: plt.imshow(gridNL100.read(1), cmap='pink')
gridNL100.close()

dataset = rasterio.open(lasttifname)

dataset.width

dataset.height

{i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}

dataset.bounds

dataset.crs

plt.imshow(dataset.read(1), cmap='pink',origin='lower')

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

lasttifname=calcgdir+'/landUmxi-Ut01.tif'
gridNL100b = rasteruts1.createNLgrid(100,lasttifname,2,'Ut')

imageMXIo= rasteruts1.makerasterdef(itotUtr,'MXI_22',gridNL100b)

imageMXIp= rasteruts1.makerasterpts(itotUtr,'MXI_22',gridNL100b)

gridNL100b.write(imageMXIo,1)
gridNL100b.write(imageMXIp,2)
gridNL100b.close()

dataset2 = rasterio.open(lasttifname)


def setaxreg(ax,reg):
    if reg=='htn':
        ax.set_xlim(left=137000, right=143000)
        ax.set_ylim(bottom=444000, top=452000)
    elif reg=='utr':    
        ax.set_xlim(left=113000, right=180000)
        ax.set_ylim(bottom=480000, top=430000)


fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show(dataset2, cmap='OrRd',ax=ax)
setaxreg(ax,'htn')

#read back points, using rasterio directly
coord_sample = [(x, y) for x, y in zip(itotUtr[ "center"].x, itotUtr[ "center"].y)]
itotUtr["mxifromgrida"] = [x[0] for x in dataset2.sample(coord_list,indexes=1)]
itotUtr["mxifromgridp"] = [x[0] for x in dataset2.sample(coord_list,indexes=2)]

itotUtr["mxifromgridp"]

rasteruts1.misstats("mxifromgrida","MXI_22",itotUtr)            

itotUtr[abs(itotUtr["mxifromgrida"] - 0.94999998807907) <1e-6]

fietskern1= rasteruts1.kernelfietspara()
dcalc=(fietskern1['Z2D']-fietskern1['Z'])
print(np.max(np.abs(dcalc)) )

testfietso= rasteruts1.convfietsimg(imageMXIo,fietskern1)

fillrela=True

# +
#open en close binnen 1 cel om verwarring te voorkomen

# +
lasttifname=calcgdir+'/oriTN.tif'
from rasterio.enums import ColorInterp
gridori = rasteruts1.createNLgrid(100,lasttifname,7,'Ut')
if fillrela:
    image1 = rasteruts1.makerasterdef(itotUtr,'R_MXI22T',gridori)
    image2 = rasteruts1.makerasterdef(itotUtr,'R_MXI22N',gridori)
else:
    image1 = rasteruts1.makerasterpts(itotUtr,'O_MXI22T',gridori)
    image2 = rasteruts1.makerasterpts(itotUtr,'O_MXI22N',gridori)

image4= rasteruts1.convfietsimg(image1,fietskern1 ) 
image5= rasteruts1.convfietsimg(image2,fietskern1) 
image6= rasteruts1.convfietsimg( imageMXIo,fietskern1) 
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
# -

image4g= rasteruts1.convfiets2d(image1,fietskern1['Z'] )     

print('Maximum relative error:', np.max( np.abs(image4g- image4) ) / np.max(np.abs(image4)) )
from scipy.ndimage.filters import convolve as scipy_convolve

#scipy_result vermoedelijk te traag voor landelijk
# %timeit scipy_result = scipy_convolve(image1, fietskern1['Z'],  mode='constant', cval=0.0, origin=0)

#deze mogelijk ook voor cirkels
# %timeit image4g= rasteruts1.convfiets2d(image1 ,fietskern1['Z'] )      

# %timeit image4= rasteruts1.convfietsimg(image1,fietskern1 ) 

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

dataset3 = rasterio.open(lasttifname)

dataset3.tags(1)

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset3,5), cmap='OrRd',ax=ax)
setaxreg(ax,'htn')

# +

#heel NL niet geschaald
rasteruts1.plotrb(dataset3,4,5)    
# -

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset3,1),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((dataset3,2),cmap='Blues',ax=ax,alpha=0.5)
setaxreg(ax,'htn')

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset3,4),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((dataset3,5),cmap='Blues',ax=ax,alpha=0.5)
setaxreg(ax,'htn')

fig, axhist = plt.subplots(1, 1)
rasterio.plot.show_hist(
     (dataset3), bins=50, lw=0.0, stacked=False, alpha=0.3,
     histtype='stepfilled', title="Histogram", ax=axhist)
axhist.set_xlabel('Waarde')
axhist.set_ylabel('Aantal malen')
axhist.set_title('Title')
axhist.set_xlim(xmin=0.05)

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
# +
#nieuwe poging:
#tif file: 1e image indices, 2e image aantal pix per deel , 3+ images data, som genormaliseerd
# -


lasttifname=calcgdir+'/oriTN2-Ut.tif'
gridNL100c = rasteruts1.createNLgrid(100,lasttifname,8,'Ut')

dfrefs= rasteruts1.makegridcorr (itotUtr,gridNL100c)

dfrefs.dtype

plt.imshow(dfrefs, cmap='pink')
plt.colorbar()

missptdf= rasteruts1.findmiss(itotUtr,dfrefs)
print( len(missptdf.index), missptdf.index)
missptdf[['area_pix','area_diff','area_geo']]    

import seaborn
seaborn.scatterplot(data=itotUtr,x='area_geo',y='area_pix')

dfrefs2=rasteruts1.testmissfill(dfrefs,missptdf,gridNL100c,itotUtr)  

imagelst=rasteruts1.mkimgpixavgs(gridNL100c,dfrefs,True,fietskern1,
                                 itotUtr[['O_MXI22T','O_MXI22N']])    

len(imagelst)

np.array(imagelst).shape

plt.imshow(imagelst[0])
plt.colorbar()

gridNL100c.close()
dataset4 = rasterio.open(lasttifname)

stat4s1=rasterstats.zonal_stats( itotUtr[ "geometry"], dataset4.read(3),
                              affine=dataset3.transform, stats='sum mean')
stat4s2=rasterstats.zonal_stats( itotUtr[ "geometry"], dataset4.read(5),
                              affine=dataset3.transform, stats='sum mean')

# +
#fillrela=True werkt best mooi via mean (DSD_MXI22T 1.8e0/8.94e7)  en dan oppervlakten gebruiken
#en het telt sowieso tot de goede totalen op
#fillrela=False werkt beter via sum  (DSD_MXI22T 9.2e5/8.94e7) dan mean
if False:
    itotUtr['GSD_MXI22T'] = np.fromiter((s['sum'] for s in stats1),float) *gridm*gridm
    itotUtr['GSD_MXI22N'] = np.fromiter((s['sum'] for s in stats2),float) *gridm*gridm
    itotUtr['GSM_MXI22T'] = np.fromiter((s['mean'] for s in stats1),float) *itotUtr['area_geo']  
    itotUtr['GSM_MXI22N'] = np.fromiter((s['mean'] for s in stats2),float) *itotUtr['area_geo']  
    itotUtr['DSD_MXI22T'] = abs(itotUtr['GSD_MXI22T'] -itotUtr['O_MXI22T'] ) 
    itotUtr['DSM_MXI22T'] = abs(itotUtr['GSM_MXI22T'] -itotUtr['O_MXI22T'] )
else:
    itotUtr['GSD_MXI22T'] = np.fromiter((s['sum'] for s in stat4s1),float) 
    itotUtr['GSD_MXI22N'] = np.fromiter((s['sum'] for s in stat4s2),float) 
    itotUtr['GSM_MXI22T'] = np.fromiter((s['mean'] for s in stat4s1),float) * \
            itotUtr['area_pix']  /100/100
    itotUtr['GSM_MXI22N'] = np.fromiter((s['mean'] for s in stat4s2),float) * \
             itotUtr['area_pix']   /100/100
    itotUtr['DSD_MXI22T'] = abs(itotUtr['GSD_MXI22T'] -itotUtr['O_MXI22T'] ) 
    itotUtr['DSM_MXI22T'] = abs(itotUtr['GSM_MXI22T'] -itotUtr['O_MXI22T'] )

sr1=itotUtr[['GSD_MXI22T','GSD_MXI22T','GSM_MXI22T','GSM_MXI22T','O_MXI22T',
             'DSD_MXI22T','DSM_MXI22T'] ]
np.sum(sr1 )
# -

#maar nu hebben we ook een veel sneller alternatief voor rasterstats.zonal_stats
catlst=dataset4.read(1)
stat4s1s = rasteruts1.sumpixarea(catlst,dataset4.read(3) )
stat4s2s = rasteruts1.sumpixarea(catlst,dataset4.read(5) )
stat4s1c = rasteruts1.countpixarea(catlst )
stat4s2c = rasteruts1.countpixarea(catlst )
stat4s1a = stat4s1s / stat4s1c 
stat4s1a = stat4s1s / stat4s1c 

if 1==1:
    itotUtr['GSD_MXI22T'] = np.fromiter((s['sum'] for s in stat4s1),float) 
    itotUtr['GSD_MXI22N'] = np.fromiter((s['sum'] for s in stat4s2),float) 
    itotUtr['GSM_MXI22T'] = np.fromiter((s['mean'] for s in stat4s1),float) * \
            itotUtr['area_pix']  /100/100
    itotUtr['GSM_MXI22N'] = np.fromiter((s['mean'] for s in stat4s2),float) * \
             itotUtr['area_pix']   /100/100
    itotUtr['DMD_MXI22T'] = abs(itotUtr['GSD_MXI22T'] - stat4s1s ) 
    itotUtr['DMM_MXI22T'] = abs(itotUtr['GSM_MXI22T'] - stat4s1a * \
            itotUtr['area_pix']  /100/100 )
sr2=itotUtr[['GSD_MXI22T','GSD_MXI22T','GSM_MXI22T','GSM_MXI22T','O_MXI22T',
             'DMD_MXI22T','DMM_MXI22T'] ]
np.sum(sr2 )

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset4,3),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((dataset4,5),cmap='Blues',ax=ax,alpha=0.5)
setaxreg(ax,'htn')

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset4,4),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((dataset4,6),cmap='Blues',ax=ax,alpha=0.5)
setaxreg(ax,'htn')

#mogen we ook rekenen? -> NEE
dwerk=dataset4.read(6)-dataset4.read(4)
#notice that axis transformation has vanished -> it does not work
fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show(dwerk, cmap='OrRd',ax=ax)
setaxreg(ax,'htn')

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset4,7), cmap='OrRd',ax=ax)
setaxutr(ax)

fig, ax = plt.subplots()
base=itotUtr.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset4,8), cmap='OrRd',ax=ax)
setaxutr(ax)

# +
#en maak landelijk
# -

#dit was een onbeschreven prerequisite, omdat we index gebruiken als rij nummer, en 
# ook bij dfrefsd
Rf_net_buurt=Rf_net_buurt.reset_index()

lasttifname=calcgdir+'/oriTN2-NL.tif'
gridNL100d = rasteruts1.createNLgrid(100,lasttifname,8,'')

dfrefsd= rasteruts1.makegridcorr (Rf_net_buurt,gridNL100d)

plt.imshow(dfrefsd!=0, cmap='pink')
plt.colorbar()

dfrefsd.dtype
len(Rf_net_buurt)

missptdfd= rasteruts1.findmiss(Rf_net_buurt,dfrefsd)
print( len(missptdfd.index), missptdfd.index)

dfrefs2d=rasteruts1.testmissfill(dfrefsd,missptdfd,gridNL100d,Rf_net_buurt)  

itotUtr[['O_MXI22T','O_MXI22N']].dtypes

Rf_net_buurt[['O_MXI22T','O_MXI22N']].dtypes

imagelstd=rasteruts1.mkimgpixavgs(gridNL100d,dfrefsd,True,fietskern1,
                                 Rf_net_buurt[['O_MXI22T','O_MXI22N']])   

gridNL100d.close()
dataset5 = rasterio.open(lasttifname)

fig, ax = plt.subplots()
#base=Rf_net_buurt.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((dataset5,4), cmap='OrRd',ax=ax)

# +
#in order to resolve issue 0003: write xypccoordpc4.xlsx
#uses meta-analysis from fitparatab1-ogr01-tots.xlsx (from export from ODIN1KAFmo) : 
#    after power-law correction of OW, #OT and OW contribute the same amount of movements

# +
#now, also write some statistics
#code copied from ODIN1Kafmo, bus should be separate
#to be run BEFORE ODINcatVN
# -

stryear='2020'
cbspc4data =pd.read_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl")
cbspc4data= cbspc4data.sort_values(by=['postcode4']).reset_index()

cbspc4data['oppervlak'] = cbspc4data.area
cbspc4data['aantal_inwoners'] = np.where(cbspc4data['aantal_inwoners'] <0,0,
                                         cbspc4data['aantal_inwoners'] )

pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid= rasterio.open(pc4tifname)

#rudifunset, heb originele data niet nodig, alleen grid
#Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
#Rf_net_buurt.reset_index(inplace=True,drop=True)
#gemaakt in ROfietsbalans2
rudifuntifname=calcgdir+'/oriTN2-NL.tif'
rudifungrid= rasterio.open(rudifuntifname)


def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
pc4inwgcache = getcachedgrids(pc4inwgrid)
rudifungcache = getcachedgrids(rudifungrid)

expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0}


def mkgeowgtxy (pc4data,pc4grid,rudigrid,rudifil,pu,res):
    debug=False
    #pc4lst=pc4grid.read(1)
    pc4lst=pc4grid[1]
    outdf=pc4data[['postcode4','aantal_inwoners','oppervlak']].rename(columns={'postcode4':'PC4'} )
    outdfst= outdf.copy()
    
    R=dict()
#    R_LW= rudifungrid.read(3)
#    R_LT= rudifungrid.read(5)
    R_LW= rudigrid[3]
    R_LT= rudigrid[5]
    R_LO =  R_LT- R_LW   
    R_LW = np.power(R_LW,pu['LW'])
    R_LO = np.where(R_LO <0,-np.power(-R_LO,pu['LO']), np.power(R_LO,pu['LO']) )
    R_twg = R_LW + R_LO
    twgvals = rasteruts1.sumpixarea(pc4lst,R_twg) 
    #output corrgrid = np.zeros([grid.width, grid.height],dtype=np.int32) 
    x = np.linspace(0,rudifil.width*res, rudifil.width)
    y = np.linspace(rudifil.height*res+ 300000,300000, rudifil.height)
    X, Y = np.meshgrid(x, y)
    outdf['avgRfunX'] = rasteruts1.sumpixarea(pc4lst,X* R_twg) / twgvals
    outdf['avgRfunY'] = rasteruts1.sumpixarea(pc4lst,Y* R_twg) / twgvals
    reppts = cbspc4data.representative_point()
    outdf['georepX'] =  reppts.x
    outdf['georepY'] =  reppts.y
    return(outdf)
xypccoord= mkgeowgtxy(cbspc4data,pc4inwgcache,rudifungcache,rudifungrid,expdefs,100)
xypccoord

sns.scatterplot(data=xypccoord,x="avgRfunX",y="georepX")
sns.scatterplot(data=xypccoord,x="avgRfunY",y="georepY")

#for use in other calculations
xypccoord.to_excel("../intermediate/xypccoordpc4.xlsx")



print("Finished")


