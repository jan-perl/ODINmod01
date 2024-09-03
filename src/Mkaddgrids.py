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
#Modellering van ODIN gegevens obv ruimtelijk dichtheden
# -

# todo
# actieve modes per afstandsklasse en type
# gemiddelde afstand per afstandsklasse en type
# active mode kms en passive mode kms per PC naar & van (let op: dubbeltelling)
# referentie reiskms actieve modes
# vergelijking co2 uitstoot active modes met kentallen


# +
#todo
#fit tov fractie (l komt uit data FactorVL)
#waarde:  p=1/(1/l + 1/f) -> f= 1/ (1/p - 1/l) -> divergeert dus alleen als w>.1, anders 0
#gewicht: w= l/ (l-p) -> aparte kolom

# +
#visualiseer relaties getallen, bijv. het 50 % punt(beide componenten gelijk groot) in km
#ook: bijdrage per ring / oppervlak -> multipliers per ring visualiseren
#omdat additief model _. som van componenten is ook terug te rekenen
#uitgaande van 1 PC : som model daar =1
#ga uit ven centrum punt postcode
#-maak grid met grootste ring
#-trek bijdrages vorige ring er helemaal van af (die komen later)
#bepaal coefficienten voor  'OW','OO','OM','OA' als som van alle groepen in ring
#tel op zodat een relatie plaatje op buurt/wijk niveau ontstaat
#valdeer dat de sommen voor die PC kloppen met model uitkomst

# +
#splits code op
#- analyse top PC4 per Motief voor 2 ritten op dag en niet woon kant
#- t.o.v. gebieds oppervlak
#- bij top PC4 maak annotatie naar 1 of meerdere PC6 mogelijk
# maak ook afstandsklassen tabel (met index per indeling naam) en schrijf naar pkl

#
#- maak grid met die toewijzing op 1 pixel & visualiseer met 5 km smooth
#- maak gridder routines in apart werkboek en save de output pkl
#- selecteer alleen grootste naarhuis - motief combinaties , rest: houdt van/naar, motief rest
#      *van/ naar samen selecteren !
#- 

# +
#beschrijf kenmerken
#volledig additief
# binnen cirkels
# -

# scenarios
# #+ 10 % wonen en +10 werken (8% en 10 % oppervlak resp)
# A) beide op plaatsen waar het al is
# b) tegengesteld
# C) werk op huidige plek , wonen alleen waar nu lage dichtheid is (+20 % daar)


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import re
import time

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"

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
#plot_crs=28992
if 1==1:
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['min_zoom']= 0
    prov0['max_zoom'] =12
    print (prov0)

pland= cbspc4data.to_crs(epsg=plot_crs).plot()
cx.add_basemap(pland, source= prov0)

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot()
cx.add_basemap(phtn, source= prov0)

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']==3995)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot()
cx.add_basemap(phtn, source= prov0)

pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid= rasterio.open(pc4tifname)

#rudifunset, heb originele data niet nodig, alleen grid
#Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
#Rf_net_buurt.reset_index(inplace=True,drop=True)
#gemaakt in ROfietsbalans2
rudifuntifname=calcgdir+'/oriTN2-NL.tif'
rudifungrid= rasterio.open(rudifuntifname)


# +
def setaxhtn(ax):
    ax.set_xlim(left=137000, right=143000)
    ax.set_ylim(bottom=444000, top=452000)
    
def setaxutr(ax):
    ax.set_xlim(left=113000, right=180000)
    ax.set_ylim(bottom=480000, top=430000)


# -

def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
pc4inwgcache = getcachedgrids(pc4inwgrid)
rudifungcache = getcachedgrids(rudifungrid)


def calccats(ingr,flgs):
    invs = ingr.flatten()
#    print(invs.sum())
    nonz1 = invs[invs>0]
#    print(nonz1.sum())
    nonz1 = np.sort(nonz1)
#    print(nonz1.sum())
    nonzcs = np.cumsum(nonz1)
#    print(nonzcs[-1])
    nbin=11
    limrel = np.array(range(nbin))
    limidcs=np.searchsorted(nonzcs,limrel*(nonzcs[-1]/(nbin-1)) ) 
    limuse = np.append([-10], nonz1[limidcs] )
#    print([ limrel, limidcs ,limuse] )
    pcatted= pd.cut( invs,limuse,labels=False)
    if flgs=='noextr':
        pcatted = np.where(np.isin( pcatted ,[0,1,nbin-1]) ,0,pcatted-1)
    elif flgs=='midone':
        pcatted = np.where(np.isin( pcatted ,[0,1,nbin-1]) ,0,1)        
    elif flgs=='dfcat':
        pcatted =  pcatted 
    else:
        print ('Error')
    pcatr= np.reshape(pcatted,ingr.shape)
    statsfr = pd.DataFrame (pcatted)
    statsfr.columns=['cat']
    statsfr['bebouwd'] = invs*1e-4
    statsfr['gebied'] = 1
    statsd= statsfr.groupby('cat').agg('sum')
#    print(statsd.T)
    return pcatr
calccats(rudifungcache[3],'noextr')


# +
def _renorm1(pat,tot):
    return np.sum(tot)* pat / (np.sum(pat))

expposs= ['base' ,'same', 'swap', 'icat' ,'scat' ,'fmx1','smx1']

def writeexperiment(expname,incache0,promille,mtrrang,prefix):
    outsetnm= '{}_{:_>6}_{:0>4}_{:0>5}'.format(prefix,expname,promille,mtrrang) 
    print (outsetnm)
    fname = "../intermediate/addgrds/"+outsetnm+'.tif';
    ogrid = rasteruts1.createNLgrid(100,fname,5,'')
    mfact = promille*0.001

    incache=dict()
    incache[3]=np.where(np.isnan( incache0[3]),0, incache0[3])
    incache[5]=np.where(np.isnan( incache0[5]),0, incache0[5])
    incacheoth = incache[5] - incache[3]
    incacheoth = np.where(incacheoth <0,0,incacheoth)
#    print([np.max(incache[3]),np.max(incache[5]-incache[3]),np.min(incache[5]-incache[3]) ])
    mycache=dict()
    cats3= calccats(incache[3],'noextr')
    catso= calccats(incacheoth,'noextr')
    
    if expname == 'same':
        mycache[3] = incache[3]*mfact
        mycache[5] = incacheoth*mfact
    elif expname == 'swap':
        grw= ( incacheoth*mfact *np.sum(incache[3])/np.sum(incacheoth) )
        mycache[3] =  _renorm1 (incacheoth ,incache[3]) *mfact
        mycache[5] =  _renorm1 (incache[3],incacheoth ) *mfact
#        print([np.max(mycache[3]),np.max(mycache[5]-mycache[3]),np.min(mycache[5]-mycache[3]) ])
    elif expname == 'base':
        mycache[3] = incache[3]*0.0
        mycache[5] = incacheoth*0.0
    elif expname == 'icat':    
        mycache[3] = _renorm1 (cats3 ,incache[3]) *mfact
        mycache[5] = _renorm1 (catso ,incacheoth) *mfact
    elif expname == 'scat':    
        mycache[3] = _renorm1 (catso ,incache[3]) *mfact
        mycache[5] = _renorm1 (cats3 ,incacheoth) *mfact
    elif expname == 'smx1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FMSK =  calccats( F_OW*F_OT,'noextr')
        mycache[3] = _renorm1 (catso *FMSK ,incache[3]) *mfact
        mycache[5] = _renorm1 (cats3 *FMSK ,incache[3]) *mfact  
    elif expname == 'fmx1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FMSK =  calccats( F_OW*F_OT,'noextr')
        mycache[3] = _renorm1 (catso *FMSK ,incache[3]) *mfact
        mycache[5] = _renorm1 (cats3 *FMSK ,incache[3]) *mfact  
    else:
        print("Error: type not defined : " +expname)
        raise()
#gebuik de parameters       
    ogrid.write(mycache[3],3)
    ogrid.write(mycache[5],5)
    ogrid.close()
    ogrid= rasterio.open(fname)
    return ([ogrid, fname, mycache ])

oset03, fname, mycache=writeexperiment('fmx1',rudifungcache,10,2500,'e0903a') 
oset02, fname, mycache=writeexperiment('swap',rudifungcache,10,2500,'e0903a') 
oset01, fname, mycache=writeexperiment('same',rudifungcache,10,2500,'e0903a') 

# -



def showaddhtn(dataset3):
    fig, ax = plt.subplots()
#   dataset3=dataset2.to_crs(epsg=plot_crs)
#    base=cbspc4data.boundary.plot(color='green',ax=ax,alpha=.3);
    rasterio.plot.show((dataset3,3),cmap='Reds',ax=ax,alpha=0.5)
    rasterio.plot.show((dataset3,5),cmap='Blues',ax=ax,alpha=0.5)
    setaxutr(ax)
#    cx.add_basemap(pland, source= prov0)
showaddhtn(oset01)    

oset04, fname, mycache=writeexperiment('icat',rudifungcache,10,2500,'e0903a') 
showaddhtn(oset04)  

for exp in expposs :
    oset04, fname, mycache=writeexperiment(exp,rudifungcache,10,2500,'e0903a') 
    print (showaddhtn(oset04)   )

showaddhtn(oset03)  


