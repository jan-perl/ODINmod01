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

import ODiN2readpkl

gemeentendata ,  wijkgrensdata ,    buurtendata = ODiN2readpkl.getgwb(2020)

grgem = gemeentendata[(gemeentendata['H2O']=='NEE') & (gemeentendata['AANT_INW']>1e5) ]

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

pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid= rasterio.open(pc4tifname)

#rudifunset, heb originele data niet nodig, alleen grid
#Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
#Rf_net_buurt.reset_index(inplace=True,drop=True)
#gemaakt in ROfietsbalans2
rudifuntifname=calcgdir+'/oriTN2-NL.tif'
rudifungrid= rasterio.open(rudifuntifname)


def setaxreg(ax,reg):
    if reg=='htn':
        ax.set_xlim(left=137000, right=143000)
        ax.set_ylim(bottom=444000, top=452000)
    elif reg=='utr':    
        ax.set_xlim(left=113000, right=180000)
        ax.set_ylim(bottom=480000, top=430000)


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
        pcatted = np.where(np.isin( pcatted ,[0,1,2*nbin-1]) ,0,pcatted-1)
    elif flgs=='midone':
        pcatted = np.where(np.isin( pcatted ,[0,1,2*nbin-1]) ,0,1)  
    elif flgs=='pres':
        pcatted = np.where(pcatted>=2 ,0,1)          
    elif flgs=='dfcat':
        pcatted =  pcatted 
    elif flgs=='vals':
        pcatted =  invs     
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

#flags
expposs1= ['base' ,'same', 'swap','verd' ,'icat' ,'scat' ]
expposs2= ['fmx1','smx1','frb1','srb1','xfm1','xfb1','atm1','stm1','snm1']
exprrun= expposs1+expposs2
#exprrun = ['snm1','stm1']
exprdists= [2500]
#exprdists= [2500,3700,5000]

# +
def _renorm1(pat,tot):
    return np.sum(tot)* pat / (np.sum(pat))


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
    fmp=0.25
    
    if expname == 'same':
        mycache[3] = incache[3]*mfact
        mycache[5] = incacheoth*mfact
    elif expname == 'verd':    
        #werk meer ophogen ivm macht voor wonen en selectie dat die hier gebruikt worden
        mycache[3] = _renorm1 (incache[3]* incache[3] ,incache[3]) *mfact
        mycache[5] = _renorm1 (incacheoth*incacheoth ,incacheoth) *mfact *1.2
    elif expname == 'swap':
#        grw= ( incacheoth*mfact *np.sum(incache[3])/np.sum(incacheoth) )
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
    elif expname == 'srb1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FMSK =  calccats( np.power(F_OW*F_OT,fmp),'noextr')
        cb= (cats3-catso)
        mycache[3] = _renorm1 (np.where(cb<0,-cb,0 ) *FMSK ,incache[3]) *mfact
        mycache[5] = _renorm1 (np.where(cb>0,cb,0 )*FMSK ,incacheoth) *mfact  
    elif expname == 'frb1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FMSK =  calccats( np.power(F_OW*F_OT,fmp),'noextr')
        cb= (catso-cats3)
        mycache[3] = _renorm1 (np.where(cb<0,-cb,0 ) *FMSK ,incache[3]) *mfact
        mycache[5] = _renorm1 (np.where(cb>0,cb,0 )*FMSK ,incacheoth) *mfact  
    elif expname == 'xfm1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        OW_MSK =  calccats( F_OW,'pres')
        #transformeer gebouw-> woning waar niet-woningen minder zijn dan woningen 
        #en bouw rest waar nu ook gebouwen zijn
        mycache[3] = _renorm1 (catso * (cats3>catso)  ,incache[3]) *mfact
        mycache[5] = _renorm1 (catso ,incacheoth) *2* mfact  -  _renorm1 (mycache[3],incacheoth) *mfact
    elif expname == 'xfb1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        OW_MSK =  calccats( F_OW,'noextr')
        #transformeer  woning -> gebouw waar niet-woningen minder zijn dan woningen 
        #en bouw wonigen waar woningen nu in minderheid zijn
        mycache[5] = _renorm1 ( (cats3) * (catso<4) * (catso<cats3) ,incacheoth) *mfact        
        mycache[3] = _renorm1 ( catso *  OW_MSK ,incache[3]) *2 *mfact - _renorm1 (mycache[5],incache[3])*mfact
    elif expname == 'smx1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FMSK =  calccats( np.power(F_OW*F_OT,fmp),'noextr')
        mycache[3] = _renorm1 (catso *FMSK ,incache[3]) *mfact
        mycache[5] = _renorm1 (cats3 *FMSK ,incacheoth) *mfact  
    elif expname == 'fmx1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FMSK =  calccats( np.power(F_OW*F_OT,fmp),'noextr')
        mycache[3] = _renorm1 (cats3 *FMSK ,incache[3]) *mfact
        mycache[5] = _renorm1 (catso *FMSK ,incacheoth) *mfact  
    elif expname == 'atm1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        mycache[3] = _renorm1 (cats3 *F_OW*F_OT ,incache[3]) *mfact
        mycache[5] = _renorm1 (catso *F_OW*F_OT ,incacheoth) *mfact      
    elif expname == 'stm1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        mycache[3] = _renorm1 (catso *F_OW*F_OT ,incache[3]) *mfact
        mycache[5] = _renorm1 (cats3 *F_OW*F_OT ,incacheoth) *mfact
    elif expname == 'snm1':
        filt=rasteruts1.roundfilt(100,mtrrang)
        F_OW = rasteruts1.convfiets2d(incache[3], filt ,bdim=8)
        F_OT = rasteruts1.convfiets2d(incacheoth, filt ,bdim=8)
        FM1 = F_OW * (np.sum(F_OW+F_OT)) / np.sum (F_OW) / (F_OW+F_OT)
        mycache[3] = _renorm1 (catso *F_OW*F_OT * (FM1 <0.90) ,incache[3]) *mfact
        mycache[5] = _renorm1 (cats3 *F_OW*F_OT * (FM1 >0.45) ,incacheoth) *mfact                
    else:
        print("Error: type not defined : " +expname)
        raise()
#gebuik de parameters       
    ogrid.write(mycache[3],3)
    ogrid.write(mycache[5],5)
    ogrid.close()
    ogrid= rasterio.open(fname)
    return ([ogrid, outsetnm, mycache ])

oset03, fname03, mycache03=writeexperiment('fmx1',rudifungcache,10,2500,'tst') 
# -
oset03, fname03, mycache03=writeexperiment('atm1',rudifungcache,10,2500,'txt') 


def showaddutr(dataset3):
    fig, ax = plt.subplots()
#   dataset3=dataset2.to_crs(epsg=plot_crs)
#    base=grgem.boundary.plot(color='green',ax=ax,alpha=.2);
    base=gemeentendata.boundary.plot(color='green',ax=ax,alpha=.2);
    
    rasterio.plot.show((dataset3,3),cmap='Reds',ax=ax,alpha=0.1)
    rasterio.plot.show((dataset3,5),cmap='Blues',ax=ax,alpha=0.5)
    setaxreg(ax,'utr')
#    cx.add_basemap(pland, source= prov0)
showaddutr(oset03)    

oset04l, fnamel, mycachel=writeexperiment('atm1',rudifungcache,10,2500,'tst0904b') 


def showlogs(dataset3):
    fig, ax = plt.subplots()
#   dataset3=dataset2.to_crs(epsg=plot_crs)
    base=grgem.boundary.plot(color='green',ax=ax,alpha=.2);
    rasterio.plot.show((dataset3,3),cmap='Reds',ax=ax,alpha=0.1)
    rasterio.plot.show((dataset3,5),cmap='Blues',ax=ax,alpha=0.5)
    setaxreg(ax,'utr')
#    cx.add_basemap(pland, source= prov0)
showlogs(oset03) 

showaddutr(oset03)

# +
nlextent=[0,280000,300000, 625000]
def logpltland(ecache,fld,selextent,fname,txt):
    minv=1
    mos=np.log(minv)/ np.log(10)
    image1= np.log(np.where(ecache[3]<minv,1,ecache[3]/minv)) /np.log(10)
    image2= np.log(np.where(ecache[5]<minv,1,ecache[5]/minv)) /np.log(10)
    nv = - ecache[3] - ecache[5]
    image3= np.log(np.where(nv<minv,1,nv/minv)) /np.log(10)
    lststr =  'Values for {}, min1 log10 W {}, max log10 W {}, min O {} , max log10 O {}  , max log10 negs {}'. format (txt, np
                    .min(image1)-mos,np.max(image1)-mos , np.min(image2)-mos,np.max(image2)-mos,np.max(image3)-mos)
    print (lststr)
    fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(50, 20))
    #image = np.isnan(image)
    ax1.imshow(image1,cmap='jet',alpha=.6,extent=selextent)
    ax2.imshow(image2,cmap='jet',alpha=.6,extent=selextent)
    ax3.imshow(image1,cmap='Reds',alpha=.6,extent=selextent)
    ax3.imshow(image2,cmap='Blues',alpha=.6,extent=selextent)
    ax3.imshow(image3,cmap='Greens',alpha=.6,extent=selextent)
    grgem.boundary.plot(color='green',ax=ax3,alpha=.2)
    if (selextent[0] == 113000):
        setaxreg(ax3,'utr')
        ax3.invert_yaxis()
#   ax1.colorbar()
#    ax2.colorbar()
#    axes[0].plot(x1, y1)
#    axes[1].plot(x2, y2)
    ax2.set_title(lststr)
    fig.tight_layout()
    figname = "../intermediate/addgrds/fig_"+fname+'.png';
    fig.savefig(figname) 
    return fig
    
logpltland(mycache03,3,nlextent,'tstexample','tstexample')
# -

utrextent=[113000,180000,430000,480000 ]
def mkloccach(ecache,selextent,oriextent):
    oridim = ecache[3].shape
    xmul= oridim[1]/(oriextent[1]-oriextent[0])
    ymul= oridim[0]/(oriextent[2]-oriextent[3])
    #print([oridim,oriextent,xmul,ymul])
    
    xmin=int((selextent[0]-oriextent[0])*xmul)
    xmax=int((selextent[1]-oriextent[0])*xmul)
    ymin=int((selextent[2]-oriextent[3])*ymul)
    ymax=int((selextent[3]-oriextent[3])*ymul)
    #print([xmin,xmax,ymin,ymax])
    ocache=dict()
    for imgidx in [3,5]:
        #print(ecache[imgidx].shape)
        sli= ecache[imgidx][ymax:ymin,xmin:xmax]
        odim = sli.shape
        #print(odim)
        ocache[imgidx]= sli
    return ocache
utrcache03=mkloccach(mycache03,utrextent,nlextent)
logpltland(utrcache03,3,utrextent,'tstexampleut','tstexampleut')    

expcach=dict()
gset=dict()
fnamec=dict()
for exp in exprrun :
    for dist in exprdists:
        gset[exp], fnamec[exp], expcach[exp]=writeexperiment(exp,rudifungcache,10,dist,'e1121a') 
        logpltland(expcach[exp],3,nlextent,fnamec[exp],exp)
        utrcache03=mkloccach(expcach[exp],utrextent,nlextent)
        logpltland(utrcache03,3,utrextent,fnamec[exp]+'_utr',exp+'_utr')   
#    print (showaddhtn(oset04)   )

#expposs2
for exp in [expposs2]  :
    print(exp)
    print (showaddutr(gset[exp] )) 

#expposs2 
for exp in [] :
    print( logpltland(expcach[exp],3,nlextent,fnamec[exp],exp) )


