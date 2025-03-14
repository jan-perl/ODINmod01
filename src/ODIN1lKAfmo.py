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


def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
pc4inwgcache = getcachedgrids(pc4inwgrid)
rudifungcache = getcachedgrids(rudifungrid)

# nu nog MXI overzetten naar PC4 ter referentie




# +
#nu ODIN ranges opzetten
#we veranderen NIETS aan odin data
#wel presenteren we het steeds als cumulatieve sommen tot een bepaalde bin
# -

useKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
xlatKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <20].copy()
maxcuse= np.max(useKAfstV[useKAfstV ["MaxAfst"] !=0] ['KAfstCluCode'])
xlatKAfstV  = xlatKAfstVa [(xlatKAfstVa['KAfstCluCode']<=maxcuse ) |
                           (xlatKAfstVa['KAfstCluCode']==np.max(useKAfstV[ 'KAfstCluCode']) )].copy()
#print(xlatKAfstV)   
print(useKAfstV)   

useKAfstVQ  = useKAfstV [useKAfstV ["MaxAfst"] <4]
#print(xlatKAfstV)   
print(useKAfstVQ)   

# +
#import ODiN2pd
#import ODiN2readpkl
# -

usePC4MXI=True


def mkfietswijk3pc4(pc4data,pc4grid,rudigrid):
    #pc4lst=pc4grid.read(1)
    pc4lst=pc4grid[1]
    outdf=pc4data[['postcode4','aantal_inwoners']].rename(columns={'postcode4':'PC4'} )
#    outdf['aantal_inwoners_gr2'] = rasteruts1.sumpixarea(pc4lst,pc4grid.read(3) )
#    outdf['S_MXI22_BWN'] = rasteruts1.sumpixarea(pc4lst,rudifungrid.read(3) )
#    outdf['S_MXI22_BAT'] = rasteruts1.sumpixarea(pc4lst,rudifungrid.read(5) )
    outdf['aantal_inwoners_gr2'] = rasteruts1.sumpixarea(pc4lst,pc4grid[3] )
    outdf['S_MXI22_BWN'] = rasteruts1.sumpixarea(pc4lst,rudigrid[3] )
    outdf['S_MXI22_BAT'] = rasteruts1.sumpixarea(pc4lst,rudigrid[5] )
    outdf['S_MXI22_BAN'] = outdf['S_MXI22_BWN'] - outdf['S_MXI22_BAT'] 
    if usePC4MXI:
        outdf['S_MXI22_NS'] = outdf['S_MXI22_BWN']  / (outdf['S_MXI22_BWN']  + outdf['S_MXI22_BAN'] )
        outdf['S_MXI22_BB'] = outdf['S_MXI22_NS']

    outdf['S_MXI22_BG'] = outdf['S_MXI22_BWN'] / pc4data['oppervlak']        
    outdf['S_MXI22_GB'] = pd.qcut(outdf['S_MXI22_BB'], 10)
    outdf['S_MXI22_GG'] = pd.qcut(outdf['S_MXI22_BG'], 10)
    outdf['aantal_inwoners_d2'] = outdf['aantal_inwoners_gr2'] -outdf['aantal_inwoners']
    return outdf
#fietswijk3pc4=mkfietswijk3pc4(cbspc4data,pc4inwgrid,rudifungrid)
fietswijk3pc4=mkfietswijk3pc4(cbspc4data,pc4inwgcache,rudifungcache)
bd=fietswijk3pc4 [abs(fietswijk3pc4['aantal_inwoners_d2'] ) > 1 ]

expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0}


# +
#eerst een dataframe dat
#2) per lengteschaal, en PC variabelen eerst geografisch sommeert dan waarde ophaalt per punt
#   bijv BAM BAT of  (BAM*BAT)/(BAM+BAT)
#en dit dan per PC 4 sommeert
#dit zijn min of meer statische utigangstabellen waarop de modellen dan door kunnen gaan
#Hierna kan daar dan op gemodelleerd worden (data is niet geo meer maar per PC4)
#let op dit beschrijft het NAAR deel van de reis: het van deel 
#dit moet nog vermenigvuligd worden met het vergelijkbare (heel locale) tuple ->
#dus steeds 9 kolommen LW LO LM x OW OO OM 
#radius =0: some over hele land voor O
def filtgridprecalc (rudigrid,myKAfstV,pu):
    debug=False
    
    R=dict()
#    R_LW= rudifungrid.read(3)
#    R_LT= rudifungrid.read(5)
    R_LW= rudigrid[3]
    R_LT= rudigrid[5]
    R_LO =  R_LT- R_LW   
    R_LW = np.power(R_LW,pu['LW'])
    R_LO = np.where(R_LO <0,0, np.power(R_LO,pu['LO']) )
    R['LW']= R_LW
    R['LO'] =  R_LO
#    R['LM'] =  (R_LO* R_LW) / (R_LO + R_LW+1e-10)

    for lkey in R.keys():

        
        for okey in ('OW','OO','OM'):
            colnam="M_"+ lkey +"_" + okey
#            print(colnam)
            outdf[colnam] = 0*lvals            
    R_LW_land= np.sum(R_LW)
    R_LO_land= np.sum(R_LO)
    
    for index, row in myKAfstV[myKAfstV['MaxAfst']!=0].iterrows():        
        filt=rasteruts1.roundfilt(100,1000*row["MaxAfst"])
        filtarea = np.power(np.sum(filt)*1e-6,pu['OA'])

        F=dict()
        tstart= time.perf_counter()
        F_OW = rasteruts1.convfiets2d(R_LW, filt ,bdim=8) /R_LW_land
        F_OT = rasteruts1.convfiets2d(R_LT, filt ,bdim=8) /R_LO_land
        tend= time.perf_counter()
    if debug:
        print(("blklen" ,len(outdfst), "outlen" ,len(outdf)) )
    return(outdf)

#geoschpc4allQ=mkgeoschparafr(cbspc4data,pc4inwgrid,rudifungrid,useKAfstVQ,1.2,1.0)
#precgrids1=filtgridprecalc(rudifungcache,useKAfstVQ,expdefs)

# +
#eerst een dataframe dat
#2) per lengteschaal, en PC variabelen eerst geografisch sommeert dan waarde ophaalt per punt
#   bijv BAM BAT of  (BAM*BAT)/(BAM+BAT)
#en dit dan per PC 4 sommeert
#dit zijn min of meer statische utigangstabellen waarop de modellen dan door kunnen gaan
#Hierna kan daar dan op gemodelleerd worden (data is niet geo meer maar per PC4)
#let op dit beschrijft het NAAR deel van de reis: het van deel 
#dit moet nog vermenigvuligd worden met het vergelijkbare (heel locale) tuple ->
#dus steeds 9 kolommen LW LO LM x OW OO OM 
#radius =0: some over hele land voor O
def mkgeoschparafr (pc4data,pc4grid,rudigrid,myKAfstV,pu):
    debug=False
    #pc4lst=pc4grid.read(1)
    pc4lst=pc4grid[1]
    outdf=pc4data[['postcode4','aantal_inwoners','oppervlak']].rename(columns={'postcode4':'PC4'} )
    outdf['KAfstCluCode'] = np.max(myKAfstV["KAfstCluCode"])
    outdf['MaxAfst'] = 0
    outdfst= outdf.copy()
    
    R=dict()
#    R_LW= rudifungrid.read(3)
#    R_LT= rudifungrid.read(5)
    R_LW= rudigrid[3]
    R_LT= rudigrid[5]
    R_LO =  R_LT- R_LW   
    R_LW = np.power(R_LW,pu['LW'])
    R_LO = np.where(R_LO <0,-np.power(-R_LO,pu['LO']), np.power(R_LO,pu['LO']) )
    R['LW']= R_LW
    R['LO'] =  R_LO
#    R['LM'] =  (R_LO* R_LW) / (R_LO + R_LW+1e-10)

    for lkey in R.keys():
        lvals = rasteruts1.sumpixarea(pc4lst,R[lkey])
        colnam="M_"+ lkey +"_AL"
        outdf[colnam] = lvals  
        outdfst[colnam] = lvals 
        okey = "OA"
        colnam="M_"+ lkey +"_" + okey
        outdf[colnam] = 0
#                print(colnam,(lvals[np.isnan(lvals)]))

        
        for okey in ('OW','OO','OM'):
            colnam="M_"+ lkey +"_" + okey
#            print(colnam)
            outdf[colnam] = 0*lvals            
    R_LW_land= np.sum(R_LW)
    R_LO_land= np.sum(R_LO)
    
    for index, row in myKAfstV[myKAfstV['MaxAfst']!=0].iterrows():        
        outdfadd=outdfst.copy()
        outdfadd['KAfstCluCode']= row["KAfstCluCode"]
        outdfadd['MaxAfst'] = row["MaxAfst"]
#        print(row["KAfstCluCode"], row["MaxAfst"])
        filt=rasteruts1.roundfilt(100,1000*row["MaxAfst"])
        filtarea = np.power(np.sum(filt)*1e-6,pu['OA'])

        F=dict()
        tstart= time.perf_counter()
        F_OW = rasteruts1.convfiets2d(R_LW, filt ,bdim=8) /R_LW_land
        F_OT = rasteruts1.convfiets2d(R_LT, filt ,bdim=8) /R_LO_land
        tend= time.perf_counter()
        F['OW'] =  F_OW  
        F_OO =  F_OT- F_OW        
        F['OO'] =  F_OO 
        F['OM'] =  (F_OO* F_OW) / (F_OO + F_OW+1e-10)
        print ((row["KAfstCluCode"], row["MaxAfst"], filt.shape , tend-tstart, " seconds") ) 
        
        for lkey in R.keys():
            outdfadd["M_"+ lkey +"_" + "OA" ] = outdfadd["M_"+ lkey +"_" + "AL" ] * filtarea
            for okey in ('OW','OO','OM'):
                lvals = rasteruts1.sumpixarea(pc4lst,np.multiply(R[lkey],F[okey]))
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outdfadd[colnam] = lvals            
        
        outdf=outdf.append(outdfadd)
    if debug:
        print(("blklen" ,len(outdfst), "outlen" ,len(outdf)) )
    return(outdf)

#geoschpc4allQ=mkgeoschparafr(cbspc4data,pc4inwgrid,rudifungrid,useKAfstVQ,1.2,1.0)
geoschpc4allQ=mkgeoschparafr(cbspc4data,pc4inwgcache,rudifungcache,useKAfstVQ,expdefs)
#en inspecteer voorbeeld
geoschpc4allQ[geoschpc4allQ['PC4']==3991]
# -

geoschpc4all=mkgeoschparafr(cbspc4data,pc4inwgcache,rudifungcache,useKAfstV,expdefs)


# +
def _qcutmxi(df,mxivarin1,mxivarin2,mxiout,nbins):
    mxiout,binsmxi =  pd.qcut(df[mxivarin1], nbins, retbins=True, labels=False)
    normout,binsnorm =  pd.qcut(df[mxivarin2], 2, retbins=True, labels=False)    
    df['mxiout'] = mxiout + nbins*normout
    print (binsmxi , binsnorm)  
#    rv= pd.Series(mxiout*1.1, index=range(len(df)))
    return  df[['mxiout']]

#add df into mxi bins, retulting in outdf with mxi in addition to PC4
#returned binfr is not used, provided for debugging

def addmxibins (df,nbins):
    outdf=df.set_index(keys=['KAfstCluCode','PC4'],drop=False)
#    print(outdf)
    outdf['normMXI']= np.where(outdf['MaxAfst'] ==0, 
               (outdf['M_LW_AL'] + outdf['M_LO_AL'] ) , 
       (outdf['M_LW_OW'] + outdf['M_LW_OO'] + outdf['M_LO_OW'] + outdf['M_LO_OO'] ) ) 
    outdf['scaleMXI']= np.where(outdf['MaxAfst'] ==0, outdf['M_LW_AL'], 
                                  (outdf['M_LW_OW'] + outdf['M_LO_OW'] ) ) /outdf['normMXI']
#    outdf['mxigrp'] =4
    binfr=1
    allret= outdf.drop(columns=['PC4','KAfstCluCode']).groupby('KAfstCluCode').apply(_qcutmxi, 'scaleMXI','normMXI',
                                                                                     'mxigrp',nbins)
    print(allret.dtypes)
    outdf['mxigrp'] =allret
    outdf=outdf.reset_index(drop=True)
    return ([outdf,binfr])
nmxibins_glb=7
geoschpc4, geobingr = addmxibins(geoschpc4all,nmxibins_glb)
# -

geoschpc4

#kijk even naar sommen
geoschpc4.groupby(['KAfstCluCode','MaxAfst']).agg('sum')

geoschpc4

useKAfstVland = useKAfstV [useKAfstV['MaxAfst']==0]
geoschpc4land=mkgeoschparafr(cbspc4data,pc4inwgcache,rudifungcache,useKAfstVland,expdefs)
geoschpc4land



from importlib import reload  # Python 3.4+
if False:
        foo = reload(ODINcatVNuse)

#het inlezen van odinverplgr loopt in deze versie via ODINcatVNuse
import ODINcatVNuse

print(useKAfstV) 
maskKAfstV= list(useKAfstV['KAfstCluCode'])
maskKAfstV

# +
#odinverplklinfo = ODINcatVNuse.odinverplklinfo_o[np.isin(ODINcatVNuse.odinverplklinfo_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
#odinverplgr =ODINcatVNuse.odinverplgr_o[np.isin(ODINcatVNuse.odinverplgr_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
#odinverplflgs =ODINcatVNuse.odinverplflgs_o[np.isin(ODINcatVNuse.odinverplflgs_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)

MainUseSelFactorV='FactorVGen'
odinverplgr= ODINcatVNuse.deffactorv(ODINcatVNuse.odinverplgr_o,maskKAfstV,MainUseSelFactorV )
odinverplklinfo = ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplklinfo_o,maskKAfstV,MainUseSelFactorV)
odinverplflgs =ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplflgs_o,maskKAfstV,MainUseSelFactorV)


# +
#de ingelezen odinverplgr_o heeft nog heel veel KAfstCluCode s
#hierin eerst snijden
# -

#was odinverplgr=pd.read_pickle("../intermediate/ODINcatVN01db.pkl")
def findskippc(rv):
    rv['FactorV'] = np.where ((rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ,
               0,rv['FactorVGen'] + 0* rv['FactorVSpec'] )
    skipsdf = rv [(rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ] [['PC4','MotiefV']].copy(deep=False)
    return skipsdf
skipPCMdf = findskippc(odinverplgr)
skipPCMdf

# vind spec fractie
s1= ODINcatVNuse.odinverplgr [ODINcatVNuse.odinverplgr['KAfstCluCode']==15].sum()
sf = s1['FactorVSpec'] / (s1['FactorVSpec']  + s1['FactorVGen'] )
sf*100


# +
#maak 2 kolommen met totalen aankomst en vertrek (alle categorrieen)
#worden als kolommen aan addf geoschpc4land toegevoegd per PC
#is eigenlijk alleen van belang voor diagnostische plots
#en totalen hannen ook uit aggregaten gehaald kunnen worden
#TODO cleanup

def mkvannaarcol(rv,verpldf,xvarPC):
    dfvrecs = verpldf [(verpldf ['GeoInd'] == xvarPC) ]
    pstats = dfvrecs.groupby('PC4')[['FactorV']].sum().reset_index()
    outvar='TotaalV'+xvarPC
#    pstats=pstats.rename(columns={xvarPC:'PC4'})
#    print(pstats)
    addfj = rv.merge(pstats,how='left')
    rv[outvar] = addfj['FactorV']
    print(len(pstats))
    
mkvannaarcol(geoschpc4land,odinverplgr,'AankPC')
mkvannaarcol(geoschpc4land,odinverplgr,'VertPC')

# +
#geoschpc4 is een mooi dataframe met generieke woon en werk parameters
#Er is nog wel een mogelijk verzadigings effect daar waar de waarden voor
#grotere afstanden die van de landelijke waarden benaderen
#geoschpc4land
# -
#nog netjes importeren
largranval = -9999999999
ODINmissint = -99997 


#nu een kijken:
#inw / geo opp , #inw / wo  wo / geo opp
#dit is voor buurten al gedaan in viewCBS
#hergebruik code
print(largranval)
ODINmissint = -99997 


# +
def loglogregrplot(indfo,xcol,ycol,savtag,xtit):
    print ((np.min(indfo[xcol] ) ,np.max(indfo[xcol] ),np.min(indfo[ycol]),np.max(indfo[ycol]))) 
    
    indf = indfo[ (indfo [xcol] >0)  & (indfo [ycol] >0)].copy()
    print(( len(indfo) ,len(indf)) )
    lm = linear_model.LinearRegression(fit_intercept=True)
    model = lm.fit(np.log(indf[xcol].values.reshape(-1, 1)),
                   np.log(indf[ycol].values.reshape(-1, 1))) 
    #lm.fit(indf[xcol],indf[ycol]) 
    indf['Predict']=np.exp(model.predict(np.log(indf[xcol].values.reshape(-1, 1))))
    indf['Predictu']=indf['Predict']*np.exp(1)
    indf['Predictl']=indf['Predict']*np.exp(-1)
    print( model.coef_, model.intercept_)
    fig, ax = plt.subplots(figsize=(6, 4))
    chart=seaborn.lineplot(data=indf,x=xcol,y='Predict', color='g',ax=ax)
    seaborn.lineplot(data=indf,x=xcol,y='Predictu', color='r',ax=ax)
    seaborn.lineplot(data=indf,x=xcol,y='Predictl', color='r',ax=ax)
    seaborn.scatterplot(data=indf,x=xcol,y=ycol,ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.suptitle("Centrale lijn macht %.2f  asafsnede %.2g"% (model.coef_[0][0],np.exp(model.intercept_)) )
    ax.set_xlabel(xtit)
#    chart.set_ylabels(ylab)
    figname = "../output/gplo_reg_"+savtag+"_"+'G1.svg';
    fig.savefig(figname, bbox_inches="tight") 
    print(fig)
    return model

#een replicatie is genoeg en verwijder NAs
geoschpc4r1=geoschpc4land[(  ~ np.isnan(geoschpc4land['M_LW_AL'])) & 
                     ( ~ np.isnan(geoschpc4land['aantal_inwoners'])) &
                     ( geoschpc4land['aantal_inwoners'] != ODINmissint )]

#geoschpc4r2= cbspc4data[['postcode4','oppervlak'] ] .merge (geoschpc4r1 ,left_on=('postcode4'), right_on = ('PC4') )
geoschpc4r2 = geoschpc4r1

buurtlogmod= loglogregrplot(geoschpc4r2,'M_LW_AL','aantal_inwoners' ,'M_LW_AL_PC4','Woon oppervlak (m2) per PC4')

# +
#enigszine onverwacht komnt hier dezelfde relatie uit al in wijken en buurten
#wat dit betreft lijken postcodes dus homogeen
# -

buurtlogmod= loglogregrplot(geoschpc4r2,'oppervlak','aantal_inwoners','OPPINW_PC4','Gebied oppervlak (are) per PC4')

buurtlogmod= loglogregrplot(geoschpc4r2,'oppervlak','M_LW_AL' ,'OPPBEB_PC4','Gebied oppervlak (are) per PC4')

# +
#Poging2: uit behouwings ratio
geoschpc4r2['m2perinw']= geoschpc4r2['M_LW_AL']/geoschpc4r2['aantal_inwoners' ]
geoschpc4r2['inwperare']= 10000*geoschpc4r2['aantal_inwoners' ]/ geoschpc4r2['M_LW_AL']
geoschpc4r2['pctow']= geoschpc4r2['M_LW_AL']/geoschpc4r2['oppervlak' ]
buurtlogmod= loglogregrplot(geoschpc4r2,'pctow','inwperare' ,'RELINW_PC4','Woonoppervlak fractie (m2/are) per PC4')

#deze ratio zouden we ook op buurtniveau kunnen berekenen
# -
geogeenwerk = geoschpc4r2 [geoschpc4r2 ['M_LW_AL'] > 20 * geoschpc4r2 ['M_LO_AL']] 
geogeenwerk


alleenwoonexp=loglogregrplot(geogeenwerk,'M_LW_AL','TotaalVAankPC' ,'MOBV_PC4','Woonoppervlak (m2) per PC4')


# +
def laagwoonexp(model,LW_OWin):
    rv = np.exp(model.coef_[0]*np.log(LW_OWin) + model.intercept_ )
    return rv
    
#print( np.array ([ laagwoonexp(alleenwoonexp,geogeenwerk['M_LW_OW']),geogeenwerk['TotaalVAankPC'] ] ).T )


# -

geoschpc4r2 ['VminWAankPC']=  geoschpc4r2 ['TotaalVAankPC'] -laagwoonexp(alleenwoonexp,geoschpc4r2 ['M_LW_AL'])
geogeenwoon = geoschpc4r2 [(geoschpc4r2 ['M_LW_AL'] *5 < geoschpc4r2 ['M_LO_AL'] ) & (geoschpc4r2 ['VminWAankPC'] >0) ] 
#geogeenwoon

geenwoonexpC= loglogregrplot(geogeenwoon,'M_LO_AL','VminWAankPC','MOBGEWO_VminW','Woonoppervlak (m2) per PC4')
geenwoonexpA= loglogregrplot(geogeenwoon,'M_LO_AL','TotaalVAankPC','MOBGEWO_TotV','Woonoppervlak (m2) per PC4')


# +
#nu eerst geo data voor fit naar geo groepen verzamelen:
# let op: er zijn daarna 2 schalen, die ieder nuttig zijn:
# PC4 (om lokale zaken te bekijken) en geogroep voor statistiek om et fitten
# let op PC4 naar mxigrp vertaling is verschillend per 'KAfstCluCode'
# er komt hierna correctie voor records waar DEZE postcode  niet mee doet
# er wordt ook gesommeerd over postcodes waar geen ombservaties zijn
# daarom zal som van de output van deze routine alrijd groter zijn
# dan wanneer alleen records worden meegeteld waar ook observatie is
# Let Op: deze sommeert (via integraal) ook over records waar de ANDERE
# wijk in geblokte, of niet waargenomen postcode zit

def summmxigrp(dfin):
    dfgrp= dfin.copy(deep=False)
    dfgrp['nPCsgeo']=1
    newgrp= ['KAfstCluCode','mxigrp']
    rv= dfgrp.groupby(newgrp).agg('sum')
#    rv['MaxAfst'] = rv['MaxAfst'] /rv['nPCsgeo']
#    rv['scaleMXI'] = rv['scaleMXI'] /rv['nPCsgeo']
    rv=rv.reset_index().drop(columns=['MaxAfst','PC4'] )
    return rv
#.drop(columns='PC4')
geoschmxigrp= summmxigrp(geoschpc4)
debugmxisum = True
if debugmxisum:
    print( len(geoschmxigrp)) 
    print (geoschmxigrp.sum().T) 
    #geoschmxigrp.dtypes


# +
def _partsummmxigrp(thismot,geoschpc4):
    pcset = list(thismot['PC4'])
#    print(pcset)
    pc4sel = geoschpc4[np.isin(geoschpc4['PC4'],pcset)]
#    print (pc4sel)
    dfo = summmxigrp(pc4sel)
    return dfo

def summmxicorrgrp(dfin,skips):
    rv = skips.groupby('MotiefV').apply(_partsummmxigrp,dfin).reset_index().drop(columns='level_1')
#    print (rv)
    return (rv)
geoschmxicorr= summmxicorrgrp(geoschpc4,skipPCMdf)
#een correctitie

def showM_gxmxicontrib(dfparts,dfall):
    print (len(geoschmxicorr) )
    #dfparts.dtypes
    gs1=dfparts.groupby('MotiefV').sum()
    gs2= dfall.sum()
    gsr=100*gs1/gs2
    print ("percentage M_ geblokt ")
    print (gsr) 
    
if debugmxisum:
    showM_gxmxicontrib(geoschmxicorr,geoschmxigrp)    


# +
def _addsummmxigrp(thismot,dfin1,corrdfin):    
    imatch = ['KAfstCluCode','mxigrp']
    motcorr = corrdfin[corrdfin['MotiefV']==thismot] #.set_index(imatch)
    motcorridx = dfin1[imatch] #.set_index(imatch)
    mmatch = ['MotiefV'] + imatch
    corrdf = motcorridx.merge(motcorr,how='left') .set_index(mmatch)
#    corrdf = pd.DataFrame(np.where(np.isnan(corrdf ),0,corrdf))
    corrdf = np.where(np.isnan(corrdf ),0,corrdf)
    dfinidxed = dfin1.copy(deep=False)
    dfinidxed['MotiefV'] =  thismot
    dfinidxed= dfinidxed.set_index(mmatch)
    addf= (dfinidxed - corrdf).reset_index()
    return addf

def allmotmxicorrgrp(dfin1,corrdfin,maxmotief):   
    rv = ( _addsummmxigrp(thismot+1,dfin1,corrdfin ) for thismot in range(np.int(maxmotief)) )
    rv=pd.concat(rv)
#    print([ np.sum(np.isnan(rv )),len(rv) ])

    #    print (rv)
    return (rv)
geoschmixpMotief= allmotmxicorrgrp(summmxigrp(geoschpc4),
                    summmxicorrgrp(geoschpc4,skipPCMdf),np.max(odinverplgr['MotiefV']))
#deze te gebruiken in plaata van geoschpc4 voor mixclusts
len(geoschmixpMotief)

if debugmxisum:
    showM_gxmxicontrib(geoschmixpMotief,geoschmxigrp)  
# -

#eerst kleine categorieen verwijderen uit odinverplgr
#niet zo doen: zo ontstaan dubbele records per PC3. Dus: verplaatsen naar ODINcatVN
grpexpcontrs= ['isnaarhuis','GeoInd','MotiefV','GrpExpl']
def replacesmallGrpExpl(dfin,landcod):    
    defcode=13
    numgrprecs=dfin[dfin['KAfstCluCode']==landcod].groupby(grpexpcontrs)[['GrpExpl']].agg('count').rename(
       columns={'GrpExpl':'NumExplRecs'}).reset_index()
    smallnum = numgrprecs [numgrprecs ['NumExplRecs'] <400].sort_values(by='NumExplRecs')
    expl13 =   numgrprecs [numgrprecs ['MotiefV'] ==defcode].rename(columns={'GrpExpl':'DefExpl'})
    print(  smallnum  )
    dfout=dfin.copy(deep=False)
    dfrepl = dfin.merge(smallnum,how='left').reset_index()
    dfrepl['toch'] = np.isnan(dfrepl['NumExplRecs']) ==False 
    dfout['MotiefV'] = np.where(dfrepl['toch'],defcode,dfout['MotiefV'] )
    newexpl = dfout.merge( expl13 , how='left').reset_index()
    print(newexpl)
    dfout['GrpExpl'] = np.where(dfrepl['toch'],newexpl['DefExpl'],dfout['GrpExpl']  )
    print(dfout.reset_index()[dfrepl['toch']])
    print(dfout.dtypes)
    return(dfout)
#odinverplgr2 = replacesmallGrpExpl (odinverplgr,ODINcatVNuse.landcod)
#odinverplgrtext = replacesmallGrpExpl (odinverplgr2,ODINcatVNuse.landcod)


# +
#nu ook mxigrp toevoegen aan odinverplgr
def odinmergemxi(odindf,mxigrps,cluflds):
#    print(mxigrps.dtypes)
    addclu=['KAfstCluCode','mxigrp']
    mxitab= mxigrps[['PC4']+addclu]
    itab= odindf.merge(mxitab,how='left')
    itab['NPcsF'] =1
    stab= itab.groupby(addclu+cluflds).agg('sum')  .reset_index().drop(columns=['PC4','index'])  
#    print(( len (odindf), len(rv)) )
    return stab

odinverplmxigr = odinmergemxi (odinverplgr ,geoschpc4,grpexpcontrs)
len(odinverplmxigr )
odinverplmxigr[(odinverplmxigr['KAfstCluCode'] ==15) & (odinverplmxigr['MotiefV'] ==1) ]
#deze te gebruiken in plaata vanodinverplgr voor mixclusts

# +
#print(allodinyr2)
#allodinyr = allodinyr2

# +
#todo: check waarom ifs niet werken
#todo: opmerken dat srcarr[0] niet meer van belang is bij gebruik odinverplgr

def addparscol (df,coltoadd):
    if "/" in coltoadd:
        srcarr=  coltoadd.split ('/')
#        print(srcarr)
        if srcarr[1] == "rudifun":
            PCcols=["AankPC","VertPC"]
            if 0 &  ~ (srcarr[0] in PCcols):
                print ("Error in addparscol: col ",srcarr[0]," not usable for ", coltoadd)
            elif 0 & ~ (srcarr[2] in list(fietswijk3pc4.columns)):
                print ("Error in addparscol: col ",srcarr[2]," not usable for ", coltoadd)
            else:
                fwin4 = fietswijk3pc4[['PC4',srcarr[2]]]
                df=df.merge(fwin4,on='PC4',how='left')
                df=df.rename(columns={srcarr[2]:coltoadd})
        else:
            print ("Error in addparscol: source ",srcarr[1]," not found for", coltoadd)
#    print ("addparscol: nrecords: ",len(df.index))
    return (df)
addparscol(odinverplgr ,"AankPC/rudifun/S_MXI22_BB").dtypes


# +
#dan een dataframe dat
#2) per lengteschaal, 1 PC (van of naar en anderegroepen (maar bijv ook Motief ODin data verzamelt)

def mkdfverplxypc4d1 (pstatsc,pltgrps,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4):
    debug=True
#    dfvrecs = df [(df['Verpl']==1 ) & (df[xvarPC] > 500)  ]   
    for pgrp in pltgrps:
        dfvrecs=addparscol(pstatsc,pltgrps)


#    dfgrps= useKAfstV   [  [ 'KAfstCluCode']] 
    #let op: right mergen: ALLE postcodes meenemen, en niet waargenomen op lage waarde zetten
#    vollandrecs = cartesian_product_multi (mygeoschpc4, dfgrps)
#    vollandrecs.columns= list(mygeoschpc4.columns) + list(dfgrps.columns)
    vollandrecs= mygeoschpc4
#    print(vollandrecs)
    if debug:
        print( ( "alle land combinaties", len(vollandrecs) , len(mygeoschpc4)) )
    #zonder volledige uitklap heeft dit weinig zin: 
    # vollandrevs moet dan alle fit (vannaar, motief , pcs en explfield invullen)
    # anders matcht er van alles niet, en dat is nog slechter dan nu
    pstatsc=pstatsc.merge(vollandrecs,how='left')
    if debug:
        print( ( "return rdf", len(pstatsc)) )
    
    return(pstatsc)

#code werk nog niet 

def mkdfverplxypc4 (dfg2,pltgrps,selstr,myKAfstV,myxlatKAfstV,geoschpc4in,pu):
    mygeoschpc4 = geoschpc4in
    if 1==0:
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            okey='OA'
            colnam="M_"+ lkey +"_" + okey
            mygeoschpc4[colnam] = mygeoschpc4[colnamAL] * (
                   np.power(mygeoschpc4['MaxAfst']*0.01,2*pu['OA']) )
#            okey='AO'
#            colnam="M_"+ lkey +"_" + okey
#            mygeoschpc4.drop(inplace=True,columns=colnam)
    rv= mkdfverplxypc4d1 (dfg2,pltgrps,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4)
    
    return rv

fitgrps=['MotiefV','isnaarhuis']
indatverplpc4gr = mkdfverplxypc4 (odinverplgr ,fitgrps,'Motief en isnaarhuis',
                                useKAfstV,xlatKAfstV,geoschpc4,2.0)
len(indatverplpc4gr)
# -
indatverplmxigr = mkdfverplxypc4 (odinverplmxigr ,fitgrps,'Motief en isnaarhuis',
                                useKAfstV,xlatKAfstV,geoschmixpMotief,2.0).merge(useKAfstV,how='left')
#MLlen(indatverplmxigr)

#M_LW_AL, M_LO_AL zou uiteindelijk voor landelijke schatting alleen afhankelijk moeten zijn van excluded PCs
#Voor vergelijking met de data moet je aannemen dat als een PC mist in de oorspronkelijke data
#het aantal reizen klein is, en dat dit hoort te passen bij de schatting
def showtotmot(df,metm):
    dfl = df[df['KAfstCluCode']==15].copy(deep=False)
    dfl['nrecdfl']=1
    shgr=['FactorV', 'nrecdfl']
    if metm:
        shgr = shgr + ['M_LW_AL','M_LO_AL']
    dflg= dfl.groupby(['MotiefV','GeoInd'])[shgr].agg('sum')
    return dflg.T
showtotmot(indatverplmxigr,True)    

indatverplmxigr [(indatverplmxigr ['FactorV']>0 ) ==False]

showtotmot(indatverplpc4gr,True)  

#alleen eerste kolom vergelijken. de M_ velden zitten niet in deze database
showtotmot(odinverplgr,False)


# +
#daarom verder met kolommen met een F_ (filtered)

#oude versie: ieder record fit naar ofwel ALsafe of naar osafe, of nergens heen

def choose_cutoffold(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    outframe=indat[[grpind,'GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    recisAL=indat['MaxAfst']==0
    wval1= indat[recisAL] [['FactorV',grpind,'GeoInd'] +pltgrps].copy(deep=False)
    wval1= wval1.rename(columns={'FactorV':'EstVPAL'})
    outframe=outframe.merge(wval1,how='left')    
    outframe['FactorVFAL'] = indat['FactorV']  /outframe['EstVPAL'] 
    outframe['FactorVFoCor'] =indat['FactorV'] 
    outframe['FactorVFo'] = indat['FactorV']  /outframe['EstVPAL']

    if hasfitted:
        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 500.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.3 * prevrres['FactorEstAL'] ) | (indat['FactorV'] < .3 *  prevrres['FactorEstAL'] )
        outframe['osafe']  = outframe['osafe']  & (outframe['MaxAfst']!=0)
        outframe['FactorVFo'] = indat['FactorV']  /prevrres['FactorEstAL']
#        outframe['FactorVFoCor'] =indat['FactorV']  * prevrres['FactorEstAL'] /outframe['EstVPAL'] 
#        outframe['ALsafe'] = indat['FactorV'] > .9 * outframe['EstVPAL'] 
#        outframe['osafe']  = (outframe['FactorVFoCor'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
    elif True:        
        outframe['ALsafe'] = indat['FactorV'] > 1.9 * outframe['EstVPAL'] 
        outframe['osafe'] =  indat['FactorV'] < .4 * outframe['EstVPAL'] 
        outframe['ALsafe'] = outframe['ALsafe'] | (outframe['MaxAfst']==0)
        outframe['osafe'] =  outframe['osafe'] & (outframe['MaxAfst']!=0) & (indat['FactorV'] > 0)
    else:
        outframe['ALsafe'] =True
        outframe['osafe'] = True
        outframe['FactorVFoCor'] =indat['FactorV'] 
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            for okey in ('OW','OO'):
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe['ALsafe'] = outframe['ALsafe'] & (indat[colnam] > 0.99  * indat[colnamAL])
                outframe['osafe']  = outframe['osafe']  & (indat[colnam] < 0.2 * indat[colnamAL])
        outframe['ALsafe'] = outframe['ALsafe'] | (outframe['MaxAfst']==0)
        outframe['osafe'] = outframe['osafe'] & (outframe['MaxAfst']!=0)
    if 1==1:
        outframe['ALsafe'] = outframe['ALsafe'].astype(int)
        outframe['osafe'] = outframe['osafe'].astype(int)
        overlap = outframe['osafe'] * outframe['ALsafe']
        outframe['ALsafe'] = outframe['ALsafe'] - overlap
        outframe['osafe'] = outframe['osafe'] - overlap
        if np.sum(outframe['osafe'] * outframe['ALsafe']) !=0:
            raise ("Error: overlapping fits")

        outframe['FactorVP'] =indat['FactorV']
        outframe['FactorVF'] = ( indat['FactorV'] * outframe['ALsafe'] +
                                  outframe['FactorVFoCor' ]* outframe['osafe']  )
        for lkey in ('LW','LO'):
            colnamAL ="M_"+ lkey +"_AL"
            colnamALo="F_"+ lkey +"_AL"
            outframe[colnamALo] = indat[colnamAL] * (outframe['ALsafe'])
            for okey in ('OW','OO','OM','OA'):
                colnam ="M_"+ lkey +"_" + okey
                colnamo="F_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe[colnamo] = indat[colnam] * (outframe['osafe'])
#    outframe['ALmult'] = ( (outframe['ALsafe']==False).astype(int))
    return outframe

#cut2=  choose_cutoffold(indatverplgr,fitgrps,False,0,'PC4',expdefs)   
cut2=  choose_cutoffold(indatverplmxigr,fitgrps,False,0,'mxigrp',expdefs)   
#cut2

# +
#todo
#fit tov fractie (l komt uit data FactorVL)
#waarde:  p=1/(1/l + 1/f) -> f= 1/ (1/p - 1/l) -> divergeert dus alleen als w>.1, anders 0
#gewicht: w= (p * (1/p - 1/l))** (+ pow+1)   -> aparte kolom
# -

def pointspertype(cutdf):
    cutcnt= cutdf.copy(deep=False)[['osafe','ALsafe','FactorVP','GrpExpl','FactorVFAL']]
    cutcnt['FactorVok'] =cutcnt['FactorVP'] >0
    cutcnt['allrecs'] = cutcnt['FactorVok']*0+1
    cutcnt['osafrat'] = cutcnt['FactorVFAL'] * cutcnt['osafe']
    rv= cutcnt.groupby('GrpExpl').agg('sum')
    rv['osafrat'] = rv['osafrat'] / rv['osafe']
    mlim=0
    return rv[rv['allrecs'] > mlim]
pointspertype(cut2)


# +
#originele code had copy. Kost veel geheugen en tijd
#daarom verder met kolommen met een F_ (filtered)

def choose_cutoffnw(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    curvpwr = pu['CP']
    outframe=indat[['PC4','GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    minwgt=.5
    recisAL=indat['MaxAfst']==0
    if hasfitted:
#        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 5.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
#        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
        outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
        outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
        outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
    else: 
        wval1= indat[recisAL] [['FactorV','PC4','GeoInd'] +pltgrps].copy(deep=False)
        wval1= wval1.rename(columns={'FactorV':'EstVPAL'})
        wval1['EstVPAL'] =np.power(wval1['EstVPAL'],curvpwr)
        outframe=outframe.merge(wval1,how='left')
        outframe['EstVP']  =np.power(indat['FactorV'],curvpwr)
        outframe['EstVPo'] =np.where (recisAL,1e20,1e-9 )    
    if 1==1:
        outframe['FactorVP'] =np.power(indat['FactorV'],curvpwr)
        denomo=  (1/outframe['EstVP']- 1/outframe['EstVPAL'])
        outframe['osafe'] = np.where( outframe['EstVP'] * denomo <=0,0, np.power(
                             outframe['EstVP']*denomo,  curvpwr+1) )  
        outframe['osafe'] =np.where (outframe['osafe'] <minwgt,0,outframe['osafe'] )       
        outframe['FactorVFo'] = np.where(outframe['EstVP']* denomo <=0,0, 
                            np.power(denomo ,-1/curvpwr))
        outframe['FactorVFo'] = indat['FactorV'] *outframe['FactorVFo'] /outframe['EstVP'] 
    if hasfitted:
        denomAL= (1/outframe['EstVP']- 1/outframe['EstVPo'])
    else:
        denomAL= (1/outframe['EstVP']- np.power(outframe['FactorVFo'],- curvpwr ) ) 
    if 1==1:
        outframe['FactorVFAL'] = np.where(outframe['EstVP']* denomAL <=0,0, 
                            np.power(denomAL ,-1/curvpwr))
        outframe['FactorVFAL'] = indat['FactorV'] *outframe['FactorVFAL'] /outframe['EstVP'] 
        outframe['FactorVFAL'] = np.where (recisAL,indat['FactorV'],
                            outframe['FactorVFAL'] )               
    if hasfitted:
        outframe['ALsafe'] = np.where( outframe['EstVP'] * denomAL <=0,0, np.power(
                             outframe['EstVP']*denomAL,  curvpwr+1) )  
        outframe['ALsafe'] = np.where (outframe['ALsafe'] <minwgt,0,outframe['ALsafe'] )  
        outframe['ALsafe'] = np.where (recisAL,1.0,outframe['ALsafe'] )                             

        chooseAL=np.where( (outframe['ALsafe'] > outframe['osafe'] )| recisAL ,1,0)
        outframe['ALsafe'] = outframe['ALsafe'] *chooseAL
        outframe['osafe'] = outframe['osafe'] *(1-chooseAL)
    else:                
        outframe['ALsafe'] = np.where (recisAL,1,0)
    if 1==1:
        if np.sum(outframe['osafe'] * outframe['ALsafe']) !=0:
            raise ("Error: overlapping fits")
        outframe['FactorVF'] = (outframe['FactorVFo']*outframe['osafe'] +
                                outframe['FactorVFAL'] * outframe['ALsafe'] )
        for lkey in ('LW','LO'):
            colnamAL ="M_"+ lkey +"_AL"
            colnamALo="F_"+ lkey +"_AL"
            outframe[colnamALo] = indat[colnamAL] * outframe['ALsafe']
            for okey in ('OW','OO','OM','OA'):
                colnam ="M_"+ lkey +"_" + okey
                colnamo="F_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe[colnamo] = indat[colnam] * outframe['osafe']
#        outframe['osafe2'] =outframe['osafe']
#        outframe['ALsafe2'] =outframe['ALsafe']
        print( ( np.sum(outframe['ALsafe']),np.sum(outframe['osafe']),
                 np.max(outframe['ALsafe']),np.max(outframe['osafe'])) )
    return outframe

def choose_cutoff(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr):
    if False:
        return choose_cutoffnw(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)
    else:
        return choose_cutoffold(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)


#cut2=  choose_cutoff(indatverplgr,fitgrps,False,0,'PC4',expdefs)   
cut2=  choose_cutoff(indatverplmxigr,fitgrps,False,0,'mxigrp',expdefs)   
#cut2
# -

def fitinddiag(fitdf,motiefc,naarhuisc,geoindex,grpind,pu):
    curvpwr = pu['CP']
    seldf = fitdf [ (fitdf ['MotiefV'] ==motiefc) &
                  (fitdf ['isnaarhuis'] ==naarhuisc) &
                  (fitdf ['GeoInd'] ==geoindex) & 
                  (fitdf ['FactorVP'] >0)] .copy()
    if False:
        print(seldf[['EstVPAL','MotiefV']].groupby(['EstVPAL']).agg('count').\
         groupby(['MotiefV',' isnaarhuis']).agg('count'))
    pldf=    seldf[['osafe','ALsafe']].copy()
    pldf['FactorVPrel'] =    seldf['FactorVP'] /  seldf['EstVPAL']  
    sumchk = np.power (np.power(seldf['FactorVFo'],-curvpwr) + 
                       np.power(seldf['FactorVFAL'] ,-curvpwr) , -curvpwr )
    pldf['FactorVSrel'] = sumchk /  seldf['FactorVP']
    pldf['FactorVFoCor'] =   np.where(seldf['osafe']==0,-.1,
                                      seldf['FactorVFo'] /  seldf['FactorVP']  )
    pldf['FactorVFALCor'] =   np.where(seldf['ALsafe']==0,-.1,
                                seldf['FactorVFAL'] /  seldf['FactorVP'] )
    if True:
        print(pldf[pldf['FactorVFALCor'] >5])
        print(seldf[pldf['FactorVFALCor'] >5])
        print(pldf[( abs(pldf['FactorVSrel']-1) >1e-3) & (( abs(pldf['FactorVSrel']-0) >1e-3)) ])
    plmelt =  pd.melt(pldf, 'FactorVPrel', var_name='cols',  value_name='vals')
    
    if False:
        print(seldf.sort_values(by=[grpind,'KAfstCluCode'])[['FactorVFo','FactorVP']] )
    fig, ax = plt.subplots()    
    seaborn.scatterplot(data=plmelt,x="FactorVPrel",y="vals", hue='cols', ax=ax)
    ax.set_xscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#fitinddiag(cut2,10,5,'VertPC','mxigrp',expdefs)    


# +
def _regressgrp(indf, yvar, xvars,pcols):  
#        reg_nnls = LinearRegression(fit_intercept=False )
#        print(('o',len(indf)) )
        y_train=indf[yvar]
        X_train=indf[xvars]
        if 1==1:
            #smask = ~ (np.isnan(y_train) | np.isnan(np.sum(X_train)) )
            smask =  (y_train >0) & (np.sum(X_train,axis=1) >0) 
            indf= indf[smask]
            y_train=indf[yvar]
            X_train=indf[xvars]
#            print(('f', len(indf)))
        else:
            y_train[np.isnan(y_train)]=0.1
        if(len(indf)==0) :
            rv=np.zeros(len(xvars))
        else:
            fit1 = nnls(X_train, y_train)    
            rv=pd.DataFrame(fit1[0],index=pcols).T
        return(rv)


#@jit(parallel=True)
def _fitsub(indf,fitgrp,_regressgrp,  colvacols2, colpacols2):
    rf= indf.groupby(fitgrp ).apply( _regressgrp, 'FactorVF', colvacols2, colpacols2)
    return rf
    
    
def fit_cat_parameters(indf,topreddf,pltgrp,pu):
    debug=False
    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'F_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    Fitperscale=False
    if Fitperscale:
        fitgrp=pltgrp +['KAfstCluCode','GeoInd'  ]
    else:
        fitgrp=pltgrp + ['GeoInd' ]
    rf= _fitsub(indf,fitgrp,_regressgrp,  colvacols2, colpacols2).reset_index()
    return rf

def predict_values(indf,topreddf,pltgrp,rf,pu,stobijdr):
    curvpwr = pu['CP']
#    indf = indf[(indf['MaxAfst']!=95.0) & (indf[pltgrp]<3) ]
    debug=False
    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'F_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    
    outdf = topreddf.merge(rf,how='left')
    #let op: voorspel uit M kolommen
    colvacols2 = np.array( list ( (re.sub(r'P_','M_',s) for s in list(colpacols2) ) ) )
    
    colpacols2alch = np.array( list ( (re.sub(r'_AL','_xx',s) for s in list(colvacols2) ) ) )
    colpacols2alchisAL = (colpacols2alch !=colvacols2)
    if (debug):
        print(colpacols2alchisAL)
    blk1=outdf[colvacols2 ] * ((colpacols2alchisAL ==False ).astype(int))
    blk2=outdf[colpacols2 ] * ((colpacols2alchisAL ==False ).astype(int))
#    print(blk1)
    s2= np.sum(np.array(blk1)*np.array(blk2),axis=1).astype(float)

    blk1al=outdf[colvacols2 ] * (colpacols2alchisAL.astype(int))
    blk2al=outdf[colpacols2 ] * (colpacols2alchisAL.astype(int))
#    print(blk1)
    s2al= np.sum(np.array(blk1al)*np.array(blk2al),axis=1).astype(float)    
    outdf['FactorEstAL']  =s2al
#todo: als s2al 0 is, opzoeken waar MaxAfst ==0 en de s2al daarvan invullen als default
#dat levert dan min of meer consistente kantelpunten op
    outdf['FactorEstNAL'] =s2
    if (debug):
        print ((s2al, s2))
    #s2ch= np.min( (np.where((s2==0),s2al,s2 ), np.where((s2al==0),s2,s2al ) ) ,axis=0)
    s2ch= np.where((s2<=0),np.where(outdf['MaxAfst']==0, s2al,0), 
                           np.where((s2al==0),s2,
                           np.power (np.power(s2,-curvpwr) + np.power(s2al,-curvpwr), -curvpwr )) )
    if (debug):
        print (s2ch)
    outdf['FactorEst'] = s2ch
    outdf['DiffEst'] = np.where(outdf['FactorV']>0, outdf['FactorV']-s2ch,np.nan)
    if stobijdr:
        outdf[colvacols2 ] = np.array(blk1al)*np.array(blk2al)
    return(outdf)

def _dofitdatverplgr(indf,topreddf,pltgrp,pu):
    rf = fit_cat_parameters(indf,topreddf,pltgrp,pu)
    return predict_values(indf,topreddf,pltgrp,rf,pu,False)

fitpara= fit_cat_parameters(cut2,indatverplmxigr,fitgrps,expdefs)
fitdatverplgr = predict_values(cut2,indatverplmxigr,fitgrps,fitpara,expdefs,False)
#fitdatverplgr = dofitdatverplgr(cut2,indatverplgr,fitgrps,expdefs)
#fitdatverplgr = dofitdatverplgr(cut2,indatverplmxigr,fitgrps,expdefs)
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="FactorEst",y="DiffEst",hue="GeoInd")
# -

cut3=  choose_cutoff(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs)  
#cut3=  choose_cutoff(indatverplgr,fitgrps,True,fitdatverplgr,expdefs)  
#cut3

# +
#fitinddiag(cut3,10,5,'VertPC',p_CP)    
# -

#voor de time being, overschrijf de vorige selectie gegevens
for r in range(2):
    cut3=  choose_cutoff(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs) 
    fitpara= fit_cat_parameters(cut3,indatverplmxigr,fitgrps,expdefs)
    fitdatverplgr = predict_values(cut3,indatverplmxigr,fitgrps,fitpara,expdefs,False)
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="FactorEst",y="DiffEst",hue="GeoInd")

fitdatverplgr["x_LM_AL"] = fitdatverplgr["M_LW_AL"] * fitdatverplgr["M_LO_AL"]
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="x_LM_AL",y="DiffEst",hue="GeoInd")

seaborn.scatterplot(data=fitdatverplgrx,x="M_LO_AL",y="DiffEst",hue="GeoInd")

pointspertype(cut3)

gr5km=fitdatverplgr[(fitdatverplgr['MaxAfst']==5) & (fitdatverplgr['MotiefV']==1)].copy()
gr5km['linpmax']=gr5km['FactorEstNAL']/ gr5km['FactorEstAL']
gr5km['linpch']= gr5km['FactorEst']/ gr5km['FactorEstAL']
gr5km['drat']= gr5km['FactorV']/ gr5km['FactorEstAL']
fig, ax = plt.subplots()
seaborn.scatterplot(data=gr5km,x="linpmax",y="drat",size=.02,hue="GeoInd",ax=ax)
ax.set_xscale('log')
ax.set_yscale('log')

# +
#gr5km[gr5km['linpch'] >1000][['FactorEst','FactorEstAL','FactorEstNAL'] ]
# -

ds=fitdatverplgr[(fitdatverplgr['MotiefV']==1) & (fitdatverplgr['MaxAfst']==0)].sort_values(by='DiffEst').copy()
ds['dchk' ] = ds['DiffEst'] / ds ['FactorVSpec']
ds


# +
#noot normeren naar 1e6 rittenjaar
#noot2: er zal een afstand schaal zijn waarbij M_LW_OM en M_LO_OM een betere fit info geven
# het is niet zo vreemd dat het totale aantal woon-werk ritten ook een 
# nabijheids component heeft

def mxitotdiagpl(ds,xax):
    fig, ax = plt.subplots()
    #seaborn.scatterplot(data=ds,x='mxigrp',y='DiffEst',hue='GrpExpl')
    ds['huecol'] = ds['GrpExpl'] + ' ' + ds['GeoInd']
    seaborn.scatterplot(data=ds,x=xax,y='FactorV',hue='huecol')
    seaborn.lineplot(data=ds,x=xax,y='FactorEst',hue='huecol')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('1 groep landelijke sommen per mxigrp : data versus fit')
    print(ds.groupby([ 'GrpExpl','GeoInd'])[['FactorV','FactorEst','DiffEst']].agg('sum') )
#mxitotdiagpl(ds,'M_LO_AL') 
mxitotdiagpl(ds,'mxigrp')
# -

fitdatverplgr[fitdatverplgr['MaxAfst']==0]

paratab=fitdatverplgr[fitdatverplgr['MaxAfst']==0].groupby (['MotiefV','isnaarhuis','GeoInd']).agg('mean')
paratab.to_excel("../output/fitparatab1.xlsx")
paratab

fitdatverplgr.groupby (['MotiefV','isnaarhuis','GeoInd','MaxAfst']).agg('mean')


# +
def getmaxafstadmax( dd, landcod):
    rf = dd[dd['KAfstCluCode'] == landcod ] 
    binst= useKAfstV.iloc[-2,1]
    print(binst)
    binstm=1
    rf['MaxShow'] = binstm * rf['FactorKm_c'] / rf['FactorV_c'] - (binstm-1)*binst
    rf['MaxShStat'] =  rf['FactorV_c'] 
    rf = rf[ODINcatVNuse.fitgrpse +['MaxShow','MaxShStat']]
    return rf
#maxvals = 
getmaxafstadmax(ODINcatVNuse. odindiffflginfo, ODINcatVNuse.landcod)

# hier komen waarden uit ONDER binstm. Dat is niet goed.

# +
#mogelijk: maxafst joinen uit diifdata


def pltmotdistgrp (mydati,horax,vertax,vnsep):
    mydat=pd.DataFrame(mydati)    
    opdel=['MaxAfst','GeoInd','MotiefV','GrpExpl']
#    mydat['FactorEst2'] = np.where(mydat['FactorEstNAL']==0,0, 
#                                 1/ (1/mydat['FactorEstAL'] + 1  /mydat['FactorEstNAL'] ) )
    fsel=['FactorEst','FactorV', 'FactorEstNAL','FactorEstAL']

    rv2= mydat.groupby (opdel)[fsel].agg(['sum']).reset_index()
#    print ( rv2[ (rv2['MotiefV']==1) & (rv2['MaxAfst']==0) ] ) 
    rv2.columns=opdel+fsel
    if(vertax=='FactorEst'):
        limcat=2e9
    else:
        limcat=1e9
    bigmotd= rv2[(rv2['MaxAfst']==0) & (rv2['FactorV']>limcat)  ].groupby('GrpExpl').agg(['count']).reset_index()
    bigmotl = list( bigmotd['GrpExpl'])
#    print(bigmotl)
    stelafst =100.0
    rv2['MaxAfst']=np.where(rv2['MaxAfst']==0 ,stelafst ,rv2['MaxAfst'])
#    rv2['MaxAfst']=rv2['MaxAfst'] * np.where(rv2['GeoInd']=='AankPC',1,1.02)
    rv2['Qafst']=1/(1/(rv2['MaxAfst']  *0+1e10) +1/ (np.power(rv2['MaxAfst'] ,1.8) *2e8 ))
    rv2['linpmax'] = rv2['FactorEstNAL']/ rv2['FactorEstAL']
    rv2['linpch']= rv2['FactorEst']/ rv2['FactorEstAL']
    rv2['drat']= rv2['FactorV']/ rv2['FactorEstAL']

    rvs = rv2[np.isin(rv2['GrpExpl'],bigmotl)]
#    rv2['MotiefV']=rv2['MotiefV'].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(12, 6))
    
#    print ( rvs[ (rvs['MotiefV']==1) & (rvs['MaxAfst']== stelafst) ] ) 
    
    rvs['huecol'] = rvs['GrpExpl']
    if vnsep:
        rvs['huecol'] = rvs['huecol'] + ' ' + rvs['GeoInd']
    if(vertax=='FactorEst'):
        seaborn.scatterplot(data=rvs,x=horax,y='FactorV',hue='huecol',ax=ax)
        seaborn.lineplot(data=rvs,x=horax,   y='FactorEst',hue='huecol',ax=ax)
#Qafst nooit zomaar plotten: ggeft grote verwarring !
#        seaborn.lineplot(data=rvs,x=horax,   y='Qafst',ax=ax)
    elif(vertax=='linpch'):
        seaborn.scatterplot(data=rvs,x=horax,y='drat',hue='huecol',ax=ax)
        ax.axhline(0.5)
        seaborn.lineplot(data=rvs,x=horax,   y='linpch',hue='huecol',ax=ax)
    ax.set_xscale('log')
#    ax.set_yscale('log')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figname = "../output/gplo_fmdg_"+"horax"+"_"+vertax+"_"+'G1.svg';
    fig.savefig(figname, bbox_inches="tight") 
    return (rv2)
ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','FactorEst',False)
# -

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==1],'MaxAfst','FactorEst',True)

ov=pltmotdistgrp(fitdatverplgr,'linpmax','linpch',False)

ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','linpch',False)


def calcchidgrp (mydati):
    mydat=pd.DataFrame(mydati)
    mydat['chisq'] = mydat['DiffEst'] ** 2
    mydat['insq'] = mydat['FactorV'] ** 2
    csel=['chisq','insq']
    opdel=['MaxAfst','GeoInd']
    rv= mydat.groupby (opdel)[csel].agg(['sum']).reset_index()
    rv.columns=opdel+csel
    rv['ChiRat'] = rv['chisq']/ rv['insq']
    mydat['HeeftEst']=(np.isnan(mydat['FactorEst'])==False).astype(int)
    mydat['HeeftFV']=(np.isnan(mydat['FactorV'])==False).astype(int)
    fsel=['FactorEst','FactorV','HeeftEst','HeeftFV']
    rv2= mydat.groupby (opdel)[fsel].agg(['sum']).reset_index()
    rv2.columns=opdel+fsel
#    print(rv2)
    rv2['EstRat'] = rv2['FactorEst']/ rv2['FactorV']
    rv=rv.merge(rv2,how='left')
    return rv
calcchidgrp(fitdatverplgr)


# +
def trypowerland (pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV,myskipPCMdf,pltgrps,puin,v1i,v1v,v2i,v2v):
    pu= puin.copy()
    pu[v1i]=v1v
    pu[v2i]=v2v
    print(pu)
    mygeoschpc4all= mkgeoschparafr(pc4data,pc4grid,rudigrid,myKAfstV,pu)
    mygeoschpc4i, geobingr = addmxibins(mygeoschpc4all,nmxibins_glb)
    mygeoschmixpMotief= allmotmxicorrgrp(summmxigrp(mygeoschpc4i),
                    summmxicorrgrp(mygeoschpc4i,myskipPCMdf),np.max(odinverplgr['MotiefV']))
    myodinverplmxigr = odinmergemxi (odinverplgr ,mygeoschpc4i,grpexpcontrs)
    myxlatKAfstV=myKAfstV[['KAfstCluCode']].merge(inxlatKAfstV,how='left')
#    print (myxlatKAfstV)
    mydatverplgr = mkdfverplxypc4 (myodinverplmxigr ,fitgrps,'Motief en isnaarhuis',
                                myKAfstV,xlatKAfstV,mygeoschmixpMotief,2.0).merge(myKAfstV,how='left')
    
    cut2i=  choose_cutoff(mydatverplgr,pltgrps,False,0,'mxigrp',pu)  
    myfitpara= fit_cat_parameters(cut2i,mydatverplgr,pltgrps,pu)
    myfitverplgr = predict_values(cut2i,mydatverplgr,pltgrps,myfitpara,pu,False)

#    myfitverplgr = dofitdatverplgr(cut2i,mydatverplgr,pltgrps,pu)
    for r in range(2):
        cut3i=  choose_cutoff(mydatverplgr,pltgrps,True,myfitverplgr,'mxigrp',pu) 
        myfitpara= fit_cat_parameters(cut2i,mydatverplgr,pltgrps,pu)
        myfitverplgr = predict_values(cut2i,mydatverplgr,pltgrps,myfitpara,pu,False)
#        myfitverplgr = dofitdatverplgr(cut3i,mydatverplgr,pltgrps,pu)
    rdf=calcchidgrp(myfitverplgr)
    chisq= np.sum(rdf['chisq'].reset_index().iloc[:,1])
    return([chisq,myfitpara,rdf])
    
#rv=trypowerland(cbspc4data,pc4inwgrid,rudifungrid,useKAfstVland,xlatKAfstV,1.3,1.0,2.0)
rv,pdf,rdf=trypowerland(cbspc4data,pc4inwgcache,rudifungcache,useKAfstVQ,xlatKAfstV,
                skipPCMdf,fitgrps,
                expdefs,'LW',1.1,'yy',2)
rv
# -



# +
#let op p_LO ongelijk 1 geeft vooral veel negatieve waarden en dus NAs, die punten tellen dan niet mee in chi^2

#@jit(parallel=True)
def chisqsampler (pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV,pltgrps):
#    expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0}
    pl = expdefs.copy()
    lw = np.linspace(1.1,1.3,3)
    oa = np.linspace(1.6,2.0,3)
    lo = np.linspace(1.7,2.0,3)
    p_LW,p_OA= np.meshgrid(lw, oa)
#    l_OA=2.0
#    l_LW=1.2
#    print( (p_LW,p_LO))
    myfunc =lambda  l_LW,l_OA  :trypowerland (pc4data,pc4grid,rudigrid,
                                              myKAfstV,inxlatKAfstV,pltgrps,
                                              pl,'LW',l_LW,'OA',l_OA)
    vfunc = np.vectorize(myfunc)
    z= ( vfunc(p_LW,p_OA) )
    z=np.array(z)
    return z
#chitries= chisqsampler (cbspc4data,pc4inwgrid,rudifungrid,useKAfstVland,xlatKAfstV)    
if False:
    chitries= chisqsampler (cbspc4data,pc4inwgcache,rudifungcache,useKAfstV,xlatKAfstV,fitgrps)    
    print (chitries)

#    lw = np.linspace(1,1.4,3)
#    lo = np.linspace(0.8,1.0,3)
#[[7.59435412e+16 7.40995871e+16 7.55013017e+16]
# [7.59435412e+16 7.40995871e+16 7.55013017e+16]
# [7.08332159e+16 6.82369212e+16 6.84797520e+16]]
#    lo = np.linspace(1.0,1.4,3)
#[7.08332159e+16 6.82369212e+16 6.84797520e+16]
# [7.59435412e+16 7.40995871e+16 7.55013017e+16]
# [7.59435412e+16 7.40995871e+16 7.55013017e+16]]
# -

seaborn.scatterplot(data=fitdatverplgr[fitdatverplgr['MaxAfst']==0],x="FactorEst",y="DiffEst",hue="GeoInd")

fitdatverplpc4gr = predict_values(cut3,indatverplpc4gr,fitgrps,fitpara,expdefs,False)

fitdatverplpc4gr[fitdatverplpc4gr['FactorEst']>1e7][['PC4', 'GrpExpl','GeoInd',
          'FactorEst', 'DiffEst','FactorV','FactorVSpec']].sort_values(
    by=['PC4', 'GrpExpl','GeoInd'])

pllanddiff= cbspc4data[['postcode4']].merge(fitdatverplpc4gr[(fitdatverplpc4gr['MaxAfst']==0) 
                    &  (fitdatverplgr ['MotiefV'] ==6 )][['PC4','DiffEst','FactorEst','FactorV']],
            how='left',left_on=('postcode4'), right_on = ('PC4'))
print ( (len(cbspc4data), len(pllanddiff) ) )


def largestdiffsPC (pc4dta,indf):
    pllanddiffam= pc4dta[['postcode4']].merge(indf[(indf['MaxAfst']==0) ] ,
                                    how='left',left_on=('postcode4'), right_on = ('PC4'))
    pllanddiffam['DiffAbs'] =abs(pllanddiffam['DiffEst'])
    okval= pllanddiffam[np.isnan(pllanddiffam['DiffAbs']) == False]
    rv =okval.sort_values(by='DiffAbs').tail(30).groupby(
          ['postcode4','GrpExpl'])[['isnaarhuis','DiffEst']].agg(
           {'isnaarhuis':'count','DiffEst':'mean'} )
    return rv
largestdiffsPC (cbspc4data,fitdatverplpc4gr)

# +
#inspectie
#grote onderschatters landelijK; winkelcentra, schiphol, en mindere mate universiteit

#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(3511,3512,3584 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(6511,6525 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(2513 ,2333)))]
chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(2333,2334,2331) ))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(5611,5612 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(2678,2691 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(1012,1017,1043,1101,1118 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(2262,2333,2511,2595 )))]
pchkpckrt = chkpckrt.to_crs(epsg=plot_crs).plot(alpha=.3)
cx.add_basemap(pchkpckrt, source= prov0)
# -

pllanddiff.index


# +
def quicklandpc4plot(pc4df,pc4grid,pc4dfcol):
    #idximg=pc4grid.read(1)
    idximg=pc4grid[1]
    print( np.min(idximg),np.max(idximg) )
    image=np.zeros(idximg.shape,dtype=np.float32)
    print( np.min(pc4df[pc4dfcol] ),np.max(pc4df[pc4dfcol] ) )
    rasteruts1.fillvalues(image,idximg,np.float32(pc4df[pc4dfcol]) )
    print( np.min(image),np.max(image) )
    fig, ax = plt.subplots()
    nlextent=[0,280000,300000, 625000]
    #image = np.isnan(image)
    plt.imshow(image,cmap='jet',alpha=.6,extent=nlextent)
    plt.colorbar()
    
quicklandpc4plot(pllanddiff,pc4inwgcache,'DiffEst')
# -

#basemap slows, so swith off by default
if 1==0:
    fig, ax = plt.subplots()
    pland= pllanddiff.plot(column= 'DiffEst',
                                cmap='jet',legend=True,alpha=.6,ax=ax)
elif 0==1:    
    pland= pllanddiff.to_crs(epsg=plot_crs).plot(column= 'DiffEst',
                                cmap='jet',legend=True,alpha=.6)
    cx.add_basemap(pland, source= prov0)
#plland.plot( column= 'DiffEst', Legend=True)



# +
#verwachte relaties
#woon-werk (naarhuis andersom - kijk hoe te plotten)
#oppervlak werk (niet genorm op geo) ~ aantal ritten met werk aan niet-woon zijde
#oppervlak woon (niet genorm op geo) ~ aantal ritten met werk aan woon zijde
#dit zou er in x-y plotjes uit moeten kunnen komen
#verwachting: onafhankelijk van afstandsklasse of jaar
#verwachting: regressie per klasse is mogelijk -> levert ook info op
#voor andere motieven: niet-woon zijde kan relatie met werk of woon opp hebben -> check
#kijk eens naar verdelingen totaal aantal verplaatsingen per persoon
#maak zonodig extra kolommen aan in database, waar meerdere PC4 databases mogelijk zijn
# format [AankPC][VertPC]/[dbpc4]/[dbpc4 field] 
# maak groepen op aantallen postcodes (of oppervlakken ?)
#1 plot punt per PC4 -> middelen gaat in regressie

def mkpltverplxypc4 (df,myspecvals,xvar,pltgrp,selstr,ngrp):
    xsrcarr=  xvar.split ('/')
    xvarPC = xsrcarr[0]
    gvarPC='PC4'
    dfvrecs = df [ (df[gvarPC] > 500) &(df['GeoInd']==xvarPC ) ]   
    dfvrecs=addparscol(dfvrecs,pltgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[pltgrp, gvarPC,'FactorV']].groupby([pltgrp, gvarPC]).sum().reset_index()
    pstats =addparscol(pstats,xvar)
    #print(pstats)
    denoms= pstats [[pltgrp, 'FactorV']].groupby([pltgrp]).sum().reset_index().rename(columns={'FactorV':'Denom'} )
    denoms [ 'Denom'] =0.01
    #print(denoms)
    pstatsn = pstats.merge(denoms,how='left')
    pstatsn['FractV'] = pstatsn['FactorV'] *100.0/ pstatsn['Denom']
    if ngrp !=0:
        pstatsn['GIDX'] = pd.qcut(pstatsn[xvar], ngrp)
        pstatsn = pstatsn.groupby([pltgrp,'GIDX']).mean().reset_index()
    
#    vardescr = dbk_2022_cols [dbk_2022_cols['Variabele_naam_ODiN_2022'] == xvar] ['Variabele_label_ODiN_2022']
#    print(vardescr)
    vardescr=[]
    if len(vardescr) ==0:
        vardescr = ""        
        heeftlrv = True
    else:
        vardescr = vardescr.item()
        heeftlrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==xvar) ] ) !=0
#    print(vardescr,heeftlrv)

    if 0==0:
        grplrv=True
    else:
        grplrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==pltgrp) ] ) !=0
    if grplrv==False:
        explhere = myspecvals [myspecvals['Variabele_naam'] == pltgrp].copy()
        explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
#   print(explhere)
        pstatsn=pstatsn.merge(explhere,left_on=pltgrp, right_on='Code', how='left')    
        pstatsn[pltgrp] = pstatsn[pltgrp].astype(str)  + " : " + pstatsn['Code_label']    
        pstatsn= pstatsn.drop(columns=['Code','Code_label'])

    ylab="Percentage of FractV in group"
    xlab=xvar + " : "+ vardescr 
    if heeftlrv:
#        pstatsn['Code_label'] = pstatsn[collvar] 
#        print(pstatsn)
        if vardescr == "" :
            chart= sns.relplot(data=pstatsn, y='FractV', x=xvar, hue=pltgrp, kind="scatter",height=5, aspect=2.2)
        else:
            chart= sns.relplot(data=pstatsn, y='FractV', x=xvar, hue=pltgrp, kind="line",height=5, aspect=2.2)
    else:
        chart= sns.relplot(data=pstatsn, y='FractV', x=xvar, hue=pltgrp, kind="scatter",height=5, aspect=2.2)
    chart.fig.suptitle(selstr)
    chart.set_xlabels(xlab)
    chart.set_ylabels(ylab)
    return(pstatsn)
   
#naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]    
naarhuis= odinverplgr [odinverplgr ['isnaarhuis'] ==6 ]
#dummy voor specvaltab
dspecvaltab={'Code':0 , 'Variabele_naam':'none'}
datpltverplp = mkpltverplxypc4 (naarhuis,dspecvaltab,
                                'AankPC/rudifun/S_MXI22_BWN','MotiefV','Naar huis',100)


# +
#verwachte relaties
#woon-werk (naarhuis andersom - kijk hoe te plotten)
#oppervlak werk (niet genorm op geo) ~ aantal ritten met werk aan niet-woon zijde
#oppervlak woon (niet genorm op geo) ~ aantal ritten met werk aan woon zijde
#dit zou er in x-y plotjes uit moeten kunnen komen
#verwachting: onafhankelijk van afstandsklasse of jaar
#verwachting: regressie per klasse is mogelijk -> levert ook info op
#voor andere motieven: niet-woon zijde kan relatie met werk of woon opp hebben -> check
#kijk eens naar verdelingen totaal aantal verplaatsingen per persoon
#maak zonodig extra kolommen aan in database, waar meerdere PC4 databases mogelijk zijn
# format [AankPC][VertPC]/[dbpc4]/[dbpc4 field] 
# maak groepen op aantallen postcodes (of oppervlakken ?)
#1 plot punt per PC4 -> middelen gaat in regressie

def mkfitverplxypc4 (df,myspecvals,xvar,pltgrp,selstr,ngrp):
    xsrcarr=  xvar.split ('/')
    xvarPC = xsrcarr[0]
    gvarPC='PC4'
    dfvrecs = df [ (df[gvarPC] > 500) &(df['GeoInd']==xvarPC ) ]   
    
    dfvrecs=addparscol(dfvrecs,pltgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[pltgrp, gvarPC,'FactorV']].groupby([pltgrp, gvarPC]).sum().reset_index()
    pstats =addparscol(pstats,xvar)
    #print(pstats)
    denoms= pstats [[pltgrp, 'FactorV']].groupby([pltgrp]).sum().reset_index().rename(columns={'FactorV':'Denom'} )
    denoms [ 'Denom'] =0.01
    #print(denoms)
    pstatsn = pstats.merge(denoms,how='left')
    pstatsn['FractV'] = pstatsn['FactorV'] *100.0/ pstatsn['Denom']
    if ngrp !=0:
        pstatsn['GIDX'] = pd.qcut(pstatsn[xvar], ngrp)
        pstatsn = pstatsn.groupby([pltgrp,'GIDX']).mean().reset_index()
    
    #vardescr = dbk_2022_cols [dbk_2022_cols['Variabele_naam_ODiN_2022'] == xvar] ['Variabele_label_ODiN_2022']
#    print(vardescr)
    vardescr=[]
    if len(vardescr) ==0:
        vardescr = ""        
        heeftlrv = True
    else:
        vardescr = vardescr.item()
        heeftlrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==xvar) ] ) !=0
#    print(vardescr,heeftlrv)

    if 0==0:
        grplrv=True
    else:
        grplrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==pltgrp) ] ) !=0
    if grplrv==False:
        explhere = myspecvals [myspecvals['Variabele_naam'] == pltgrp].copy()
        explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
#   print(explhere)
        pstatsn=pstatsn.merge(explhere,left_on=pltgrp, right_on='Code', how='left')    
        pstatsn[pltgrp] = pstatsn[pltgrp].astype(str)  + " : " + pstatsn['Code_label']    
        pstatsn= pstatsn.drop(columns=['Code','Code_label'])

    ylab="Percentage of FractV in group"
    xlab=xvar + " : "+ vardescr 
    if heeftlrv:
#        pstatsn['Code_label'] = pstatsn[collvar] 
#        print(pstatsn)
        if vardescr == "" :
            chart= sns.relplot(data=pstatsn, y='FractV', x=xvar, hue=pltgrp, kind="scatter",height=5, aspect=2.2)
        else:
            chart= sns.relplot(data=pstatsn, y='FractV', x=xvar, hue=pltgrp, kind="line",height=5, aspect=2.2)
    else:
        chart= sns.relplot(data=pstatsn, y='FractV', x=xvar, hue=pltgrp, kind="scatter",height=5, aspect=2.2)
    chart.fig.suptitle(selstr)
    chart.set_xlabels(xlab)
    chart.set_ylabels(ylab)
    return(pstatsn)
   

datpltverplp = mkfitverplxypc4 (naarhuis,dspecvaltab,
                                'AankPC/rudifun/S_MXI22_BWN','MotiefV','Naar huis',100)

# +
#Geen jaar onderscheid meer
#datpltverplp = mkpltverplxypc4 (naarhuis,dspecvaltab,'VertPC/rudifun/S_MXI22_BWN',
#                                'Jaar','Naar huis',100)
# -

haarhuisapart = addparscol(naarhuis,'VertPC/rudifun/S_MXI22_BWN')
haarhuisapart = haarhuisapart[ (haarhuisapart['VertPC/rudifun/S_MXI22_BWN'] <1e3 ) | 
                               (haarhuisapart['VertPC/rudifun/S_MXI22_BWN'] >2e6 ) ]

haarhuisapart.groupby(['PC4']).agg({'PC4':'count','VertPC/rudifun/S_MXI22_BWN':'mean'} )

cbspc4data.merge(haarhuisapart, left_on='postcode4', right_on='PC4', how='right').plot()


# +
#kijk eens naar verdelingen totaal aantal verplaatsingen per persoon
#maak zonodig extra kolommen aan in database, waar meerdere PC4 databases mogelijk zijn
# format [AankPC][VertPC]/[dbpc4]/[dbpc4 field] 
# maak groepen op aantallen postcodes (of oppervlakken ?)

def mkpltverplp (df,myspecvals,collvar,normgrp,selstr):
    xsrcarr=  collvar.split ('/')
    xvarPC = xsrcarr[0]

    gvarPC='PC4'
    dfvrecs = df [ (df[gvarPC] > 500) &(df['GeoInd']==xvarPC ) ]   

    dfvrecs=addparscol(dfvrecs,collvar)
    dfvrecs=addparscol(dfvrecs,normgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[normgrp, collvar,'FactorV']].groupby([normgrp, collvar]).sum().reset_index()
    #print(pstats)
    denoms= pstats [[normgrp, 'FactorV']].groupby([normgrp]).sum().reset_index().rename(columns={'FactorV':'Denom'} )
    #print(denoms)
    pstatsn = pstats.merge(denoms,how='left')
    pstatsn['FractV'] = pstatsn['FactorV'] *100.0/ pstatsn['Denom']
    #vardescr = dbk_2022_cols [dbk_2022_cols['Variabele_naam_ODiN_2022'] == collvar] ['Variabele_label_ODiN_2022']
    vardescr=[]
#    print(vardescr)
    if len(vardescr) ==0:
        vardescr = ""        
        heeftlrv = True
    else:
        vardescr = vardescr.item()
        heeftlrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==collvar) ] ) !=0
#    print(vardescr,heeftlrv)
    xlab="Percentage of FractV in group"
    ylab=collvar + " : "+ vardescr 
    if heeftlrv:
#        pstatsn['Code_label'] = pstatsn[collvar] 
#        print(pstatsn)
        if vardescr == "" :
            chart= sns.catplot(data=pstatsn, x='FractV', y=collvar, hue=normgrp, kind="bar",orient="h",height=5, aspect=2.2)
        else:
            chart= sns.relplot(data=pstatsn, y='FractV', x=collvar, hue=normgrp,  kind="line",height=5, aspect=2.2)
            xlab=ylab
            ylab="Percentage of FractV in group"
    else:
        explhere = myspecvals [myspecvals['Variabele_naam'] == collvar].copy()
        explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
#        print(explhere)
        pstatsn=pstatsn.merge(explhere,left_on=collvar, right_on='Code', how='left')
#        print(pstatsn)
        pstatsn['Code_label'] = pstatsn[collvar].astype(str)  + " : " + pstatsn['Code_label']
        chart= sns.catplot(data=pstatsn, x='FractV', y='Code_label', hue=normgrp, kind="bar",orient="h",height=5, aspect=2.2)            
    chart.fig.suptitle(selstr)
    chart.set_xlabels(xlab)
    chart.set_ylabels(ylab)
    return(pstatsn)
   
#naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
#datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','Jaar','Alle ritten')
#datpltverplp = mkpltverplp (allodinyr,specvaltab,'VertUur','Jaar','Alle ritten')
datpltverplp = mkpltverplp (naarhuis,dspecvaltab,'AankPC/rudifun/S_MXI22_GB','MotiefV','Alle ritten')
# +
#top . nu eerst kijken naar kengetallen van basis fit


# +
#check 1 reisigerskm auto kilometers
# -

odindiffflginfo= ODINcatVNuse.convert_diffgrpsidat(odinverplflgs,
                ODINcatVNuse.fitgrpse,[],ODINcatVNuse.kflgsflds, [],"_c",ODINcatVNuse.landcod,False)

#ddc_indat =  ODINcatVNuse.mkdatadiff(fitdatverplgr,ODINcatVNuse.fitgrpse,ODINcatVNuse.landcod)
ddc_indat =  ODINcatVNuse.mkdatadiff(fitdatverplgr,
                         ODINcatVNuse.fitgrpse,ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)
totinf_indat = ODINcatVNuse.mkinfosums(ddc_indat,odindiffflginfo,
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
totinf_indat

ddc_fitdat =  ODINcatVNuse.mkdatadiff(fitdatverplgr.rename (
       columns={'FactorV':'FactorO', 'FactorEst':'FactorV' }),
            ODINcatVNuse.fitgrpse,  ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)

totinf_fitdat = ODINcatVNuse.mkinfosums(ddc_fitdat,odindiffflginfo,
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
totinf_fitdat.groupby(["GeoInd"]).agg('sum')

totinf_indat.groupby(["GeoInd"]).agg('sum')


#let op dit kan niet, want hierover hebben we gesommeerd !
def permotief(totin,totfit,kflgs):
    renai = dict ( ( (flg,flg+"_i") for flg in kflgs) ) 
    agrpi=totin.groupby(["MotiefV","GeoInd"])[kflgs].agg('sum') /1e6
    agrpi['Gemafst_i'] = agrpi ['FactorKm'] /agrpi ['FactorV'] 

    renaf = dict ( ( (flg,flg+"_f") for flg in kflgs) )
    agrpf=totfit.groupby(["MotiefV","GeoInd"])[kflgs].agg('sum') /1e6
    agrpf['Gemafst_f'] = agrpf ['FactorKm'] /agrpf ['FactorV'] 
    agrp=pd.merge(agrpi.reset_index().rename(columns=renai),  
                  agrpf.reset_index().rename(columns=renaf) )
    return agrp
motlst= permotief(totinf_indat,totinf_fitdat,ODINcatVNuse.kflgsflds)  
motlst.to_excel("../output/orif_permot.xlsx")


def woonbalans1(totin,kflgs):
    ause = totin[np.isin(totin['isnaarhuis'],[5,6]) &
                 np.isin(totin['MotiefV'],[1]) ].copy(deep=False)
    ause['Ri2']=np.where(ause['isnaarhuis']==5, 
                        np.where(ause['GeoInd']=='AankPC', 'actzijde' , 'huiszijde' ),
                        np.where(ause['GeoInd']!='AankPC', 'actzijde' , 'huiszijde' )    
                           )
    agrp = ause .groupby(['MotiefV','GeoInd','isnaarhuis'])[kflgs].agg('sum')    /1e6   
    agrp ['Gemafst'] = agrp ['FactorKm'] /agrp ['FactorV'] 
    agrp = agrp.reset_index()
#    agrp ['gcol'] = "Mot_"+ np.array(agrp ['MotiefV'] .astype('string'))+"_nh_"+ np.array(agrp ['isnaarhuis'] .astype('string'))
    agrp ['p1'] = "Mot_"
    agrp ['p2'] = "_nh_"
    mc= agrp['MotiefV'].astype(int).astype(str)
    nc= agrp['isnaarhuis'].astype(str)
    p1= agrp['p1'].astype(str)
    p2= agrp['p2'].astype(str)
    agrp ['gcol'] =p1 + mc +p2 + nc 
    agrp2 = agrp.pivot(index="GeoInd", columns="gcol", values="FactorV")
    return agrp2
woonbalans1(totinf_indat,ODINcatVNuse.kflgsflds)                       


def woonbalans(totin,kflgs):
    iso=False
    for mot in [1,7]:
        for vn in [5,6]:
            colnm = ('Mot_{}_nh_{}').format(mot,vn)
            fruse= totin [(totin['MotiefV']==mot )&(totin['isnaarhuis']==vn*1.0 )]
            fruse = fruse[['GeoInd','FactorV']].rename(columns={'FactorV': colnm})
            fruse=fruse.copy()
            if iso:
                outf=outf.merge(fruse)
            else:
                outf=fruse
            iso=True
    return outf
woonbalans(totinf_indat,ODINcatVNuse.kflgsflds)

woonbalans(totinf_fitdat,ODINcatVNuse.kflgsflds).reset_index() # .drop(columns='gcol')    


# +
def predictnewdistr(pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV,myskipPCMdf,pltgrps,puin,
                    myfitpara,predgrping):
    pu= puin.copy()
    mygeoschpc4all= mkgeoschparafr(pc4data,pc4grid,rudigrid,myKAfstV,pu)
    if predgrping == 'mxigrp':        
        mygeoschpc4i, geobingr = addmxibins(mygeoschpc4all,nmxibins_glb)
        mygeoschmxigrp = summmxigrp(mygeoschpc4i)
        mygeoschpc4i['PC4ori'] =mygeoschpc4i['PC4']
    elif predgrping == 'PC4':    
        mygeoschpc4i= mygeoschpc4all
        mygeoschpc4i['mxigrp'] =mygeoschpc4i['PC4']
    else:
        raise('predictnewdistr: unimplemented predgrping:'+predgrping)
    mygeoschmixpMotief= allmotmxicorrgrp( summmxigrp(mygeoschpc4i),
                    summmxicorrgrp(mygeoschpc4i,myskipPCMdf),np.max(odinverplgr['MotiefV']))
    myodinverplmxigr = odinmergemxi (odinverplgr ,mygeoschpc4i,grpexpcontrs)
    myxlatKAfstV=myKAfstV[['KAfstCluCode']].merge(inxlatKAfstV,how='left')
#    print (myxlatKAfstV)
    mydatverplgr = mkdfverplxypc4 (myodinverplmxigr ,fitgrps,'Motief en isnaarhuis',
                                myKAfstV,xlatKAfstV,mygeoschmixpMotief,2.0).merge(myKAfstV,how='left')
    
    cut2i=  choose_cutoff(mydatverplgr,pltgrps,False,0,'mxigrp',pu)  
#deze dus niet    myfitpara= fit_cat_parameters(cut2i,mydatverplgr,pltgrps,pu)
    myfitverplgr = predict_values(cut2i,mydatverplgr,pltgrps,myfitpara,pu,False)

    rdf=calcchidgrp(myfitverplgr)
    chisq= np.sum(rdf['chisq'].reset_index().iloc[:,1])
    return(myfitverplgr)

rdf00=predictnewdistr (cbspc4data,pc4inwgcache,rudifungcache,useKAfstVQ,xlatKAfstV,
                skipPCMdf,fitgrps,expdefs,fitpara,'mxigrp')
rdf00
#to change for PC4 /geo comparison: use something like indatverplpc4gr instead of  myodinverplmxigr 

# +
def runexperiment(expname,incache0,mult,fitp,myuseKAfst,predgrping):
    print("runexperiment: start processing "+expname)
    incache=dict()
    for fld in [3,5]:
        incache[fld]=np.where(np.isnan( incache0[fld]),0, incache0[fld])
    fname = "../intermediate/addgrds/"+expname+'.tif';
    ogrid= rasterio.open(fname)
    addcache = getcachedgrids(ogrid)

    mycache=dict()
    mycache[3] = incache[3] + mult*addcache[3]
    mycache[5] = incache[5] + mult * (addcache[3] + addcache[5])
    mycache[5]= np.where(mycache[5] < mycache[3] , mycache[3],  mycache[5])    
    ogrid.close()

#gebuik de parameters       
    rdf=predictnewdistr(cbspc4data,pc4inwgcache,mycache,myuseKAfst,xlatKAfstV,
                skipPCMdf,fitgrps, expdefs,fitp,predgrping)
    return rdf

rdf01=runexperiment('e0903a___swap_0010_02500',rudifungcache,1,fitpara,useKAfstVQ,'mxigrp')
# -

globset="e0904a"
flst = glob.glob ("../intermediate/addgrds/"+globset+"*[a-z]_00*.tif")
elst = list(re.sub(".tif$",'',re.sub('^.*/','',f) ) for f in flst) 
elst


def grosumm(dfm,lbl,myuseKAfstV,normfr):
    dfm.reset_index().to_pickle("../output/fitdf_%s.pd"%(lbl))
    mymaskKAfstV= list(myuseKAfstV['KAfstCluCode'])
    if lbl=='brondat':
        dfmu=dfm[np.isin(dfm['KAfstCluCode'],mymaskKAfstV)].copy (deep=False)
    else:
        dfmu=dfm.rename (columns={'FactorV':'FactorO', 'FactorEst':'FactorV' })
    ddc_fitdat =  ODINcatVNuse.mkdatadiff(dfmu, ODINcatVNuse.fitgrpse,  ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)

    # myodinverplflgs / myodindiffflginfo kunnen ook buiten loop worden berekend, maar dit borgt consisitente
    # voor relatief weinig extra rekentijd
    myodinverplflgs =ODINcatVNuse.odinverplflgs_o[np.isin(
         ODINcatVNuse.odinverplflgs_o['KAfstCluCode'],mymaskKAfstV)].copy (deep=False)
    myodindiffflginfo= ODINcatVNuse.convert_diffgrpsidat(myodinverplflgs,
                ODINcatVNuse.fitgrpse,[],ODINcatVNuse.kflgsflds, [],"_c",ODINcatVNuse.landcod,False)
    totinf_fitdat = ODINcatVNuse.mkinfosums(ddc_fitdat,myodindiffflginfo,                                            
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
    rv =totinf_fitdat.groupby(["GeoInd"]).agg('sum')
    wb =woonbalans(totinf_fitdat,ODINcatVNuse.kflgsflds).groupby(["GeoInd"]).agg('sum')
    rv=rv.join(wb)
#    rv=rv.set_index(['GeoInd'])
    if (len (normfr) >0):
        rv = rv/ normfr
        rv['label']=lbl
    return rv 
gs00=grosumm(rdf00,"orig",useKAfstVQ,[])
#print(gs00)
gs00T = grosumm(rdf00,"origchk",useKAfstVQ,gs00)
gs00T

rdf00PC4=predictnewdistr (cbspc4data,pc4inwgcache,rudifungcache,useKAfstVQ,xlatKAfstV,
                skipPCMdf,fitgrps,expdefs,fitpara,'PC4')
gs00PC4 = grosumm(rdf00,"origPC4chk",useKAfstVQ,gs00)
gs00PC4


def grosres (explst,incache0,mult,fitp,oridat,myuseKAfst,setname,predgrping):
    rdf00N=predictnewdistr (cbspc4data,pc4inwgcache,rudifungcache,myuseKAfst,myuseKAfst,
                skipPCMdf,fitgrps,expdefs,fitpara,predgrping)
    gs00N = grosumm(rdf00N,"orig",myuseKAfst,[])
    st = ( grosumm(runexperiment(exp,incache0,mult,fitp,myuseKAfst,predgrping),
                   exp,myuseKAfst ,gs00N)  for exp in explst )
    st = pd.concat (st)
    dto= grosumm(oridat,'brondat',myuseKAfst ,gs00N)
    #print(dto)
    st=st.append(dto)
    st.reset_index().to_excel("../output/fitrelres_"+setname+".xlsx")
    return st
stQ = grosres (elst[0:3],rudifungcache,1,fitpara, fitdatverplgr,
               useKAfstVQ,'Dbg01Q-'+globset,'mxigrp')
stQ

stQ

     

# +
#check eens alles
#stQa = grosres (elst,rudifungcache,1,fitpara, fitdatverplgr,useKAfstVQ,'DBgf01Q-'+globset)
#stQa
# -
print("Finished")



