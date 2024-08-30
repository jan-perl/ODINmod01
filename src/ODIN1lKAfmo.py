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

#rudifunset
Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
Rf_net_buurt.reset_index(inplace=True,drop=True)
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

useKAfstV=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
xlatKAfstV=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")

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
geoschpc4 = geoschpc4all

#kijk even naar sommen
geoschpc4.groupby(['KAfstCluCode','MaxAfst']).agg('sum')

geoschpc4

useKAfstVland = useKAfstV [useKAfstV['MaxAfst']==0]
geoschpc4land=mkgeoschparafr(cbspc4data,pc4inwgcache,rudifungcache,useKAfstVland,expdefs)
geoschpc4land

odinverplgr=pd.read_pickle("../intermediate/ODINcatVN01db.pkl")
def deffactorv(rv):
    rv['FactorV'] = np.where ((rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ,
               np.nan,rv['FactorVGen'] + 0* rv['FactorVSpec'] )
deffactorv(odinverplgr)


# +
#maak 2 kolommen met totalen aankomst en vertrek (alle categorrieen)
#worden als kolommen aan addf geoschpc4land toegevoegd per PC
#is eigenlijk alleen van belang voor diagnostische plots
#en totalen hannen ook uit aggregaten gehaald kunnen wornde
#TODO cleanup

def mkvannaarcol(rv,verpldf,xvarPC):
    dfvrecs = verpldf [verpldf ['GeoInd'] == xvarPC]
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
def loglogregrplot(indfo,xcol,ycol):
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
    seaborn.lineplot(data=indf,x=xcol,y='Predict', color='g',ax=ax)
    seaborn.lineplot(data=indf,x=xcol,y='Predictu', color='r',ax=ax)
    seaborn.lineplot(data=indf,x=xcol,y='Predictl', color='r',ax=ax)
    seaborn.scatterplot(data=indf,x=xcol,y=ycol,ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    print(fig)
    return model

#een replicatie is genoeg en verwijder NAs
geoschpc4r1=geoschpc4land[(  ~ np.isnan(geoschpc4land['M_LW_AL'])) & 
                     ( ~ np.isnan(geoschpc4land['aantal_inwoners'])) &
                     ( geoschpc4land['aantal_inwoners'] != ODINmissint )]

#geoschpc4r2= cbspc4data[['postcode4','oppervlak'] ] .merge (geoschpc4r1 ,left_on=('postcode4'), right_on = ('PC4') )
geoschpc4r2 = geoschpc4r1

buurtlogmod= loglogregrplot(geoschpc4r2,'M_LW_AL','aantal_inwoners' )

# +
#enigszine onverwacht komnt hier dezelfde relatie uit al in wijken en buurten
#wat dit betreft lijken postcodes dus homogeen
# -

buurtlogmod= loglogregrplot(geoschpc4r2,'oppervlak','aantal_inwoners' )

buurtlogmod= loglogregrplot(geoschpc4r2,'oppervlak','M_LW_AL' )

# +
#Poging2: uit behouwings ratio
geoschpc4r2['m2perinw']= geoschpc4r2['M_LW_AL']/geoschpc4r2['aantal_inwoners' ]
geoschpc4r2['inwperare']= 10000*geoschpc4r2['aantal_inwoners' ]/ geoschpc4r2['M_LW_AL']
geoschpc4r2['pctow']= geoschpc4r2['M_LW_AL']/geoschpc4r2['oppervlak' ]
buurtlogmod= loglogregrplot(geoschpc4r2,'pctow','inwperare' )

#deze ratio zouden we ook op buurtniveau kunnen berekenen
# -
geogeenwerk = geoschpc4r2 [geoschpc4r2 ['M_LW_AL'] > 20 * geoschpc4r2 ['M_LO_AL']] 
geogeenwerk


alleenwoonexp=loglogregrplot(geogeenwerk,'M_LW_AL','TotaalVAankPC' )


# +
def laagwoonexp(model,LW_OWin):
    rv = np.exp(model.coef_[0]*np.log(LW_OWin) + model.intercept_ )
    return rv
    
#print( np.array ([ laagwoonexp(alleenwoonexp,geogeenwerk['M_LW_OW']),geogeenwerk['TotaalVAankPC'] ] ).T )


# -

geoschpc4r2 ['VminWAankPC']=  geoschpc4r2 ['TotaalVAankPC'] -laagwoonexp(alleenwoonexp,geoschpc4r2 ['M_LW_AL'])
geogeenwoon = geoschpc4r2 [(geoschpc4r2 ['M_LW_AL'] *5 < geoschpc4r2 ['M_LO_AL'] ) & (geoschpc4r2 ['VminWAankPC'] >0) ] 
#geogeenwoon

geenwoonexpC= loglogregrplot(geogeenwoon,'M_LO_AL','VminWAankPC')
geenwoonexpA= loglogregrplot(geogeenwoon,'M_LO_AL','TotaalVAankPC')


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
    pstatsc=pstatsc.merge(vollandrecs,how='right')
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
indatverplgr = mkdfverplxypc4 (odinverplgr ,fitgrps,'Motief en isnaarhuis',
                                useKAfstV,xlatKAfstV,geoschpc4,2.0)
indatverplgr
# +
#originele code had copy. Kost veel geheugen en tijd
#daarom verder met kolommen met een F_ (filtered)

#oude versie: ieder record fit naar ofwel ALsafe of naar osafe, of nergens heen

def choose_cutoffold(indat,pltgrps,hasfitted,prevrres):
    outframe=indat[['PC4','GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    if hasfitted:
        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 5.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
    else:
        outframe['ALsafe'] =True
        outframe['osafe'] = True
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            for okey in ('OW','OO'):
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe['ALsafe'] = outframe['ALsafe'] & (indat[colnam] > 0.2  * indat[colnamAL])
                outframe['osafe']  = outframe['osafe']  & (indat[colnam] < 0.01 * indat[colnamAL])
        outframe['ALsafe'] = outframe['ALsafe'] | (outframe['MaxAfst']==0)
        outframe['osafe'] = outframe['osafe'] & (outframe['MaxAfst']!=0)
    if 1==1:
        outframe['FactorVF'] =indat['FactorV'] * ((outframe['ALsafe'] |outframe['osafe'] ).astype(int))
        for lkey in ('LW','LO'):
            colnamAL ="M_"+ lkey +"_AL"
            colnamALo="F_"+ lkey +"_AL"
            outframe[colnamALo] = indat[colnamAL] * ((outframe['ALsafe']).astype(int))
            for okey in ('OW','OO','OM','OA'):
                colnam ="M_"+ lkey +"_" + okey
                colnamo="F_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe[colnamo] = indat[colnam] * ((outframe['osafe']).astype(int))
#    outframe['ALmult'] = ( (outframe['ALsafe']==False).astype(int))
    return outframe

cut2=  choose_cutoffold(indatverplgr,fitgrps,False,0)   
cut2


# +
#todo
#fit tov fractie (l komt uit data FactorVL)
#waarde:  p=1/(1/l + 1/f) -> f= 1/ (1/p - 1/l) -> divergeert dus alleen als w>.1, anders 0
#gewicht: w= (p * (1/p - 1/l))** (+ pow+1)   -> aparte kolom

# +
#originele code had copy. Kost veel geheugen en tijd
#daarom verder met kolommen met een F_ (filtered)

def choose_cutoffnw(indat,pltgrps,hasfitted,prevrres,pu):
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
        wval1= indat[recisAL] [['FactorV','PC4','GeoInd'] +pltgrps].copy()
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

def choose_cutoff(indat,pltgrps,hasfitted,prevrres,curvpwr):
    if True:
        return choose_cutoffnw(indat,pltgrps,hasfitted,prevrres,curvpwr)
    else:
        return choose_cutoffold(indat,pltgrps,hasfitted,prevrres)


cut2=  choose_cutoff(indatverplgr,fitgrps,False,0,expdefs)   
#cut2
# -

def fitinddiag(fitdf,motiefc,naarhuisc,geoindex,pu):
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
        print(seldf.sort_values(by=['PC4','KAfstCluCode'])[['FactorVFo','FactorVP']] )
    fig, ax = plt.subplots()    
    seaborn.scatterplot(data=plmelt,x="FactorVPrel",y="vals", hue='cols', ax=ax)
    ax.set_xscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fitinddiag(cut2,10,5,'VertPC',expdefs)    


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
    
    
def dofitdatverplgr(indf,topreddf,pltgrp,pu):
    curvpwr = pu['CP']
#    indf = indf[(indf['MaxAfst']!=95.0) & (indf[pltgrp]<3) ]
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
    s2ch= np.where((s2==0),np.where(outdf['MaxAfst']==0, s2al,0), 
                           np.where((s2al==0),s2,
                           np.power (np.power(s2,-curvpwr) + np.power(s2al,-curvpwr), -curvpwr )) )
    if (debug):
        print (s2ch)
    outdf['FactorEst'] = s2ch
    outdf['DiffEst'] =outdf['FactorV']-s2ch
    return(outdf)


fitdatverplgr = dofitdatverplgr(cut2,indatverplgr,fitgrps,expdefs)
seaborn.scatterplot(data=fitdatverplgr,x="FactorEst",y="DiffEst",hue="GeoInd")
# -

cut3=  choose_cutoff(indatverplgr,fitgrps,True,fitdatverplgr,expdefs)  
cut3=  choose_cutoffold(indatverplgr,fitgrps,True,fitdatverplgr)  
#cut3

# +
#fitinddiag(cut3,10,5,'VertPC',p_CP)    
# -

#voor de time being, overschrijf de vorige selectie gegevens
for r in range(0):
    cut3=  choose_cutoff(indatverplgr,fitgrps,True,fitdatverplgr,expdefs)  
    fitdatverplgr = dofitdatverplgr(cut3,indatverplgr,fitgrps,expdefs)
seaborn.scatterplot(data=fitdatverplgr,x="FactorEst",y="DiffEst",hue="GeoInd")

fitdatverplgr["x_LM_AL"] = fitdatverplgr["M_LW_AL"] * fitdatverplgr["M_LO_AL"]
seaborn.scatterplot(data=fitdatverplgr,x="x_LM_AL",y="DiffEst",hue="GeoInd")

seaborn.scatterplot(data=fitdatverplgr,x="M_LO_AL",y="DiffEst",hue="GeoInd")

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

fitdatverplgr[fitdatverplgr['MaxAfst']==0]

paratab=fitdatverplgr[fitdatverplgr['MaxAfst']==0].groupby (['MotiefV','isnaarhuis','GeoInd']).agg('mean')
paratab.to_excel("../output/fitparatab1.xlsx")
paratab

fitdatverplgr.groupby (['MotiefV','isnaarhuis','GeoInd','MaxAfst']).agg('mean')


def pltmotdistgrp (mydati,horax):
    mydat=pd.DataFrame(mydati)    
    opdel=['MaxAfst','GeoInd','MotiefV','GrpExpl']
#    mydat['FactorEst2'] = np.where(mydat['FactorEstNAL']==0,0, 
#                                 1/ (1/mydat['FactorEstAL'] + 1  /mydat['FactorEstNAL'] ) )
    fsel=['FactorEst','FactorV', 'FactorEstNAL','FactorEstAL']
    rv2= mydat.groupby (opdel)[fsel].agg(['sum']).reset_index()
    rv2.columns=opdel+fsel
    if(horax=='MaxAfst'):
        limcat=2e9
    else:
        limcat=1e9
    bigmot= rv2[(rv2['MaxAfst']==0) & (rv2['FactorV']>limcat)  ].groupby('GrpExpl').agg(['count']).reset_index()
    print(bigmot)
    rv2['MaxAfst']=np.where(rv2['MaxAfst']==0 ,100,rv2['MaxAfst'])
    rv2['MaxAfst']=rv2['MaxAfst'] * np.where(rv2['GeoInd']=='AankPC',1,1.02)
    rv2['Qafst']=1/(1/(rv2['MaxAfst']  *0+1e10) +1/ (np.power(rv2['MaxAfst'] ,1.8) *2e8 ))
    rv2['linpmax'] = rv2['FactorEstNAL']/ rv2['FactorEstAL']
    rv2['linpch']= rv2['FactorEst']/ rv2['FactorEstAL']
    rv2['drat']= rv2['FactorV']/ rv2['FactorEstAL']

    rvs = rv2[np.isin(rv2['GrpExpl'],bigmot['GrpExpl'])]
#    rv2['MotiefV']=rv2['MotiefV'].astype(int).astype(str)
    fig, ax = plt.subplots()
    if(horax=='MaxAfst'):
        seaborn.scatterplot(data=rvs,x=horax,y='FactorV',hue='GrpExpl',ax=ax)
        seaborn.lineplot(data=rvs,x=horax,   y='FactorEst',hue='GrpExpl',ax=ax)
        seaborn.lineplot(data=rvs,x=horax,   y='Qafst',ax=ax)
    elif(horax=='linpmax'):
        seaborn.scatterplot(data=rvs,x=horax,y='drat',hue='GrpExpl',ax=ax)
        seaborn.lineplot(data=rvs,x=horax,   y='linpch',hue='GrpExpl',ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return (rv2)
ov=pltmotdistgrp(fitdatverplgr,'MaxAfst')

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==1],'MaxAfst')

ov=pltmotdistgrp(fitdatverplgr,'linpmax')


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
def trypowerland (pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV,pltgrps,puin,v1i,v1v,v2i,v2v):
    pu= puin.copy()
    pu[v1i]=v1v
    pu[v2i]=v2v
    print(pu)
    mygeoschpc4= mkgeoschparafr(pc4data,pc4grid,rudigrid,myKAfstV,pu)
    myxlatKAfstV=myKAfstV[['KAfstCluCode']].merge(inxlatKAfstV,how='left')
#    print (myxlatKAfstV)
    mydatverplgr = mkdfverplxypc4 (odinverplgr ,fitgrps,'Motief en isnaarhuis',
                                myKAfstV,xlatKAfstV,mygeoschpc4,2.0)
    
    cut2i=  choose_cutoff(mydatverplgr,pltgrps,False,0,pu)  
    myfitverplgr = dofitdatverplgr(cut2i,mydatverplgr,pltgrps,pu)
    for r in range(2):
        cut3i=  choose_cutoff(mydatverplgr,pltgrps,True,myfitverplgr,pu) 
        myfitverplgr = dofitdatverplgr(cut3i,mydatverplgr,pltgrps,pu)
    rdf=calcchidgrp(myfitverplgr)
    return(np.sum(rdf['chisq'].reset_index().iloc[:,1]))
    
#rv=trypowerland(cbspc4data,pc4inwgrid,rudifungrid,useKAfstVland,xlatKAfstV,1.3,1.0,2.0)
rv=trypowerland(cbspc4data,pc4inwgcache,rudifungcache,useKAfstVland,xlatKAfstV,fitgrps,
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
chitries= chisqsampler (cbspc4data,pc4inwgcache,rudifungcache,useKAfstVland,xlatKAfstV,fitgrps)    


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

chitries

seaborn.scatterplot(data=fitdatverplgr[fitdatverplgr['MaxAfst']==0],x="FactorEst",y="DiffEst",hue="GeoInd")

pllanddiff= cbspc4data[['postcode4']].merge(fitdatverplgr[(fitdatverplgr['MaxAfst']==0) 
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
largestdiffsPC (cbspc4data,fitdatverplgr)

# +
#inspectie
#grote onderschatters landelijK; winkelcentra, schiphol, en mindere mate universiteit

#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(3511,3512,3584 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(6511,6525 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(2513 ,2333)))]
chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(7511) ))]
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
# -


