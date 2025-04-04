# -*- coding: utf-8 -*-
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
#maak ODIN van/naar categorieen
#en save naar pkl in intermediates
# -

# todo
# actieve modes per afstandsklasse en type
# gemiddelde afstand per afstandsklasse en type
# active mode kms en passive mode kms per PC naar & van (let op: dubbeltelling)
# referentie reiskms actieve modes
# vergelijking co2 uitstoot active modes met kentallen


# +
#todo
#OA klasse ook via convolutie -> voor bij grafieken
#vannaar per postcode combinaite maken voor geselecteerde postcodes
#  voor visualisatie, uit convolutie. Exporteer alleen !=0, bijvoorbeeld PCs in Utrecht
#draai minimaal 1 maal rasteruts in docker container om geopandas te laden

# +
#splits code op; DIT DEEL is:

#- analyse top PC4 per Motief voor 2 ritten op dag en niet woon kant
#- t.o.v. gebieds oppervlak
#- bij top PC4 maak annotatie naar 1 of meerdere PC6 mogelijk
# maak ook afstandsklassen tabel (met index per indeling naam) en schrijf naar pkl

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
import io   

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
#plot_crs=3857
#plot_crs=28992
if 1==1:
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['min_zoom']= 0
    prov0['max_zoom'] =12
    print (prov0)

pland= cbspc4data.plot(alpha=0.4)
cx.add_basemap(pland, source= prov0,crs=cbspc4data.crs)

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot()
cx.add_basemap(phtn, source= prov0)

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']==3995)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot()
cx.add_basemap(phtn, source= prov0)

pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid= rasterio.open(pc4tifname)


def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
pc4inwgcache = getcachedgrids(pc4inwgrid)

# nu nog MXI overzetten naar PC4 ter referentie




#import ODiN2pd
import ODiN2readpkl



ODiN2readpkl.allodinyr.dtypes


def chklevstat(df,grpby,dbk,vnamcol,myNiveau):
    chkcols = dbk [ (dbk.Niveau == myNiveau) & ~ ( dbk[vnamcol].isin( excols) )]
    for chkcol in chkcols[vnamcol]:
        nonadf= df[~ ( df[chkcol].isna() |  df[grpby].isna() ) ]
#        print (chkcol)
#        print (nonadf['RitID'])
#        sdvals= nonadf. groupby([grpby]) [chkcol].std_zero()
        sdvals= nonadf. groupby([grpby]) [chkcol].agg(['min','max']).reset_index()
#        print(sdvals)
        sdrng = (sdvals.iloc[:,2] != sdvals.iloc[:,1]).replace({True: 1, False: 0})
#        print(sdrng)
        maxsd=sdrng.max()
        #als alle standaard deviaties 0 zijm, zijn de waarden homogeen in de groep
        if maxsd !=0:
            print (chkcol,maxsd)


dbk_2022 = ODiN2readpkl.dbk_allyr
dbk_2022_cols = dbk_2022 [~ dbk_2022.Variabele_naam_ODiN_2022.isna()]
dbk_2022_cols [ dbk_2022_cols.Niveau.isna()]

# +
largranval =ODiN2readpkl.largranval 

specvaltab = ODiN2readpkl.mkspecvaltab(dbk_2022)
specvaltab


# -

def intcast0nan (ser):
    return np.where(np.isnan(ser),0,ser.astype(int))
def _onlyverpl(df):
    return  df [(df['Verpl']==1 ) & (df['AankPC'] > 500) & (df['VertPC'] > 500)  ] .copy(deep=False)  
allodinyr=_onlyverpl(ODiN2readpkl.allodinyr)
#blijf float gebruiken: dan kun je op is isnan selectern na merge
#allodinyr['VertPC']=intcast0nan ( allodinyr['VertPC'])
#allodinyr['AankPC']=intcast0nan ( allodinyr['AankPC'])
#allodinyr['MotiefV']=intcast0nan ( allodinyr['MotiefV'])
len(allodinyr.index)


#voor normalisatie van Factor V per jaar
def getjaren(indf):
    ntab = indf.groupby('Jaar')[['Jaar']].agg('count')
    return ntab
welkejaren = getjaren (allodinyr).index
aantaljaren=len(welkejaren)
print(aantaljaren)


# +
#nu ODIN ranges opzetten
#we veranderen NIETS aan odin data
#wel presenteren we het steeds als cumulatieve sommen tot een bepaalde bin

# +
#maak summmatie tabellen (joinbaar)
def pickAnrs(myspecvals,xvar,newmaxbndidx):
    orimin=1    
    allKAfstV=myspecvals [ (myspecvals ['Code'] !=largranval) & 
                           (myspecvals ['Code'] !='<missing>') & 
#                           ((myspecvals ['Code']).astype('int64') >= orimin) & 
                           (myspecvals ['Variabele_naam'] ==xvar) ][['Code','Code_label']].\
                           copy().reset_index(drop=True)    
    allKAfstV['Code']= allKAfstV['Code'].astype('int64')
    orimax=np.max(allKAfstV['Code'])
    allKAfstV['MaxAfst'] = list(re.sub(r',','.',re.sub(r' km.*$','',re.sub(r'^.* tot ','',s)  ))\
                                for s in allKAfstV['Code_label'] )
    mindispdist=0.1
    allKAfstV.loc[0,'MaxAfst']=mindispdist
    allKAfstV['MaxAfst'] =allKAfstV['MaxAfst'].astype('float32')
    lastval = orimax
    oarr=np.array(allKAfstV['Code'])
    if 1==1:
        oxlatKAfstV = allKAfstV[allKAfstV['Code']>=orimin] [['Code']] .rename( 
                                 columns={'Code':'KAfstV'})
        oxlatKAfstV['KAfstCluCode'] =  orimax
        xlatKAfstV = oxlatKAfstV.copy()
        for n in newmaxbndidx:     
            if n>0:
                toapp = oxlatKAfstV[oxlatKAfstV['KAfstV']<=n].copy()
                toapp ['KAfstCluCode'] =  n
                xlatKAfstV= xlatKAfstV.append(toapp) .reset_index(drop=True)                      
    else:
        for ir in range(len(allKAfstV)):
            i= len(allKAfstV)-ir-1
            if oarr[i] in newmaxbndidx or oarr[i] ==0:
                lastval = oarr[i] 
            oarr[i] = lastval
    allKAfstV['KAfstCluCode']=oarr
#    print(allKAfstV)   
    selKAfstV= allKAfstV.iloc[ newmaxbndidx, ]
    selKAfstV.loc[ orimax , "MaxAfst"]  =0    
    return (  [  selKAfstV[['KAfstCluCode',"MaxAfst"]] , xlatKAfstV ])

useKAfstVo,xlatKAfstVo  = pickAnrs (specvaltab,'KAfstV',[1,2,3,4,5,6,7,8,-1] )
print(xlatKAfstVo)   
print(useKAfstVo)   
# -

kvallsel = list(i+1 for i in  range(np.max(useKAfstVo['KAfstCluCode']-1))) +[-1]
print(kvallsel)
#noot: in het wegschijven is het geen probleem om alle afstanden te doen
useKAfstV,xlatKAfstV= pickAnrs (specvaltab,'KAfstV',kvallsel)
useKAfstVall,xlatKAfstVall  = pickAnrs (specvaltab,'KAfstV',kvallsel)
useKAfstVall.iloc[-1,1] = 200
useKAfstVall['MinAfst'] = useKAfstVall['MaxAfst'] .shift(1,fill_value=0.0)
useKAfstVall['AvgAfst'] = (useKAfstVall['MaxAfst'] + useKAfstVall['MinAfst'])/2
useKAfstVall= useKAfstVall.rename(columns={'KAfstCluCode':'KAfstV'})
print(useKAfstVall)
#gebruikt voor ritkms

# +
#maak 2 kolommen met totalen aankomst en vertrek (alle categorrieen)
#worden als kolommen aan addf geoschpc4land toegevoegd per PC
#is eigenlijk alleen van belang voor diagnostische plots
#en totalen hannen ook uit aggregaten gehaald kunnen wornde
#TODO cleanup
def arrvannaarcol(rv,verpldf,xvarPC):
    dfvrecs = verpldf [verpldf['Verpl']==1]
    gcols= [xvarPC]
    pstats = dfvrecs[gcols+['FactorV']].groupby(gcols).sum().reset_index()
    outvar='TotaalV'+xvarPC
    pstats=pstats.rename(columns={xvarPC:'PC4'})
#    print(pstats)
    addfj = rv.merge(pstats,how='outer')
    rv[outvar] = addfj['FactorV']
    print(len(rv))
    
def mktotalpc4cols(allpc4,verpldf): 
    rv=allpc4[['postcode4','aantal_inwoners','oppervlak']].rename(columns={'postcode4':'PC4'} ) .copy()
    arrvannaarcol(rv,verpldf,'AankPC')
    arrvannaarcol(rv,verpldf,'VertPC')
    return rv

odinschpc4= mktotalpc4cols(cbspc4data,allodinyr) 


# +
#geoschpc4 is een mooi dataframe met generieke woon en werk parameters
#Er is nog wel een mogelijk verzadigings effect daar waar de waarden voor
#grotere afstanden die van de landelijke waarden benaderen
# +
#Maak tabellen voor karakteristieke gemeente en provincie per  PC

def gebpc4fracts1(df,myspecvals,av,grp):
    dfvrecs = df [(df['Verpl']==1 ) & (df['AankPC'] > 500) & (df['VertPC'] > 500)  ]   
    dfvrecn=dfvrecs.rename(columns={av+'PC':'PC4',av+grp:'GrpVal','FactorV':'wgt'}) 
    summdf=dfvrecn.groupby(['PC4','GrpVal']) [['wgt']].agg('sum').reset_index()
    summdf['GrpTyp']=grp
    if grp=='Prov':
            explhere1 = myspecvals [myspecvals['Variabele_naam'] == av+grp].copy()
            explhere1['Code'] = pd.to_numeric(explhere1['Code'],errors='coerce')
        #    print(explhere1)
            summdf=summdf.merge(explhere1,left_on='GrpVal', right_on='Code', how='left').rename(
               columns={'Code_label':'GrpV_label'}) 
            summdf=summdf.drop(columns=['Variabele_naam','Code'])
    else:
            summdf['GrpV_label']='Gem xx'
    return summdf

def mkgebpc4fracts (df,myspecvals, myinfogrps ):
    fr2= (gebpc4fracts1(df,myspecvals,av,grp)
            for grp in myinfogrps for av in ['Aank','Vert'] )
    blk4=pd.concat(fr2).reset_index()
#    print(blk4)
    rv= blk4.groupby(['GrpTyp','PC4','GrpVal','GrpV_label']) [['wgt']].agg('sum').reset_index()
    rv= rv.sort_values(by='wgt')
#    print(rv)
    normdf= rv.rename(columns={'wgt':'wgtsum'}). groupby(['GrpTyp','PC4']) [['wgtsum'
                                                ]].agg('sum').reset_index()
#    print(normdf)
    rv =rv.merge( normdf, how='left')
    rv['wgt'] = rv['wgt']  /rv['wgtsum']              
    rv= rv.drop(columns=['wgtsum']). groupby(['GrpTyp','PC4']) .agg('last').reset_index()
    
    return  rv

PC4naargemprov = mkgebpc4fracts (allodinyr ,specvaltab,['Gem','Prov'])    
PC4kargemprov =  PC4naargemprov.pivot(index="PC4", columns="GrpTyp", values="GrpV_label")
print(PC4naargemprov.sort_values(by='wgt').head(10))
PC4kargemprov 
# -


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
geoschpc4r1=odinschpc4[ 
                     ( ~ np.isnan(odinschpc4['aantal_inwoners'])) &
                     ( odinschpc4['aantal_inwoners'] != ODINmissint )]

geoschpc4r2= cbspc4data[['postcode4','omgevingsadressendichtheid' ]] .merge (geoschpc4r1 ,left_on=('postcode4'), right_on = ('PC4') )
#geoschpc4r2=geoschpc4r1
#buurtlogmod= loglogregrplot(geoschpc4r2,'M_LW_AL','aantal_inwoners' )

# +
#enigszine onverwacht komnt hier dezelfde relatie uit al in wijken en buurten
#wat dit betreft lijken postcodes dus homogeen
# -

geoschpc4r2['adrschat'] = geoschpc4r2['omgevingsadressendichtheid'] * geoschpc4r2['oppervlak']*1e-6
buurtlogmod= loglogregrplot(geoschpc4r2,'adrschat','aantal_inwoners' )

geoschpc4r2[geoschpc4r2['adrschat'] >1e5]

geoschpc4r2[geoschpc4r2['oppervlak'] >1e8]

geoschpc4r2[geoschpc4r2['adrschat'] <10]

buurtlogmod= loglogregrplot(geoschpc4r2,'oppervlak','aantal_inwoners' )


# +
#buurtlogmod= loglogregrplot(geoschpc4r2,'oppervlak','M_LW_AL' )

# +
#print(allodinyr2)
#allodinyr = allodinyr2

# +
#kijk eens naar totaal aantal verplaatsingen per persoon
#dan nog NIET met factoren vermenigvuldigen
def mkpltverplpp (df,collvar):
    dfvrecs = df [df['Verpl']==1]
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs.groupby(['Jaar', collvar]).count()[['VerplID']].reset_index()
    #print(pstats)
    phisto = pstats.groupby(['Jaar', 'VerplID']).count()[[collvar]].reset_index()
#    print(phisto)
    phisto[collvar] = phisto[collvar] *100.0/ len(dfvrecs.index)
    sns.catplot(data=phisto, x="VerplID", y=collvar, hue="Jaar", kind="bar")
    return(phisto)
   
    
#datpltverplpp = mkpltverplpp (ODiN2pd.df_2019)
datpltverplpp = mkpltverplpp (allodinyr,'OPID')

# +
isnhexpl = {7:"rondje huis",6: "naar huis" , 5:"van huis",4:"ronde elders" }

allodinyr['isnaarhuis'] =  np.where(allodinyr ['Doel'] ==1, 
                    np.where(allodinyr ['VertLoc']<=2 , 7,6),
                    np.where(allodinyr ['VertLoc']<=2 , 5,4 ) ) 
allodinyr['isnaarhuisexpl'] =  allodinyr['isnaarhuis'].astype(str) + " "+ allodinyr['isnaarhuis'].map(isnhexpl)


# -

def pcnietwoon(df):
#    oprecs = df [df['OP']==1]
    pstats = df [df['Verpl']==1] .groupby(['Jaar', 'OPID'])[['VerplID']].count().reset_index()
    pstats = pstats.rename(columns={'VerplID' : 'Nverplaats' } )
    df2= df.merge(pstats,how='left')
#    print(df2)
    rv = np.where ((df2['Nverplaats'] != 2 ) | (df2['Verpl']!=1), 0, 
                   np.where (df2['isnaarhuis'] == 6,df2['VertPC'] ,
                   np.where (df2['isnaarhuis'] == 5,df2['AankPC'] ,0) ))
    return rv
allodinyr['OthPC'] =pcnietwoon(allodinyr)

# +
some_string="""OthPC,MotiefV,Comment
1775,1,kassen werk/oppervl overschat
1779,1,kassen werk/oppervl overschat
2641,1,kassen werk/oppervl overschat
2651,1,kassen werk/oppervl overschat
2665,1,kassen werk/oppervl overschat
2671,1,kassen werk/oppervl overschat
2675,1,kassen werk/oppervl overschat
2676,1,kassen werk/oppervl overschat
2678,1,kassen werk/oppervl overschat
2681,1,kassen werk/oppervl overschat
2691,1,kassen werk/oppervl overschat
5928,1,logistiek werk/oppervl overschat
1118,8,binnenstad Adam
1118,11,binnenstad Adam
6525,6,radboud U
2333,1,werk Leiden
3011,1,werk rotterdam
3011,7,winkelen rotterdam
3511,1,werk utrecht
1334,7,winkel
1442,7,winkel
1703,7,winkel
1825,7,winkel
2321,7,winkel
2513,7,winkel
3562,7,winkel
3825,7,winkel
5038,7,winkel
8911,7,winkel
9712,7,winkel
7811,7,winkel
9723,7,winkel
3511,7,winkelen utrecht
3311,7,Binnenstad Dordrcht
3431,7,Nieuwegein City Plaza
3995,7,Houten Rond en Castellum
7311,7,Binnenstad Apeldoorn
7511,7,Binnenstad Enschede
9203,7,Drachten winkels
9401,7,Assen winkels
"""

#read CSV string into pandas DataFrame
highman = pd.read_csv(io.StringIO(some_string), sep=",")
highman


# +
def surplusPCmotief(allpc4,df,man,filtered,njaar):
    pc4data=allpc4[['postcode4','aantal_inwoners','oppervlak']].rename(columns={'postcode4':'OthPC'} ) .copy()
    pc4totaal = pc4data[['aantal_inwoners','oppervlak']].sum()
#       .reset_index().rename(
#          columns={'aantal_inwoners':'inwoners_totaal', 'oppervlak': 'oppervlak_totaal' } )
#    print(pc4totaal.oppervlak)
    allemotief = (df.groupby(['OthPC', 'MotiefV'])[['FactorV']].sum()/njaar).reset_index()
    allemotief = allemotief[allemotief['OthPC']!=0].merge(pc4data,how='left')
    motieftotaal = (allemotief.groupby([ 'MotiefV'])[['FactorV']].sum()).reset_index().rename(
        columns={'FactorV':'MotiefTot'} )
#    print(motieftotaal)
    allemotief = allemotief.merge(motieftotaal,how='left')
    allemotief['oppfrac']=  allemotief['MotiefTot'] * allemotief['oppervlak']/ pc4totaal.oppervlak
#    print(allemotief)
    allemotief['siggrens'] = 2e5+  .002 *  allemotief['MotiefTot'] + .2*allemotief['oppfrac']
    tehoog=abs(allemotief['FactorV'] - allemotief['oppfrac'])  >allemotief['siggrens'] 
    tehoog = tehoog | ( (allemotief['FactorV'] <.05 * allemotief['oppfrac']) | \
                      (allemotief['FactorV'] >100 * allemotief['oppfrac'])  ) & \
                      ( (allemotief['FactorV'] + allemotief['oppfrac'] ) >4e5 )
    man['flgman']=1.1
    allemotief=allemotief.merge(man,how='left')
    allemotief['isSpec']= tehoog | (False== np.isnan(allemotief['flgman'])) 
    if filtered:
        rv =allemotief[ allemotief['isSpec']]
    else:
        rv =allemotief
    return rv
                                                                                            
highpcs = surplusPCmotief(cbspc4data,allodinyr,highman,True,aantaljaren)
print(len(highpcs))
print( highpcs.groupby('MotiefV')[['OthPC']].count() )
print( highpcs.groupby('OthPC')[['MotiefV']].count() )
highpcs.to_excel("../output/highmotiefPCs.xlsx")


# +
def pcfactorgraph():
    allemotief = surplusPCmotief(cbspc4data,allodinyr,highman,False,aantaljaren)
    allemotief['FactorOppFact'] = allemotief['FactorV'] / (allemotief['oppfrac']+1e-16)
    allemotief = allemotief.sort_values(by=['isSpec','FactorOppFact'])
    allemotief['Deelfact'] =( allemotief['FactorV'] + allemotief['oppfrac']) .cumsum()
    allemotief['Deelfact'] = allemotief['Deelfact']/ max(allemotief['Deelfact'])    
    fig, ax = plt.subplots(figsize=(6, 4))
    seaborn.lineplot(data=allemotief,x='Deelfact',y='FactorOppFact', color='g',ax=ax)    
    seaborn.lineplot(data=allemotief,x='Deelfact',y=0.05, color='r',ax=ax)    
    seaborn.lineplot(data=allemotief,x='Deelfact',y=100, color='r',ax=ax)    
    seaborn.lineplot(data=allemotief,x='Deelfact',y=1, color='b',ax=ax)    
#    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('aantal reizen / oppervlak verdeling (woonfunc niet specifiek)')
    ax.set_xlabel('Fractie van alle reizen, opgedeeld in Gen/Spec')
    print(fig)
    
pcfactorgraph()


# -

def mkhighpcflgv1(df,excepts):
    excepts= excepts[['OthPC','MotiefV']].copy(deep=False)
    excepts['flg']=1.1
#    print(excepts)
    dfs= df[['MotiefV','AankPC','VertPC']]
    df2= dfs.merge (excepts.rename (columns={'OthPC':'AankPC'}),how='left').rename(columns={'flg':'AankFlg'})
    df2= df2.merge (excepts.rename (columns={'OthPC':'VertPC'}),how='left').rename(columns={'flg':'VertFlg'})
    rv= ( np.isnan(df2['AankFlg']) & np.isnan(df2['VertFlg']) ) == False
    return rv
if False:
    allodinyr['HighPCfls'] =mkhighpcflgv1(allodinyr,highpcs)
    allodinyr[ (allodinyr['HighPCfls']) ==True ] [['AankPC','VertPC','MotiefV','FactorV']]


def mkhighpcflg(df,excepts):    
    nflg = np.array ( excepts['OthPC'] *100 +  excepts['MotiefV'] )
    c1 = np.isin(df ['AankPC'] *100 +  df['MotiefV'] ,nflg) 
    c2 = np.isin(df ['VertPC'] *100 +  df['MotiefV'] ,nflg) 
    rv= c1| c2
    return rv
allodinyr['HighPCfls'] =mkhighpcflg(allodinyr,highpcs)
allodinyr[ (allodinyr['HighPCfls']) ==True ] [['AankPC','VertPC','MotiefV','FactorV']]

xlatKAfstV


#add column 'GrpExpl' to pstatsc
def addgrpexpl(indf,myspecvals,col0,col1,grp1map):
    if 1==1:
        explhere = myspecvals [myspecvals['Variabele_naam'] == col0].copy()
        explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
#        print(explhere)
        outdf=indf.merge(explhere,left_on=col0, right_on='Code', how='left')    
        outdf['GrpExpl'] = outdf[col0].astype(str) +  " "+ outdf[col1].astype(str) + \
              " : " + outdf['Code_label'] + " "+ outdf[col1].map(grp1map)
        outdf= outdf.drop(columns=['Code','Code_label','Variabele_naam']) 
    else:
        outdf=indf.copy(deep=False)
        outdf['GrpExpl']=''               
    return outdf


# +
#other file merge geo
#other file addparsrecs
#synchronize with ODINcatVNuse
FactorVincols=['FactorVGen','FactorVSpec','FactorVGenActive','FactorVSpecActive']
#dan een dataframe dat
#2) per lengteschaal, 1 PC (van of naar en anderegroepen (maar bijv ook Motief ODin data verzamelt)

def mkdfverplxypc4d1 (df,myspecvals,xvarPC,pltgrps,grp1map,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4):
    debug=True
    dfvrecs = df [(df['Verpl']==1 ) & (df[xvarPC] > 500)  ].copy(deep=False)   
#    oprecs = df [df['OP']==1]
    gcols=pltgrps+ ['KAfstV',xvarPC]
    #sommeer per groep per oorspronkelijke KAfstV, maar niet voor internationaal
    pstats = dfvrecs[gcols+FactorVincols].groupby(gcols).sum().reset_index()
    if debug:
        print( ( "oorspr lengte groepen", len(pstats)) )
    #nu kleiner dataframe, dupliceer records in groep
    pstatsc = pstats[pstats ['KAfstV'] >0].merge(myxlatKAfstV,how='left',on='KAfstV').drop( columns='KAfstV')
    if debug:    
        print( ( "oorspr lengte groepen met duplicaten", len(pstatsc)) )
    pstatsc = pstatsc.groupby(pltgrps +[ 'KAfstCluCode', xvarPC]).sum().reset_index()
    if debug:
        print( ( "lengte clusters", len(pstatsc)) )
    dfgrps = pstatsc.groupby(pltgrps )[  [ 'KAfstCluCode']].count().reset_index().drop( columns='KAfstCluCode')
    if debug:
        print( ( "types opdeling", len(dfgrps)) )
    #let op: right mergen: ALLE postcodes meenemen, en niet waargenomen op lage waarde zetten
    vollandrecs =  dfgrps
#    print(vollandrecs)
    if debug:
        print( ( "alle land combinaties", len(vollandrecs) , len(mygeoschpc4)* len(dfgrps)) )
    pstatsc=pstatsc.rename(columns={xvarPC:'PC4'}).merge(vollandrecs,how='right')
    pstatsc['GeoInd'] = xvarPC
    pstatsc = addgrpexpl(pstatsc,myspecvals, pltgrps[0], pltgrps[1],grp1map)

    if debug:
        print( ( "return rdf", len(pstatsc)) )
    
    return(pstatsc)

#code werkt nog niet 

def mkdfverplxypc4 (df,myspecvals,pltgrps,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4):
    df['FactorVGen']  = np.where(df['HighPCfls'],0, df['FactorV'] )
    df['FactorVSpec'] = np.where(df['HighPCfls'], df['FactorV'],0 )
    amodes= [5,6]
    df['FactorVGenActive']= np.where(np.isin(df['KHvm'], amodes ),df['FactorVGen']  ,0)
    df['FactorVSpecActive']= np.where(np.isin(df['KHvm'], amodes ),df['FactorVSpec']  ,0)

    fr2= (mkdfverplxypc4d1 (df,myspecvals,grp,pltgrps,isnhexpl,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4)
            for grp in ['AankPC','VertPC'] )
    rv=pd.concat(fr2).reset_index()
#        rv1= mkdfverplxypc4d1 (df,myspecvals,'AankPC',pltgrps,isnhexpl,selstr,myKAfstV,myxlatKAfstV)
#        rv2= mkdfverplxypc4d1 (df,myspecvals,'VertPC',pltgrps,isnhexpl,selstr,myKAfstV,myxlatKAfstV)
#        rv= rv1.append(rv2) .reset_index(drop=True)   
    return rv

fitgrps=['MotiefV','isnaarhuis']
odinverplgr = mkdfverplxypc4 (allodinyr ,specvaltab,fitgrps,'Motief en isnaarhuis',
                                useKAfstV,xlatKAfstV,geoschpc4r2)
odinverplgr
# -
odinverplgr[['KAfstCluCode','GeoInd']].groupby('KAfstCluCode').agg('count')

allodinyr[['KAfstV','Verpl']].groupby('KAfstV').agg('count')

odinverplgr[FactorVincols].sum()

odinverplgr[odinverplgr['PC4']==9711].groupby('MotiefV')[FactorVincols].sum()


# +
#other file merge geo
#other file addparsrecs

#dan een dataframe dat
#2) per lengteschaal, 1 PC (van of naar en anderegroepen (maar bijv ook Motief ODin data verzamelt)

def mkdfverplklas1 (df,myspecvals,xvarPC,pltgrps,grp1map,myinfoflds,myKAfstV,
                    myxlatKAfstV,njaar,UseSelFactorV):
    debug=False
    dfvrecs = df [(df['Verpl']==1 ) & (df['AankPC'] > 500) & (df['VertPC'] > 500)  ].copy(deep=False)   
#    oprecs = df [df['OP']==1]
    gcols=pltgrps+ ['KAfstV',xvarPC]
    #niet normaliseren als opgedeeld in jaren
    #voor de rest maakt het niet zo veel uit, omdat er per groep weer genormaliseerd wordt
    if xvarPC == 'Jaar':
        njaar=1
    #sommeer per groep per oorspronkelijke KAfstV, maar niet voor internationaal
    dfvrecs['FactorV']= dfvrecs[UseSelFactorV] /njaar
    dfvrecs['FactorKm']= dfvrecs['FactorV'] * dfvrecs['AfstV'] *0.1
    pstats = dfvrecs[gcols+myinfoflds].groupby(gcols).sum().reset_index()
    if debug:
        print( ( "oorspr lengte groepen", len(pstats)) )
    #nu kleiner dataframe, dupliceer records in groep
    pstatsa= pstats.merge(useKAfstVall[['KAfstV','AvgAfst']],how='left')
    #pstatsa['FactorKm']= pstatsa['FactorV'] * pstatsa['AvgAfst']    
    pstatsc = pstatsa[pstatsa ['KAfstV'] >0].merge(myxlatKAfstV,how='left').drop( columns='KAfstV')
    if debug:    
        print( ( "oorspr lengte groepen met duplicaten", len(pstats)) )
    pstatsc = pstatsc.groupby(pltgrps +[ 'KAfstCluCode', xvarPC]).sum().reset_index()
    if debug:
        print( ( "lengte clusters", len(pstats)) )
    dfgrps = pstatsc.groupby(pltgrps )[  [ 'KAfstCluCode']].count().reset_index().drop( columns='KAfstCluCode')
    if debug:
        print( ( "types opdeling", len(dfgrps)) )
    #let op: right mergen: ALLE postcodes meenemen, en niet waargenomen op lage waarde zetten
    vollandrecs =  dfgrps
#    print(vollandrecs)
    if debug:
        print( ( "alle land combinaties", len(vollandrecs) , len(mygeoschpc4)* len(dfgrps)) )
    pstatsc=pstatsc.rename(columns={xvarPC:'GrpVal'}).merge(vollandrecs,how='right')
    pstatsc['GrpVal'] = pd.to_numeric(pstatsc['GrpVal'],errors='coerce')
    pstatsc['GrpTyp'] = xvarPC
    explhere1 = myspecvals [myspecvals['Variabele_naam'] == xvarPC].copy()
    explhere1['Code'] = pd.to_numeric(explhere1['Code'],errors='coerce')
    if len(explhere1)>3:
#        print((xvarPC,explhere1))
        pstatsc=pstatsc.merge(explhere1,left_on='GrpVal', right_on='Code', how='left').rename(
           columns={'Code_label':'GrpV_label'}) 
        pstatsc=pstatsc.drop(columns=['Variabele_naam','Code'])
    #    print(pstatsc)
    else:
#        print('def lab '+xvarPC)
        pstatsc['GrpV_label']='as number'

    pstatsc = addgrpexpl(pstatsc,myspecvals, pltgrps[0], pltgrps[1],grp1map)
    pstatsc['SelFactorV']= UseSelFactorV
    if debug:
        print( ( "return rdf", len(pstatsc)) )
#    print(pstatsc)
    return(pstatsc)

#code werk nog niet 

def mkdfverplklas (df,myspecvals,pltgrps,myinfogrps,lisnhexpl,myinfoflds,myKAfstV,
                   myxlatKAfstV,njaar,UseSelFactorV):
    fr2= (mkdfverplklas1(df,myspecvals,grp,pltgrps,lisnhexpl,myinfoflds,myKAfstV,myxlatKAfstV,
                         njaar,UseSelFactorV)
            for grp in myinfogrps)
    rv=pd.concat(fr2).reset_index()
    return rv

mySelFactorVs = ['FactorV', 'FactorVGen','FactorVSpec']
#mySelFactorVs = ['FactorV']
infogrps=['KHvm','AankUur','VertUur','Jaar','Verpl']
infoflds=['FactorV','FactorKm']
#odinverplklinfo_o = mkdfverplklas (allodinyr ,specvaltab,fitgrps,infogrps,isnhexpl,infoflds,
#                                useKAfstV,xlatKAfstV,aantaljaren,'FactorV')
odinverplklinfo_o = pd.concat ( [mkdfverplklas (allodinyr ,specvaltab,fitgrps,infogrps,isnhexpl,infoflds,
                                useKAfstV,xlatKAfstV,aantaljaren,x)  for x in mySelFactorVs ] )
fitgrpse=fitgrps+['GrpExpl']
kinfoflds=["GrpTyp", "GrpVal","GrpV_label"]
odinverplklinfo_o


# +
#other file merge geo
#other file addparsrecs

#dan een dataframe dat
#2) per lengteschaal, 1 PC (van of naar en anderegroepen (maar bijv ook Motief ODin data verzamelt)

def _addfields(pstats,useKAfstVall,njaar,UseSelFactorV):
    #onterechte manier om FactorKm te vullen
    #pstatsa= pstats.merge(useKAfstVall[['KAfstV','AvgAfst']],how='left')
    #pstatsa['FactorKm']= pstatsa['FactorV'] * pstatsa['AvgAfst']
    pstatsa=pstats.copy(deep=False)
    pstatsa['FactorV']= pstatsa[UseSelFactorV] /njaar
    pstatsa['FactorKm']= pstatsa['FactorV'] * pstatsa['AfstV'] *0.1
    pstatsa['FactorKmAuto']= np.where(pstatsa['KHvm'] ==1 ,pstatsa['FactorKm']  ,0)
    amodes= [5,6]
    pstatsa['FactorKmActive']= np.where(np.isin(pstatsa['KHvm'], amodes ),pstatsa['FactorKm']  ,0)
    pstatsa['FactorVActive']= np.where(np.isin(pstatsa['KHvm'], amodes ),pstatsa['FactorV']  ,0)
    return pstatsa

def mkdfverplklasflgs(df,myspecvals,pltgrps,grp1map,myinfoflds,myKAfstV,myxlatKAfstV,njaar,UseSelFactorV):
    debug=False
    dfvrecs0 = df [(df['Verpl']==1 ) & (df['AankPC'] > 500) & (df['VertPC'] > 500)  ].copy(deep=False)   
#    oprecs = df [df['OP']==1]
    gcols=pltgrps+ ['KAfstV']
    dfvrecs = _addfields(dfvrecs0,useKAfstVall,njaar,UseSelFactorV)
    #sommeer per groep per oorspronkelijke KAfstV, maar niet voor internationaal
    pstats = dfvrecs[gcols+myinfoflds].groupby(gcols).sum().reset_index()
    if debug:
        print( ( "oorspr lengte groepen", len(pstats)) )
    #nu kleiner dataframe, dupliceer records in groep
    pstatsc = pstats[pstats ['KAfstV'] >0].merge(myxlatKAfstV,how='left').drop( columns='KAfstV')
    if debug:    
        print( ( "oorspr lengte groepen met duplicaten", len(pstats)) )
    pstatsc = pstatsc.groupby(pltgrps +[ 'KAfstCluCode']).sum().reset_index()
    if debug:
        print( ( "lengte clusters", len(pstats)) )
    dfgrps = pstatsc.groupby(pltgrps )[  [ 'KAfstCluCode']].count().reset_index().drop( columns='KAfstCluCode')
    if debug:
        print( ( "types opdeling", len(dfgrps)) )
    if debug:
        print( ( "alle land combinaties", len(vollandrecs) , len(mygeoschpc4)* len(dfgrps)) )
    pstatc= dfgrps

    pstatsc = addgrpexpl(pstatsc,myspecvals, pltgrps[0], pltgrps[1],grp1map)
    pstatsc['SelFactorV']= UseSelFactorV
    if debug:
        print( ( "return rdf", len(pstatsc)) )
#    print(pstatsc)
    return(pstatsc)

kflgsflds=['FactorV',"FactorKm","FactorKmAuto","FactorKmActive","FactorVActive"]
#odinverplflgs_o = mkdfverplklasflgs (allodinyr ,specvaltab,fitgrps,isnhexpl,kflgsflds, 
#                                   useKAfstV,xlatKAfstV,aantaljaren,'FactorV')
odinverplflgs_o = pd.concat( [mkdfverplklasflgs (allodinyr ,specvaltab,fitgrps,isnhexpl,kflgsflds, 
                                   useKAfstV,xlatKAfstV,aantaljaren,x) for x in mySelFactorVs ] )

odinverplflgs_o
# -

odinflgtots=odinverplflgs_o[odinverplflgs_o['KAfstCluCode']==15].sum()
odinflgtots

useKAfstV.to_pickle("../intermediate/ODINcatVN01uKA.pkl")
xlatKAfstV.to_pickle("../intermediate/ODINcatVN01xKA.pkl")

#odinverplgr per PC aank of vertr
#odinverplklinfo: in diverse groepen (tijd, uur, jaar) mogelijk toe te voegen data
#odinverplflgs: vlaggen die gejoind kunnen worden aan oorspronkelijke data
odinverplgr.to_pickle("../intermediate/ODINcatVN01db.pkl")
odinverplklinfo_o.to_pickle("../intermediate/ODINcatVN02db.pkl")
odinverplflgs_o.to_pickle("../intermediate/ODINcatVN03db.pkl")

# analyses welke afwijkend zijn kunnen pas in gebruiksfase


# +
#ga eerst uit van input data
def mkpc4odinact(indf,pc4fld):
    usedf = indf.copy(deep=False)
    amodes= [5,6]
    #mind: FactorKm is in allodinyr always zero
    usedf['FactorVActive']= np.where(np.isin(usedf['KHvm'], amodes ),usedf['FactorV']  ,0)
    odf= usedf[['FactorV','FactorVActive']+[pc4fld]].groupby([pc4fld]).sum()
    odf['ActFractOri'] = odf['FactorVActive']/odf['FactorV'] 
    odf['GeoInd'] =pc4fld
    odf=odf.reset_index().rename ( columns={pc4fld:'PC4'}) 
    return odf

pc4ODINinAct =  pd.concat( [ mkpc4odinact (allodinyr,'VertPC'),mkpc4odinact (allodinyr,'AankPC') ])
#pc4ODINinAct =  pd.concat( [ mkpc4odinact (allodinyr,'AankPC') ])
print(pc4ODINinAct.sum())
pc4ODINinAct
# +
#active modes grafiek


# +
#maakt een database voor active mode vergelijking
#neemt een database db1, gesommeerd naar KAfstV en tabel myKAfstV
#voegt samen, en maakt velden voor berekende waarden en labels
def prepactsdb(db1,myKAfstV):
    KafstActiveVori = db1.merge(myKAfstV,how='left')
    KafstActiveVori['FactorVr'] = KafstActiveVori['FactorV'] / np.max(KafstActiveVori['FactorV'] )
    KafstActiveVori['FactorVCum'] = KafstActiveVori['FactorV'].cumsum()
    KafstActiveVori['FactorVCum'] = KafstActiveVori['FactorVCum']/ np.max( KafstActiveVori['FactorVCum'])
    KafstActiveVori['FactorPCum'] = KafstActiveVori['FactorVCum'].shift(1,fill_value=0)+1e-6
    KafstActiveVori['KAfstVFmt'] = np.where(KafstActiveVori['MaxAfst'] ==0,"verder",
                                            KafstActiveVori['MaxAfst'].map(lambda x:"%3g"%(x)) )
    return(KafstActiveVori)

KafstActiveVori = prepactsdb(mkpc4odinact (allodinyr,'KAfstV').rename (columns={'PC4':'KAfstV'}),useKAfstVall)

#KafstActiveVori.dtypes
#sns.relplot(data=KafstActiveVori, x='KAfstV',y='ActFractOri',kind='scatter')
#sns.relplot(data=KafstActiveVori, x='KAfstV',y='FactorVr',kind='line')
# -

KafstActiveVori


# +
#neemt een database  myActiveVori en maakt standaard plot
# gebruikt kolommen FactorVCum ,ActFractOri , FactorV en KAfstVFmt
def pltactsdb(myActiveVori,savtag,title):
    KafstActiveVorid= myActiveVori.copy(deep=True)
    KafstActiveVorid['FactorPCum']=KafstActiveVorid['FactorVCum']
    KafstActiveVorid = pd.concat([ myActiveVori,KafstActiveVorid] ) .sort_values(by='FactorPCum')                       
    KafstActiveVorid  

    chart= sns.relplot(data=KafstActiveVorid, x='FactorPCum',y='ActFractOri',kind='line')
    totavgact= sum(myActiveVori['ActFractOri'] * myActiveVori['FactorV'] ) / \
                      sum( myActiveVori['FactorV'] )
    #totavgact= sum( KafstActiveVori['FactorVr'] )
    chart.fig.suptitle(title)
    #chart.fig.suptitle('Totaal aandeel actieve mobiliteit %.3f'%(totavgact))            
    chart.set_xlabels('Aandeel van de afstandklasse')
    chart.set_ylabels('Fractie van reizen, per afstandsklasse' )
    labcolor="#3498db" # choose a color
    for x, y, name in zip(myActiveVori['FactorVCum'],myActiveVori['ActFractOri'],
                          myActiveVori['KAfstVFmt']):
        chart.ax.text(x+.02, y , name, color=labcolor)
    chart.ax.text(.2,.2 , 'aandeel actieve\nmodes %.3f'%(totavgact), color=labcolor) 
    chart.ax.text(.7,.7 , 'aandeel gemotoriseerde\nmodes %.3f'%(1-totavgact), color=labcolor) 
    chart.ax.set_xlim(0,1)
    chart.ax.set_ylim(0,1)
    figname = "../output/act_reg_"+savtag+"_"+'m1.svg';
    chart.fig.savefig(figname, bbox_inches="tight")

pltactsdb(KafstActiveVori,'ori','Originele ODIN data')    


# +
#sommeer dan doorgegeven data
def mkpc4odinact(indf,pc4fld):
    usedf = indf.copy(deep=False)
    amodes= [5,6]
    #mind: FactorKm is in allodinyr always zero
    usedf['FactorVActive']= np.where(np.isin(usedf['KHvm'], amodes ),usedf['FactorV']  ,0)
    odf= usedf[['FactorV','FactorVActive']+[pc4fld]].groupby([pc4fld]).sum()
    odf['ActFractOri'] = odf['FactorVActive']/odf['FactorV'] 
    odf['GeoInd'] =pc4fld
    odf=odf.reset_index().rename ( columns={pc4fld:'PC4'}) 
    return odf

pc4ODINinAct =  pd.concat( [ mkpc4odinact (allodinyr,'VertPC'),mkpc4odinact (allodinyr,'AankPC') ])
#pc4ODINinAct =  pd.concat( [ mkpc4odinact (allodinyr,'AankPC') ])
print(pc4ODINinAct.sum())
pc4ODINinAct
# -

print("Finished")


