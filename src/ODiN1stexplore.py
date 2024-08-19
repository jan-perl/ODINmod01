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
#een set doorsnedes van ODiN gegevens
# -

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

if 1==0:
    cbspc4data = geopandas.read_file("../data/CBS/PC4STATS/cbs_pc4_2022_v1.gpkg")
    cbspc4data['postcode4'] = pd.to_numeric(cbspc4data['postcode4'])
    stryear='2022'
    cbspc4data.to_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl") 

stryear='2022'
cbspc4data =pd.read_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl")

cbspc4data['oppervlak'] = cbspc4data.area

cbspc4data.dtypes

#providers = cx.providers.flatten()
#providers
prov0=cx.providers.nlmaps.grijs.copy()
print( cbspc4data.crs)
print (prov0)
#plot_crs=3857
plot_crs=28992
if 1==1:
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
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

if 0==1:
    cbspc6data = geopandas.read_file("../data/CBS/PC6STATS/cbs_pc6_2022_v2.gpkg")
    #cbspc6data['postcode6'] = pd.to_numeric(cbspc4data['postcode6'])
    stryear='2022'
    cbspc6data.to_pickle("../intermediate/CBS/pc6data_"+stryear+".pkl") 

cbspc6data =pd.read_pickle("../intermediate/CBS/pc6data_"+stryear+".pkl")

cbspc6data['oppervlak'] = cbspc6data.area

cbspc6data.dtypes

cbspc6data.plot()

cbspc6data.plot(column="aantal_woningen",legend=True, cmap='OrRd')

cbspc6data.columns

cbspc6data.columns.tolist()

cbspc6data['PC4']= cbspc6data['postcode6'].str[0:4].astype('int64')
htnpc6data=cbspc6data[(cbspc6data['PC4'] >3990) & (cbspc6data['PC4'] <3999) ].copy()

htnpc6data["dichtstbijzijnde_treinstation_afstand_in_km"][htnpc6data["dichtstbijzijnde_treinstation_afstand_in_km"] >=-90000].max()

htnpc6inw= htnpc6data[htnpc6data["aantal_inwoners"] >=-90000]
phtn=htnpc6inw.to_crs(epsg=3857). \
          plot(column="aantal_inwoners",legend=True, cmap='OrRd',alpha=.7)
cx.add_basemap(phtn, source= prov0)

# +
#now circles instead of areas
htnpc6inwr2= htnpc6data[htnpc6data["aantal_inwoners"] >=-90000].copy()
htnpc6inwr2[ "center"]= htnpc6inwr2.representative_point()
htnpc6inwr3= htnpc6inwr2.copy()
#htnpc6inwr3.set_geometry ( "center", inplace = True)
htnpc6inwr3.set_geometry (  htnpc6inwr3.representative_point(), inplace = True)
#htnpc6inwr3= htnpc6inwr3.set_geometry ( "center", inplace = False)

phtn3=htnpc6inwr3.to_crs(epsg=3857). \
          plot(column="aantal_inwoners",legend=True, cmap='OrRd',alpha=.7)
cx.add_basemap(phtn3, source= prov0)
# -

htnpc6inwr3['geometry']

htnpc6data[htnpc6data["aantal_inwoners"] >200]

htnpc6data['PC5']= cbspc6data['postcode6'].str[0:5]
hvnpc6data=htnpc6data[htnpc6data['PC5']== '3992P']
hvnpc6data[hvnpc6data["aantal_inwoners"] >=-90000].plot(column="aantal_inwoners",legend=True, cmap='OrRd')

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

allodinyr=ODiN2readpkl.allodinyr
len(allodinyr.index)

usePC4MXI=True
fietswijk1pc4= ODiN2readpkl.fietswijk1pc4
if usePC4MXI:
    fietswijk1pc4['S_MXI22_NS'] = fietswijk1pc4['S_MXI22_BWN']  / (fietswijk1pc4['S_MXI22_BWN']  + fietswijk1pc4['S_MXI22_BAN'] )
    fietswijk1pc4['S_MXI22_BB'] = fietswijk1pc4['S_MXI22_NS']
fietswijk1pc4['S_MXI22_GB'] = pd.qcut(fietswijk1pc4['S_MXI22_BB'], 10)
fietswijk1pc4['S_MXI22_GG'] = pd.qcut(fietswijk1pc4['S_MXI22_BG'], 10)
fietswijk1pc4['PC4'] =  pd.to_numeric(fietswijk1pc4['PC4'] ,errors='coerce')
print(fietswijk1pc4.dtypes)
if 0==1:
    fwin4 = fietswijk1pc4 [['PC4','S_MXI22_GB','S_MXI22_BB','S_MXI22_GG','S_MXI22_BG']]
    allodinyr2 = allodinyr.merge(fwin4,left_on='AankPC', right_on='PC4',how='left')
    allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GB': 'Aank_MXI22_GW', 'S_MXI22_BB': 'Aank_MXI22_BW' } )
    allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GG': 'Aank_MXI22_GG', 'S_MXI22_BG': 'Aank_MXI22_BG' } )
    allodinyr2 = allodinyr2.merge(fwin4,left_on='VertPC', right_on='PC4',how='left')
    allodinyr2 = allodinyr2.rename (columns= {'S_MXI22_GB': 'Vert_MXI22_GW', 'S_MXI22_BB': 'Vert_MXI22_BW' } )
    allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GG': 'Vert_MXI22_GG', 'S_MXI22_BG': 'Vert_MXI22_BG' } )
    print(allodinyr2.dtypes)
    len(allodinyr2.index)


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


# -

#todo: check waarom ifs niet werken
#print(fietswijk1pc4.columns)
def addparscol (df,coltoadd):
    if "/" in coltoadd:
        srcarr=  coltoadd.split ('/')
#        print(srcarr)
        if srcarr[1] == "rudifun":
            PCcols=["AankPC","VertPC"]
            if 0 &  ~ (srcarr[0] in PCcols):
                print ("Error in addparscol: col ",srcarr[0]," not usable for ", coltoadd)
            elif 0 & ~ (srcarr[2] in list(fietswijk1pc4.columns)):
                print ("Error in addparscol: col ",srcarr[2]," not usable for ", coltoadd)
            else:
                fwin4 = fietswijk1pc4[['PC4',srcarr[2]]]
                df=df.merge(fwin4,left_on=srcarr[0], right_on='PC4',how='left')
                df=df.rename(columns={srcarr[2]:coltoadd})
        else:
            print ("Error in addparscol: source ",srcarr[1]," not found for", coltoadd)
#    print ("addparscol: nrecords: ",len(df.index))
    return (df)
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
addparscol(naarhuis,"AankPC/rudifun/S_MXI22_BB").dtypes


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
    dfvrecs = df [(df['Verpl']==1 ) & (df[xvarPC] > 500)  ]   
    dfvrecs=addparscol(dfvrecs,pltgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[pltgrp, xvarPC,'FactorV']].groupby([pltgrp, xvarPC]).sum().reset_index()
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
    
    vardescr = dbk_2022_cols [dbk_2022_cols['Variabele_naam_ODiN_2022'] == xvar] ['Variabele_label_ODiN_2022']
#    print(vardescr)
    if len(vardescr) ==0:
        vardescr = ""        
        heeftlrv = True
    else:
        vardescr = vardescr.item()
        heeftlrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==xvar) ] ) !=0
#    print(vardescr,heeftlrv)

    grplrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==pltgrp) ] ) !=0
    if ~grplrv:
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
   
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]    
datpltverplp = mkpltverplxypc4 (naarhuis,specvaltab,
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
    dfvrecs = df [(df['Verpl']==1 ) & (df[xvarPC] > 500)  ]   
    dfvrecs=addparscol(dfvrecs,pltgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[pltgrp, xvarPC,'FactorV']].groupby([pltgrp, xvarPC]).sum().reset_index()
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
    
    vardescr = dbk_2022_cols [dbk_2022_cols['Variabele_naam_ODiN_2022'] == xvar] ['Variabele_label_ODiN_2022']
#    print(vardescr)
    if len(vardescr) ==0:
        vardescr = ""        
        heeftlrv = True
    else:
        vardescr = vardescr.item()
        heeftlrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==xvar) ] ) !=0
#    print(vardescr,heeftlrv)

    grplrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==pltgrp) ] ) !=0
    if ~grplrv:
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
   
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]    
datpltverplp = mkfitverplxypc4 (naarhuis,specvaltab,
                                'AankPC/rudifun/S_MXI22_BWN','MotiefV','Naar huis',100)
# -

datpltverplp = mkpltverplxypc4 (naarhuis,specvaltab,'VertPC/rudifun/S_MXI22_BWN',
                                'Jaar','Naar huis',100)

haarhuisapart = addparscol(naarhuis,'VertPC/rudifun/S_MXI22_BWN')
haarhuisapart = haarhuisapart[ (haarhuisapart['VertPC/rudifun/S_MXI22_BWN'] <1e3 ) | 
                               (haarhuisapart['VertPC/rudifun/S_MXI22_BWN'] >2e6 ) ]

haarhuisapart.groupby(['VertPC']).agg({'VertPC':'count','VertPC/rudifun/S_MXI22_BWN':'mean'} )

cbspc4data.merge(haarhuisapart, left_on='postcode4', right_on='VertPC', how='right').plot()


# +
#kijk eens naar verdelingen totaal aantal verplaatsingen per persoon
#maak zonodig extra kolommen aan in database, waar meerdere PC4 databases mogelijk zijn
# format [AankPC][VertPC]/[dbpc4]/[dbpc4 field] 
# maak groepen op aantallen postcodes (of oppervlakken ?)

def mkpltverplp (df,myspecvals,collvar,normgrp,selstr):
    dfvrecs = df [df['Verpl']==1]
    dfvrecs=addparscol(dfvrecs,collvar)
    dfvrecs=addparscol(dfvrecs,normgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[normgrp, collvar,'FactorV']].groupby([normgrp, collvar]).sum().reset_index()
    #print(pstats)
    denoms= pstats [[normgrp, 'FactorV']].groupby([normgrp]).sum().reset_index().rename(columns={'FactorV':'Denom'} )
    #print(denoms)
    pstatsn = pstats.merge(denoms,how='left')
    pstatsn['FractV'] = pstatsn['FactorV'] *100.0/ pstatsn['Denom']
    vardescr = dbk_2022_cols [dbk_2022_cols['Variabele_naam_ODiN_2022'] == collvar] ['Variabele_label_ODiN_2022']
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
   
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
#datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','Jaar','Alle ritten')
#datpltverplp = mkpltverplp (allodinyr,specvaltab,'VertUur','Jaar','Alle ritten')
datpltverplp = mkpltverplp (naarhuis,specvaltab,'AankPC/rudifun/S_MXI22_GB','Jaar','Alle ritten')
# -

datpltverplp = mkpltverplp (allodinyr,specvaltab,'MotiefV','Jaar','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'KHvm','Jaar','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'KAfstV','Jaar','Alle ritten')

#maak andere grafiek met zo veel categorieen
datpltverplp = mkpltverplp (allodinyr,specvaltab,'VertUur','Jaar','Alle ritten')

#neem alleen ritten naar huis, Aankomst is dan wonen: 
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
datpltverplp = mkpltverplp (naarhuis,specvaltab,'AankPC/rudifun/S_MXI22_GB','Jaar','Naar huis')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'VertPC/rudifun/S_MXI22_GB','AankPC/rudifun/S_MXI22_GB','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','AankPC/rudifun/S_MXI22_GB','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','VertPC/rudifun/S_MXI22_GB','Alle ritten')

#naar huis: aankomst is wonen: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KHvm','AankPC/rudifun/S_MXI22_GB','Naar huis')

#naar huis: aankomst is wonen: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KAfstV','AankPC/rudifun/S_MXI22_GB','Naar huis')

#naar huis: aankomst is wonen, als functie van oppervlakte dichtheid: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KAfstV','AankPC/rudifun/S_MXI22_GG','Naar huis')

#naar huis: vertrel is utiliteit: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KHvm','VertPC/rudifun/S_MXI22_GB','Naar huis')

#neem alleen ritten naar werk, Aankomst is dan werken: 
naarwerk= allodinyr [allodinyr ['Doel'] ==2 ]
datpltverplp = mkpltverplp (naarwerk,specvaltab,'AankPC/rudifun/S_MXI22_GB','Jaar','Naar werk')

#neem alleen ritten naar werk, Aankomst is dan wonen
datpltverplp = mkpltverplp (naarwerk,specvaltab,'VertPC/rudifun/S_MXI22_GB','Jaar','Naar werk')

#neem alleen ritten naar werk, Aankomst is dan werken: 
datpltverplp = mkpltverplp (naarwerk,specvaltab,'KHvm','AankPC/rudifun/S_MXI22_GB','Naar werk')

#neem alleen ritten naar werk, Aankomst is dan werken: 
datpltverplp = mkpltverplp (naarwerk,specvaltab,'KAfstV','AankPC/rudifun/S_MXI22_GB','Naar werk')


