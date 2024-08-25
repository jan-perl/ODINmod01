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

import re

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"

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

pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid= rasterio.open(pc4tifname)

#rudifunset
Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
Rf_net_buurt.reset_index(inplace=True,drop=True)
rudifuntifname=calcgdir+'/oriTN2-NL.tif'
rudifungrid= rasterio.open(rudifuntifname)

#nu nog MXI overzetten naar PC4 ter referentie




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

useKAfstV,xlatKAfstV  = pickAnrs (specvaltab,'KAfstV',[5,8,-1] )
print(xlatKAfstV)   
print(useKAfstV)   
# -

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


def mkfietswijk3pc4(pc4data,pc4grid,rudigrid):
    pc4lst=pc4grid.read(1)
    outdf=pc4data[['postcode4','aantal_inwoners']].rename(columns={'postcode4':'PC4'} )
    outdf['aantal_inwoners_gr2'] = rasteruts1.sumpixarea(pc4lst,pc4grid.read(3) )
    outdf['S_MXI22_BWN'] = rasteruts1.sumpixarea(pc4lst,rudifungrid.read(3) )
    outdf['S_MXI22_BAT'] = rasteruts1.sumpixarea(pc4lst,rudifungrid.read(5) )
    outdf['S_MXI22_BAN'] = outdf['S_MXI22_BWN'] - outdf['S_MXI22_BAT'] 
    if usePC4MXI:
        outdf['S_MXI22_NS'] = outdf['S_MXI22_BWN']  / (outdf['S_MXI22_BWN']  + outdf['S_MXI22_BAN'] )
        outdf['S_MXI22_BB'] = outdf['S_MXI22_NS']
    
    outdf['aantal_inwoners_d2'] = outdf['aantal_inwoners_gr2'] -outdf['aantal_inwoners']
    return outdf
fietswijk3pc4=mkfietswijk3pc4(cbspc4data,pc4inwgrid,rudifungrid)
bd=fietswijk3pc4 [abs(fietswijk3pc4['aantal_inwoners_d2'] ) > 1 ]


# +
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)  

#from https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
def cartesian_product_multi(*dfs):
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))



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
def mkgeoschparafr (pc4data,pc4grid,rudigrid,myKAfstV):
    pc4lst=pc4grid.read(1)
    outdf=pc4data[['postcode4','aantal_inwoners']].rename(columns={'postcode4':'PC4'} )
    outdf['KAfstCluCode'] = np.max(myKAfstV["KAfstCluCode"])
    outdf['MaxAfst'] = 0
    outdfst= outdf.copy()
    
    R=dict()
    R_LW= rudifungrid.read(3)
    R['LW']= R_LW
    R_LT= rudifungrid.read(5)
    R_LO =  R_LT- R_LW        
    R['LO'] =  R_LO
#    R['LM'] =  (R_LO* R_LW) / (R_LO + R_LW+1e-10)

    for lkey in R.keys():
        lvals = rasteruts1.sumpixarea(pc4lst,R[lkey])
        for okey in ('OW','OO','OM'):
            colnam="M_"+ lkey +"_" + okey
#            print(colnam)
            outdf[colnam] = lvals            
    R_LW_land= np.sum(R_LW)
    R_LO_land= np.sum(R_LO)
    
    for index, row in myKAfstV[myKAfstV['MaxAfst']!=0].iterrows():        
        outdfadd=outdfst.copy()
        outdfadd['KAfstCluCode']= row["KAfstCluCode"]
        outdfadd['MaxAfst'] = row["MaxAfst"]
        print(row["KAfstCluCode"], row["MaxAfst"])
        filt=rasteruts1.roundfilt(100,row["MaxAfst"])

        F=dict()
        F_OW = rasteruts1.convfiets2d(R_LW, filt ) /R_LW_land
        F['OW'] =  F_OW  
        F_OT = rasteruts1.convfiets2d(R_LT, filt ) /R_LO_land
        F_OO =  F_OT- F_OW        
        F['OO'] =  F_OO 
        F['OM'] =  (F_OO* F_OW) / (F_OO + F_OW+1e-10)

        for lkey in R.keys():
            for okey in ('OW','OO','OM'):
                lvals = rasteruts1.sumpixarea(pc4lst,R[lkey]*F[okey])
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outdfadd[colnam] = lvals            
        
        outdf=outdf.append(outdfadd)
    print(len(outdfst))
    return(outdf)

geoschpc4=mkgeoschparafr(cbspc4data,pc4inwgrid,rudifungrid,useKAfstV)
geoschpc4


# +
#geoschpc4 is een mooi dataframe met generieke woon en werk parameters
#Er is nog wel een mogelijk verzadigings effect daar waar de waarden voor
#grotere afstanden die van de landelijke waarden benaderen

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
            elif 0 & ~ (srcarr[2] in list(fietswijk3pc4.columns)):
                print ("Error in addparscol: col ",srcarr[2]," not usable for ", coltoadd)
            else:
                fwin4 = fietswijk3pc4[['PC4',srcarr[2]]]
                df=df.merge(fwin4,left_on=srcarr[0], right_on='PC4',how='left')
                df=df.rename(columns={srcarr[2]:coltoadd})
        else:
            print ("Error in addparscol: source ",srcarr[1]," not found for", coltoadd)
#    print ("addparscol: nrecords: ",len(df.index))
    return (df)
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
addparscol(naarhuis,"AankPC/rudifun/S_MXI22_BB").dtypes


# +
#dan een dataframe dat
#2) per lengteschaal, 1 PC (van of naar en anderegroepen (maar bijv ook Motief ODin data verzamelt)

def mkdfverplxypc4 (df,myspecvals,xvar,pltgrp,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4,ngrp):
    xsrcarr=  xvar.split ('/')
    xvarPC = xsrcarr[0]
    dfvrecs = df [(df['Verpl']==1 ) & (df[xvarPC] > 500)  ]   
    dfvrecs=addparscol(dfvrecs,pltgrp)
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[pltgrp, 'KAfstV',xvarPC,'FactorV']].groupby([pltgrp, 'KAfstV', xvarPC]).sum().reset_index()
    print(len(pstats))
    pstatsc = pstats[pstats ['KAfstV'] >0].merge(myxlatKAfstV,how='left').drop( columns='KAfstV')
    print(len(pstatsc))
    pstatsc = pstatsc.groupby([pltgrp, 'KAfstCluCode', xvarPC]).sum().reset_index()
    print(len(pstatsc))
    pstatsc=pstatsc.rename(columns={xvarPC:'PC4'}).merge(mygeoschpc4,how='left')
    return(pstatsc)

#code werk nog niet 


naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]    
indatverplgr = mkdfverplxypc4 (naarhuis,specvaltab,
                                'AankPC/rudifun/S_MXI22_BWN','MotiefV','Naar huis',
                                useKAfstV,xlatKAfstV,geoschpc4,100)
indatverplgr
# -

from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import seaborn


# +
def _regressgrp(indf, yvar, xvars,pcols):  
#        reg_nnls = LinearRegression(fit_intercept=False )
        y_train=indf[yvar]
        X_train=indf[xvars]
#        print (X_train)
        fit1 = nnls(X_train, y_train)    
        rv=pd.DataFrame(fit1[0],index=pcols).T
        return(rv)


def dofitdatverplgr(indf,pltgrp):
#    indf = indf[(indf['MaxAfst']!=95.0) & (indf[pltgrp]<3) ]
    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'M_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    rf= indf.groupby([pltgrp ,'KAfstCluCode' ]).apply(
            _regressgrp, 'FactorV', colvacols2, colpacols2).reset_index()
    indf = indf.merge(rf,how='left')
    blk1=indf[colvacols2 ]
    blk2=indf[colpacols2 ]
#    print(blk1)
    s2= np.sum(np.array(blk1)*np.array(blk2),axis=1)
    indf['FactorEst'] =s2
    indf['DiffEst'] =indf['FactorV']-s2
    return(indf)


fitdatverplgr = dofitdatverplgr(indatverplgr,'MotiefV')
seaborn.scatterplot(data=fitdatverplgr,x="FactorEst",y="DiffEst")
# -

seaborn.scatterplot(data=fitdatverplgr[fitdatverplgr['MaxAfst']==0],x="FactorEst",y="DiffEst")

pllanddiff= cbspc4data.merge(fitdatverplgr[(fitdatverplgr['MaxAfst']==0) 
                    &  (fitdatverplgr ['MotiefV'] ==4 )][['PC4','DiffEst','FactorEst','FactorV']],
            how='left',left_on=('postcode4'), right_on = ('PC4'))
print ( (len(cbspc4data), len(pllanddiff) ) )
if 1==1:
    pland= pllanddiff.plot(column= 'DiffEst',
                                cmap='jet',legend=True,alpha=.6)
else:    
    pland= pllanddiff.to_crs(epsg=plot_crs).plot(column= 'DiffEst',
                                cmap='jet',legend=True,alpha=.6)
    cx.add_basemap(pland, source= prov0)
#plland.plot( column= 'DiffEst', Legend=True)

pllanddiff.index


# +
def quicklandpc4plot(pc4df,pc4grid,pc4dfcol):
    idximg=pc4grid.read(1)
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
    
quicklandpc4plot(pllanddiff,pc4inwgrid,'DiffEst')


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


