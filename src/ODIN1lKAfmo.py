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

useKAfstV,xlatKAfstV  = pickAnrs (specvaltab,'KAfstV',[1,2,3,4,5,6,7,8,-1] )
#print(xlatKAfstV)   
print(useKAfstV)   
# -

usePC4MXI=True
fietswijk1pc4= ODiN2readpkl.fietswijk1pc4
if usePC4MXI:
    fietswijk1pc4['S_MXI22_NS'] = fietswijk1pc4['S_MXI22_BWN']  / (fietswijk1pc4['S_MXI22_BWN']  + fietswijk1pc4['S_MXI22_BAN'] )
    fietswijk1pc4['S_MXI22_BB'] = fietswijk1pc4['S_MXI22_NS']
fietswijk1pc4['S_MXI22_BG'] = fietswijk1pc4['S_MXI22_BBN'] / fietswijk1pc4['S_MXI22_BGN']     
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
    outdf['S_MXI22_BG'] = fietswijk1pc4['S_MXI22_BWN'] / pc4data['oppervlak']        
    outdf['S_MXI22_GB'] = pd.qcut(fietswijk1pc4['S_MXI22_BB'], 10)
    outdf['S_MXI22_GG'] = pd.qcut(fietswijk1pc4['S_MXI22_BG'], 10)
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
#todo set columns
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    rv= pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))
#    collst = ()
#    rv.columns= list(mygeoschpc4.columns) + list(dfgrps.columns)
    return rv



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
def mkgeoschparafr (pc4data,pc4grid,rudigrid,myKAfstV,p_LW,p_LO):
    debug=False
    pc4lst=pc4grid.read(1)
    outdf=pc4data[['postcode4','aantal_inwoners']].rename(columns={'postcode4':'PC4'} )
    outdf['KAfstCluCode'] = np.max(myKAfstV["KAfstCluCode"])
    outdf['MaxAfst'] = 0
    outdfst= outdf.copy()
    
    R=dict()
    R_LW= rudifungrid.read(3)
    R_LT= rudifungrid.read(5)
    R_LO =  R_LT- R_LW   
    R_LW = np.power(R_LW,p_LW)
    R_LO = np.where(R_LO <0,-np.power(-R_LO,p_LO), np.power(R_LO,p_LO) )
    R['LW']= R_LW
    R['LO'] =  R_LO
#    R['LM'] =  (R_LO* R_LW) / (R_LO + R_LW+1e-10)

    for lkey in R.keys():
        lvals = rasteruts1.sumpixarea(pc4lst,R[lkey])
        colnam="M_"+ lkey +"_AL"
        outdf[colnam] = lvals  
        outdfst[colnam] = lvals 
        
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
            for okey in ('OW','OO','OM'):
                lvals = rasteruts1.sumpixarea(pc4lst,np.multiply(R[lkey],F[okey]))
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outdfadd[colnam] = lvals            
        
        outdf=outdf.append(outdfadd)
    if debug:
        print(("blklen" ,len(outdfst), "outlen" ,len(outdf)) )
    return(outdf)

geoschpc4all=mkgeoschparafr(cbspc4data,pc4inwgrid,rudifungrid,useKAfstV,1.2,2.0)
geoschpc4 = geoschpc4all
# -

#kijk even naar sommen
geoschpc4.groupby(['KAfstCluCode','MaxAfst']).agg('sum')

geoschpc4

useKAfstVland = useKAfstV [useKAfstV['MaxAfst']==0]
geoschpc4land=mkgeoschparafr(cbspc4data,pc4inwgrid,rudifungrid,useKAfstVland,1.2,2.0)
geoschpc4land


def arrvannaarcol(adddf,verpldf,xvarPC):
    dfvrecs = verpldf [verpldf['Verpl']==1]
    gcols= [xvarPC]
    pstats = dfvrecs[gcols+['FactorV']].groupby(gcols).sum().reset_index()
    outvar='TotaalV'+xvarPC
    pstats=pstats.rename(columns={xvarPC:'PC4'})
#    print(pstats)
    addfj = adddf.merge(pstats,how='left')
    adddf[outvar] = addfj['FactorV']
    print(len(pstats))
arrvannaarcol(geoschpc4land,allodinyr,'AankPC')
arrvannaarcol(geoschpc4land,allodinyr,'VertPC')

# +
#geoschpc4 is een mooi dataframe met generieke woon en werk parameters
#Er is nog wel een mogelijk verzadigings effect daar waar de waarden voor
#grotere afstanden die van de landelijke waarden benaderen
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
geoschpc4r1=geoschpc4land[(  ~ np.isnan(geoschpc4land['M_LW_AL'])) & 
                     ( ~ np.isnan(geoschpc4land['aantal_inwoners'])) &
                     ( geoschpc4land['aantal_inwoners'] != ODINmissint )]

geoschpc4r2= cbspc4data[['postcode4','oppervlak'] ] .merge (geoschpc4r1 ,left_on=('postcode4'), right_on = ('PC4') )

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
allodinyr['isnaarhuis'] =  (allodinyr ['Doel'] ==1 )
addparscol(allodinyr ,"AankPC/rudifun/S_MXI22_BB").dtypes


def testmergelst(xvarPC,pltgrps):
    print ([ 'KAfstV',xvarPC,'FactorV']+pltgrps)
testmergelst  ('AankPC',['MotiefV','isnaarhuis'])  


# +
#dan een dataframe dat
#2) per lengteschaal, 1 PC (van of naar en anderegroepen (maar bijv ook Motief ODin data verzamelt)

def mkdfverplxypc4d1 (df,myspecvals,xvarPC,pltgrps,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4,ngrp):
    debug=False
    dfvrecs = df [(df['Verpl']==1 ) & (df[xvarPC] > 500)  ]   
    for pgrp in pltgrps:
        dfvrecs=addparscol(dfvrecs,pltgrps)
#    oprecs = df [df['OP']==1]
    gcols=pltgrps+ ['KAfstV',xvarPC]
    #sommeer per groep per oorspronkelijke KAfstV, maar niet voor internationaal
    pstats = dfvrecs[gcols+['FactorV']].groupby(gcols).sum().reset_index()
    if debug:
        print( ( "oorspr lengte groepen", len(pstats)) )
    #nu kleiner dataframe, dupliceer records in groep
    pstatsc = pstats[pstats ['KAfstV'] >0].merge(myxlatKAfstV,how='left').drop( columns='KAfstV')
    if debug:    
        print( ( "oorspr lengte groepen met duplicaten", len(pstats)) )
    pstatsc = pstatsc.groupby(pltgrps +[ 'KAfstCluCode', xvarPC]).sum().reset_index()
    if debug:
        print( ( "lengte clusters", len(pstats)) )
    dfgrps = pstatsc.groupby(pltgrps )[  [ 'KAfstCluCode']].count().reset_index().drop( columns='KAfstCluCode')
    if debug:
        print( ( "types opdeling", len(dfgrps)) )
    #let op: right mergen: ALLE postcodes meenemen, en niet waargenomen op lage waarde zetten
    vollandrecs = cartesian_product_multi (mygeoschpc4, dfgrps)
    vollandrecs.columns= list(mygeoschpc4.columns) + list(dfgrps.columns)
#    print(vollandrecs)
    if debug:
        print( ( "alle land combinaties", len(vollandrecs) , len(mygeoschpc4)* len(dfgrps)) )
    pstatsc=pstatsc.rename(columns={xvarPC:'PC4'}).merge(vollandrecs,how='right')
    pstatsc['GeoInd'] = xvarPC
    if 1==1:
        explhere = myspecvals [myspecvals['Variabele_naam'] == pltgrps[0]].copy()
        explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
#        print(explhere)
        pstatsc=pstatsc.merge(explhere,left_on=pltgrps[0], right_on='Code', how='left')    
        pstatsc['GrpExpl'] = pstatsc[pltgrps[0]].astype(str) + pstatsc[pltgrps[1]].astype(str) + " : " + pstatsc['Code_label']    
        pstatsc= pstatsc.drop(columns=['Code','Code_label'])
    else:
        pstatsc['GrpExpl']=''
    if debug:
        print( ( "return rdf", len(pstatsc)) )
    
    return(pstatsc)

#code werk nog niet 

def mkdfverplxypc4 (df,myspecvals,pltgrps,selstr,myKAfstV,myxlatKAfstV,geoschpc4in,ngrp,p_OA):
    mygeoschpc4 = geoschpc4in
    if 1==1:
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            okey='OA'
            colnam="M_"+ lkey +"_" + okey
            mygeoschpc4[colnam] = mygeoschpc4[colnamAL] * (np.power(mygeoschpc4['MaxAfst']*0.01,p_OA) )
#            okey='AO'
#            colnam="M_"+ lkey +"_" + okey
#            mygeoschpc4.drop(inplace=True,columns=colnam)
    rv1= mkdfverplxypc4d1 (df,myspecvals,'AankPC',pltgrps,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4,ngrp)
    rv2= mkdfverplxypc4d1 (df,myspecvals,'VertPC',pltgrps,selstr,myKAfstV,myxlatKAfstV,mygeoschpc4,ngrp)
    rv= rv1.append(rv2) .reset_index(drop=True)   
    return rv


indatverplgr = mkdfverplxypc4 (allodinyr ,specvaltab,['MotiefV','isnaarhuis'],'Motief en isnaarhuis',
                                useKAfstV,xlatKAfstV,geoschpc4,100,2.0)
indatverplgr
# +
def choose_cutoff(indat,hasfitted,prevrres):
    outframe=indat.copy()
    if hasfitted:
        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 2.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
    else:
        outframe['ALsafe'] =True
        outframe['osafe'] = True
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            for okey in ('OW','OO'):
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe['ALsafe'] = outframe['ALsafe'] & (outframe[colnam] > 0.2  * outframe[colnamAL])
                outframe['osafe']  = outframe['osafe']  & (outframe[colnam] < 0.01 * outframe[colnamAL])
        outframe['ALsafe'] = outframe['ALsafe'] | (outframe['MaxAfst']==0)
        outframe['osafe'] = outframe['osafe'] & (outframe['MaxAfst']!=0)
    if 1==1:
        outframe['FactorV'] = outframe['FactorV'] * ((outframe['ALsafe'] |outframe['osafe'] ).astype(int))
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            outframe[colnamAL] = outframe[colnamAL] * ((outframe['ALsafe']).astype(int))
            for okey in ('OW','OO','OM','OA'):
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe[colnam] = outframe[colnam] * ((outframe['osafe']).astype(int))
    outframe['ALmult'] = ( (outframe['ALsafe']==False).astype(int))
    return outframe

cut2=  choose_cutoff(indatverplgr,False,0)   
cut2
# -


cut2 [(cut2['MaxAfst']!=0 ) & (cut2['FactorV']!=0 )]

cut2 [(cut2['MaxAfst']!=0 ) & (cut2['M_LW_AL']!=0 )]


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


def dofitdatverplgr(indf,topreddf,pltgrp):
#    indf = indf[(indf['MaxAfst']!=95.0) & (indf[pltgrp]<3) ]
    debug=False
    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'M_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    Fitperscale=False
    if Fitperscale:
        fitgrp=pltgrp +['KAfstCluCode','GeoInd'  ]
    else:
        fitgrp=pltgrp + ['GeoInd' ]
    rf= indf.groupby(fitgrp ).apply(
            _regressgrp, 'FactorV', colvacols2, colpacols2).reset_index()
    outdf = topreddf.merge(rf,how='left')
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
                           np.where((s2al==0),s2,1/ (1/ s2+ 1/s2al )) )
    if (debug):
        print (s2ch)
    outdf['FactorEst'] = s2ch
    outdf['DiffEst'] =outdf['FactorV']-s2ch
    return(outdf)


fitdatverplgr = dofitdatverplgr(cut2,indatverplgr,['MotiefV','isnaarhuis'])
seaborn.scatterplot(data=fitdatverplgr,x="FactorEst",y="DiffEst",hue="GeoInd")
# -

#voor de time being, overschrijf de vorige selectie gegevens
for r in range(2):
    cut3=  choose_cutoff(indatverplgr,True,fitdatverplgr)  
    fitdatverplgr = dofitdatverplgr(cut3,indatverplgr,['MotiefV','isnaarhuis'])
seaborn.scatterplot(data=fitdatverplgr,x="FactorEst",y="DiffEst",hue="GeoInd")

fitdatverplgr["M_LM_AL"] = fitdatverplgr["M_LW_AL"] * fitdatverplgr["M_LO_AL"]
seaborn.scatterplot(data=fitdatverplgr,x="M_LM_AL",y="DiffEst",hue="GeoInd")

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
        limcat=4e9
    else:
        limcat=1e9
    bigmot= rv2[(rv2['MaxAfst']==0) & (rv2['FactorV']>limcat)  ].groupby('MotiefV').agg(['count']).reset_index()
    print(bigmot)
    rv2['MaxAfst']=np.where(rv2['MaxAfst']==0 ,100,rv2['MaxAfst'])
    rv2['MaxAfst']=rv2['MaxAfst'] * np.where(rv2['GeoInd']=='AankPC',1,1.02)
    rv2['Qafst']=1/(1/(rv2['MaxAfst']  *0+1e10) +1/ (np.power(rv2['MaxAfst'] ,1.8) *2e8 ))
    rv2['linpmax'] = rv2['FactorEstNAL']/ rv2['FactorEstAL']
    rv2['linpch']= rv2['FactorEst']/ rv2['FactorEstAL']
    rv2['drat']= rv2['FactorV']/ rv2['FactorEstAL']

    rvs = rv2[np.isin(rv2['MotiefV'],bigmot['MotiefV'])]
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
def trypowerland (pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV,p_LW,p_LO,p_OA):
    print((p_LW,p_LO,p_OA))
    mygeoschpc4= mkgeoschparafr(pc4data,pc4grid,rudigrid,myKAfstV,p_LW,p_LO)
    myxlatKAfstV=myKAfstV[['KAfstCluCode']].merge(inxlatKAfstV,how='left')
#    print (myxlatKAfstV)
    mydatverplgr = mkdfverplxypc4 (allodinyr ,specvaltab ,['MotiefV','isnaarhuis'],'Motief en isnaarhuis',
                                myKAfstV,xlatKAfstV,mygeoschpc4,100,p_OA)
    cut2i=  choose_cutoff(mydatverplgr,False,0)  
    myfitverplgr = dofitdatverplgr(cut2i,mydatverplgr,['MotiefV','isnaarhuis'])
    for r in range(2):
        cut3i=  choose_cutoff(mydatverplgr,True,myfitverplgr) 
        myfitverplgr = dofitdatverplgr(cut3i,mydatverplgr,['MotiefV','isnaarhuis'])
    rdf=calcchidgrp(myfitverplgr)
    return(np.sum(rdf['chisq'].reset_index().iloc[:,1]))
    
rv=trypowerland(cbspc4data,pc4inwgrid,rudifungrid,useKAfstVland,xlatKAfstV,1.3,1.0,2.0)
rv
# -



# +
def chisqsampler (pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV):
    lw = np.linspace(1.1,1.3,3)
    oa = np.linspace(1.6,2.0,3)
    lo = np.linspace(1.7,2.0,3)
    p_LW,p_LO = np.meshgrid(lw, lo)
    l_OA=2.0
#    l_LW=1.2
    print( (p_LW,p_LO))
    myfunc =lambda  l_LW,l_LO  :trypowerland (pc4data,pc4grid,rudigrid,myKAfstV,inxlatKAfstV,l_LW,l_LO,l_OA)
    vfunc = np.vectorize(myfunc)
    z= ( vfunc(p_LW,p_LO) )
    z=np.array(z)
    return z
chitries= chisqsampler (cbspc4data,pc4inwgrid,rudifungrid,useKAfstVland,xlatKAfstV)    
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

pllanddiffam= cbspc4data[['postcode4']].merge(fitdatverplgr[(fitdatverplgr['MaxAfst']==0) ] ,
                                                             how='left',left_on=('postcode4'), right_on = ('PC4'))
pllanddiffam[abs(pllanddiffam['DiffEst']>1.5e7)].groupby(['postcode4','GrpExpl'])[['isnaarhuis']].agg('count')

# +
#inspectie
#grote onderschatters landelijK; winkelcentra, schiphol, en mindere mate universiteit

#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(3511,3512,3584 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(6511,6525 )))]
chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(5611,5612 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(1012,1017,1043,1101,1118 )))]
#chkpckrt = cbspc4data[(np.isin (cbspc4data['postcode4'],(2262,2333,2511,2595 )))]
pchkpckrt = chkpckrt.to_crs(epsg=plot_crs).plot()
cx.add_basemap(pchkpckrt, source= prov0)
# -

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


