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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import geopandas
import contextily as cx
import xyzservices.providers as xyz
from scipy.optimize import nnls

#import ODiN2pd
import ODiN2readpkl

ODiN2readpkl.allodinyr.dtypes

dbk_2022 = ODiN2readpkl.dbk_allyr
dbk_2022_cols = dbk_2022 [~ dbk_2022.Variabele_naam_ODiN_2022.isna()]
dbk_2022_cols [ dbk_2022_cols.Niveau.isna()]

#valideeer dat alleen ['P', 'V', 'R', 'W'] voor komen in bestand
dbk_2022_cols.Niveau.unique()


def gtyrdat(yr):
    return (ODiN2readpkl.allodinyr[ODiN2readpkl.allodinyr['Jaar']==yr])


# +
excols = ODiN2readpkl.excols
def std_zero(x): return np.std(x, ddof=0)
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

chklevstat(gtyrdat(2022),"RitID",dbk_2022_cols,'Variabele_naam_ODiN_2022','R')
# -

chklevstat(gtyrdat(2022),"VerplID",dbk_2022_cols,'Variabele_naam_ODiN_2022','V')

chklevstat(gtyrdat(2022),"OPID",dbk_2022_cols,'Variabele_naam_ODiN_2022','P')

# +
largranval =ODiN2readpkl.largranval 

specvaltab = ODiN2readpkl.mkspecvaltab(dbk_2022)
specvaltab


# +
def chkexplstat(df,dbk_cols,vnamcol,specvaltab):
    chkcols = dbk_cols [  ~ ( dbk_cols[vnamcol].isin( excols) )]
    #chkcols = chkcols.iloc [range(5)]
    c0=[]
    c1=[]
    c2=[]
    for chkcol in chkcols[vnamcol]:
        nonadf= df[~ ( df[chkcol].isna() ) ]
        nonadf[chkcol] = pd.to_numeric(nonadf[chkcol],errors='coerce')
        explhere = specvaltab [specvaltab ['Variabele_naam'] == chkcol]
        explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
        print(explhere)
        mspec=nonadf.merge(explhere,left_on=chkcol, right_on='Code', how='left')
        ltot=len(mspec)
        lna = len(mspec [mspec ['Code_label'].isna()] )
        c0.append(chkcol)
        c1.append(ltot)
        c2.append(lna)
    outcol_names =  ['Variabele_naam', 'ltot', 'lna'] 
    outdf=pd.DataFrame(list(zip(c0,c1,c2)),columns=outcol_names)
    return(outdf)

        
chklabs= chkexplstat(gtyrdat(2022),dbk_2022_cols,'Variabele_naam_ODiN_2022',specvaltab)
chklabs
# -

#een paar kolommen niet gelabeld, en Factoren niet
chklabs[chklabs['lna'] == 200054]

#hier verwacht men meldingen
leeftlr = specvaltab [ specvaltab  ['Code'] ==largranval ]
leeftlr

#mooi: het enige dat over is zijn zjin VeplID en RitID
#
chklabs[~ (chklabs['lna'].isin([0,200054]))] .merge(leeftlr,how='left')


# +
#en vergelijkbare code als hierboven kan dus gebruikt worden om te taggen
#geef waarschuwing als largranval ook is gezet voor kolommen -> dan werkt taggen niet 
#en is numerieke code beter, tenzij Code_label gezet is

# +
#check AantVpl met aantal verplaatingen per persoon, check ook 0 waarden
# let op: bij Rit is waarde 3 een buitenlandse rit, die tellen wel mee binnen verplaatsing
def chkaantal(df,dbk_cols,idsumm,refwrd,nwcol1,nwcol2):
    nAantCnb = df [ df [nwcol1]==1][[idsumm,refwrd]]
    nAantCnt = df [ df [nwcol2].isin([1,3])][[idsumm,nwcol2]].groupby(idsumm).count()[[nwcol2]].reset_index()
    cvmb = nAantCnb.merge(nAantCnt ,on=idsumm,how='outer')
    outdf = cvmb.groupby([refwrd,nwcol2]).count()[[idsumm]].reset_index()
    outdf['Same']= outdf[refwrd] == outdf[nwcol2]
    return(outdf)

        
chkAantVpl= chkaantal(ODiN2readpkl.allodinyr,dbk_2022_cols,'OPID','AantVpl','OP','Verpl')
chkAantVpl
# -

#check AantRit met aantal riitn per verplaating, check ook of 0 voor komt
chkAantRit= chkaantal(ODiN2readpkl.allodinyr,dbk_2022_cols,'VerplID','AantRit','Verpl','Rit')
chkAantRit


#from https://stackoverflow.com/questions/14507794/how-to-flatten-a-hierarchical-index-in-columns
def flatten_columns(self):
    """Monkey patchable function onto pandas dataframes to flatten multiindex column names from tuples. Especially useful
    with plotly.

    pd.DataFrame.flatten_columns = flatten_columns

    """
    df = self.copy()
    df.columns = [
        '_'.join([str(x)
                  for x in [y for y in item
                            if y]]) if not isinstance(item, str) else item
        for item in df.columns
    ]
    return df
pd.DataFrame.flatten_columns = flatten_columns


# +
#TODO check sum FactorV versus FactorP
# -

#TODO check AfstV versus KAfstV
def chkKafstVvsAfstV(dfin):
    df= dfin [dfin ['Rit'] ==1] 
    grp = df.groupby(['KAfstV'])[['AfstV']].agg(['min','mean','max','count']).reset_index().flatten_columns()
    grp['AfstV_mid'] = 0.5*(grp['AfstV_min'] + grp['AfstV_max'])
    return grp
chkKafstVvsAfstV(ODiN2readpkl.allodinyr)


#check AfstR versus KAfstR
#deze kloppen gewoon, ritten worden overigens niet gebruikt
def chkKafstRvsAfstR(df):
    grp = df.groupby(['KAfstR'])[['AfstR']].agg(['min','mean','max','count']).reset_index()
    return grp
chkKafstRvsAfstR(ODiN2readpkl.allodinyr)

# +
#code for issue 0003
# -

#start with birds-eye distances file written by ROfietsb_Utchk
xypccoord=pd.read_excel("../intermediate/xypccoordpc4.xlsx")


def addvogelvl(dfin,wgcoord):
    c2 = wgcoord[['PC4','avgRfunX','avgRfunY']]
    cvert = c2.copy(deep=False)
    cvert.columns=['VertPC','VertPCDRSX','VertPCDRSY']
    caank = c2.copy(deep=False)
    caank.columns=['AankPC','AankPCDRSX','AankPCDRSY']
#    print(cvert)    
    df=dfin [(dfin ['Rit'] ==1) & (dfin ['AankPC'] !=0) & (dfin ['VertPC'] !=0)]
#    df['AankPC'] = df['AankPC'].astype(int)
#    df['VertPC'] = df['VertPC'].astype(int)
#    print(df)
    dfrds=df.merge(cvert,how='left').merge(caank,how='left')
    dfrds['AfstVV'] = np.sqrt((dfrds['VertPCDRSX']-dfrds['AankPCDRSX'])**2 +
                              (dfrds['VertPCDRSY']-dfrds['AankPCDRSY'])**2)
    dfrds['AfstVVwg'] = dfrds['AfstVV'] * dfrds['FactorV'] 
    dfrds['AfstRwg'] =  dfrds['AfstR'] * dfrds['FactorV'] 
    rv = dfrds[['VertPC','AankPC','AfstRwg','AfstVVwg','FactorV']]. groupby(['VertPC','AankPC']).\
           agg('sum') .reset_index()
    rv ['AfstVVwg'] = rv['AfstVVwg'] / rv['FactorV'] /1000
    rv ['AfstRwg'] = rv['AfstRwg'] / rv['FactorV'] /10
    rv=rv.merge(cvert,how='left').merge(caank,how='left')
    return (rv)
allodvv = addvogelvl(ODiN2readpkl.allodinyr,xypccoord) 

allodvv

allvvlim=2e6
def vvplt1(dfin,lim):
    df= dfin[(dfin['FactorV']>lim) & (dfin['AankPC'] != dfin['VertPC'] )
                                      & (dfin['AfstVVwg'] <15)].copy()
    df['lim']=df['AfstVVwg']+3
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df,y='AfstVVwg',x='AfstVVwg',ax=ax)
    sns.lineplot(data=df,y='lim',x='AfstVVwg',ax=ax)
    sns.scatterplot(data=df,y='AfstRwg',x='AfstVVwg',ax=ax)
vvplt1(allodvv,allvvlim)    

stryear='2020'
cbspc4data =pd.read_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl")
cbspc4data= cbspc4data.sort_values(by=['postcode4']).reset_index()

# +
prov0=cx.providers.nlmaps.grijs.copy()
print( cbspc4data.crs)
print (prov0)

#en nu netjes, met schaal in km
def plaxkm(x, pos=None):
      return '%.0f'%(x/1000.)

def addbasemkmsch(ax,mapsrc):
    cx.add_basemap(ax,source= mapsrc,crs="epsg:28992")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))


# -

chkpckrt1 = cbspc4data[(np.isin (cbspc4data['postcode4'],(7553) ))]
chkpckrt2 = cbspc4data[(np.isin (cbspc4data['postcode4'],(7556) ))]
fig, ax = plt.subplots(figsize=(16, 12))
pchkpckrt = chkpckrt1.plot(alpha=.3,color='blue',ax=ax)
pchkpckrt = chkpckrt2.plot(alpha=.3,color='green',ax=ax)
addbasemkmsch(ax,prov0)


# +
       
def vvcplt1(dfin,lim,reg,minomr,fname,tit):    
    df2= dfin[(dfin['FactorV']>lim) & (dfin['AankPC'] != dfin['VertPC'] )
                                      & (dfin['AfstVVwg'] <15)].copy()
    
    df=df2
    fig, ax = plt.subplots(figsize=(12, 12))

    df2['cmaxed']= df2['AfstRwg']/ df2['AfstVVwg']
    if minomr==0:
        df2['cmaxed'] = np.minimum(df2['cmaxed'],2)
        df=df2
    else:
        df=df2[df2['cmaxed']>minomr]
    
    chkpckrt2 = cbspc4data[(np.isin (cbspc4data['postcode4'], list(df['AankPC'])))]        
    pchkpckrt = chkpckrt2.plot(alpha=.2,color='green',ax=ax)
    chkpckrt1 = cbspc4data[(np.isin (cbspc4data['postcode4'], list(df['VertPC'])))]        
    pchkpckrt = chkpckrt1.plot(alpha=.2,color='red',ax=ax)
        
        
    df['lim']=df['AfstVVwg']+3
#    ax = df.plot(    figsize= (12, 12),    alpha  = 0.1      )
    plt.quiver(df['VertPCDRSX'],df['VertPCDRSY'],
               df['AankPCDRSX']- df['VertPCDRSX'],df['AankPCDRSY']-df['VertPCDRSY'],
               df['cmaxed'],cmap='jet',
               angles='xy',scale_units='xy', scale=1.)
    ax.set_title(tit)    

    df['AankPCs']=df['AankPC'].astype(int).astype(str)
    aanks = df[['VertPC','AankPCs']].groupby(['VertPC'], as_index=False).agg(', '.join)
    vgrp = df.groupby(['VertPC']).agg('mean').reset_index().merge(aanks)
    
    if minomr!=0:        
        xmin, xmax, ymin, ymax = plt.axis()
        #label each point in scatter plot
        for idx, row in vgrp.iterrows():
            if ( row['VertPCDRSX']> xmin and
               row['VertPCDRSX'] <xmax and row['VertPCDRSY']> ymin and
               row['VertPCDRSY'] <ymax  ):
                ax.annotate("%.0f-%s"%(row['VertPC'],row['AankPCs']), 
                        (row['VertPCDRSX'],row['VertPCDRSY']),size="small")

    ax.set_aspect("equal")
    plt.colorbar()
    setaxreg(ax,reg)
    addbasemkmsch(ax,prov0)
    figname = "../output/showgrids/fig_"+fname+'-'+reg+'.png';
    fig.savefig(figname,dpi=300) 
    

#vvcplt1(allodvv,allvvlim*.7,'u10',0,"omrgem",
#       "Ratio reisafst vs volgevl zwaartept voor > %.1f mln ritten, < 15 km"%(allvvlim*.7/1e6))  
vvcplt1(allodvv,allvvlim*.7,'u10',1.9,"omruitsch",
       "Uitschieters reisafst vs volgevl zwaartept voor > %.1f mln ritten, < 15 km"%(allvvlim*.7/1e6))  


# -

def vvpltexcepy1(dfin,lim):
    df= dfin[(dfin['FactorV']>lim) & (dfin['AankPC'] != dfin['VertPC'] )
                                      & (dfin['AfstVVwg'] <5)& (dfin['AfstRwg'] > dfin['AfstVVwg']+3)]
    rv = df
    return rv
vvpltexcepy1(allodvv,allvvlim)    


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



def fitAfstR(dfin,lim,fitmode):
    df= dfin[(dfin['FactorV']>lim) & (dfin['AankPC'] != dfin['VertPC'] )
                                      & (dfin['AfstVVwg'] <15)].copy()
    if fitmode=='VGenlin':
        df ['AfstVVwg'] = df['AfstVVwg'] * df['FactorV'] 
        df ['AfstRwg'] = df['AfstRwg'] * df['FactorV'] 
    if 1==1:
        fcols=['FactorV','AfstVVwg']
        pcols=['ParamCnst','ParamAfstVVwg']
        rf= _regressgrp (df, 'AfstRwg', fcols, pcols)   
        print (rf)
        #fitdf.merge(rf,how='outer',on=[])
        df['FitAfstRwg'] = df['FactorV'] * rf['ParamCnst'][0] +  df['AfstVVwg'] * rf['ParamAfstVVwg'][0]
        dfin['FitAfstRwg'] = 1 * rf['ParamCnst'][0] +  dfin['AfstVVwg'] * rf['ParamAfstVVwg'][0]

    diffs = df['FitAfstRwg'] - df['AfstRwg']    
    chisq = np.sum(diffs*diffs) / np.sum(df['AfstRwg']* df['AfstRwg'])
    print(chisq)
    return (chisq,rf)

#tofit=infotots2pcdiffng.copy(deep=False)
#noot: VGensq mag er wel mooier uit zien, maar de chi^2 is een factor 4 slechter

(chisq,rf)= fitAfstR(allodvv,allvvlim,'VGenlin')
#r1=mkactpccmpfig(infotots2pcdiffng,'fitted VGenlin')
# -

def vvplt2(dfin,lim):
    df= dfin[(dfin['FactorV']>lim) & (dfin['AankPC'] != dfin['VertPC'] )
                                      & (dfin['AfstVVwg'] <15)].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df,y='AfstVVwg',x='AfstVVwg',ax=ax)
    sns.lineplot(data=df,y='FitAfstRwg',x='AfstVVwg',ax=ax)
    sns.scatterplot(data=df,y='AfstRwg',x='AfstVVwg',ax=ax)
vvplt2(allodvv,allvvlim) 

fietsodvv = addvogelvl(ODiN2readpkl.allodinyr[ODiN2readpkl.allodinyr['KHvm'].isin([5,6])],xypccoord) 

fietslim=1e6
vvplt1(fietsodvv,fietslim)    

(chisq,rf)= fitAfstR(fietsodvv,fietslim,'VGenlin')

vvplt2(fietsodvv,fietslim) 


def vvpltexcepy2(dfin,lim):
    df= dfin[(dfin['FactorV']>lim) & (dfin['AankPC'] != dfin['VertPC'] )
                                      & (dfin['AfstVVwg'] <1.5)& (dfin['AfstRwg'] > dfin['AfstVVwg']+1.0)]
    rv = df
    return rv
vvpltexcepy2(fietsodvv,fietslim) 

vvcplt1(fietsodvv,fietslim*.5,'htn') 

loopodvv = addvogelvl(ODiN2readpkl.allodinyr[ODiN2readpkl.allodinyr['KHvm']==5],xypccoord) 

looplim=1e6
vvplt1(loopodvv,looplim)    

# +
#issue 0003 conclusies
#Om goede statistiek te hebben zijn tientallen ritten nodig, die zijn er maar voor een beperkt
#aantal PC4 combinaties
#Over het algemeen zijn de ODIN afstanden daar iets langer dan vogelvlucht
#Afwijkingen met ODIN > VV bij specifieke liggingen waarbij omrijden logisch is, 
#of bij relatief uitgestrekte PC4 gebieden met veel kruisverbanden
#maar te weinig PC4 combinaties om hier trends of kaarten uit te halen
#Afwijkingen met ODIN < VV bij specifieke liggingen onderling populaire locaties
#  dicht bij onderlinge grens liggen
#lopen en fietsen lijken iets consistenter beeld te geven (minder omrijden via snellere wegen)
#maar dan (door selectie) ook weer minder data punten
#ofwel: de aanpak met vogelvlucht is niet geautomatiseerd structureel te verbeteren
# -

print("Finished")
