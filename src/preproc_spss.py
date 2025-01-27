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


