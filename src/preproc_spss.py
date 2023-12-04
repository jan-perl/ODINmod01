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

dbk_2022 = ODiN2readpkl.dbk_allyr_cols
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


