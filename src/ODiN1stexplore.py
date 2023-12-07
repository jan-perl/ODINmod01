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
import seaborn as sns

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
fietswijk1pc4= ODiN2readpkl.fietswijk1pc4

fietswijk1pc4['S_MXI22_GB'] = pd.qcut(fietswijk1pc4['S_MXI22_BB'], 10)
fietswijk1pc4['S_MXI22_GG'] = pd.qcut(fietswijk1pc4['S_MXI22_BG'], 10)
fietswijk1pc4['PC4'] =  pd.to_numeric(fietswijk1pc4['PC4'] ,errors='coerce')
print(fietswijk1pc4.dtypes)
fwin4 = fietswijk1pc4 [['PC4','S_MXI22_GB','S_MXI22_BB','S_MXI22_GG','S_MXI22_BG']]
allodinyr2 = allodinyr.merge(fwin4,left_on='AankPC', right_on='PC4',how='left')
allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GB': 'Aank_MXI22_GW', 'S_MXI22_BB': 'Aank_MXI22_BW' } )
allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GG': 'Aank_MXI22_GG', 'S_MXI22_BG': 'Aank_MXI22_BG' } )
allodinyr2 = allodinyr2.merge(fwin4,left_on='VertPC', right_on='PC4',how='left')
allodinyr2 = allodinyr2.rename (columns= {'S_MXI22_GB': 'Vert_MXI22_GW', 'S_MXI22_BB': 'Vert_MXI22_BW' } )
allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GG': 'Vert_MXI22_GG', 'S_MXI22_BG': 'Vert_MXI22_BG' } )
print(allodinyr2.dtypes)
len(allodinyr2.index)
# -

print(allodinyr2)
allodinyr = allodinyr2


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
#kijk eens naar verdelingen totaal aantal verplaatsingen per persoon

def mkpltverplp (df,myspecvals,collvar,normgrp,selstr):
    dfvrecs = df [df['Verpl']==1]
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
   
    
datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','Jaar','Alle ritten')
#datpltverplp = mkpltverplp (allodinyr,specvaltab,'VertUur','Jaar','Alle ritten')
#datpltverplp = mkpltverplp (naarhuis,specvaltab,'Aank_MXI22_GW','Jaar','Alle ritten')
# -

datpltverplp = mkpltverplp (allodinyr,specvaltab,'MotiefV','Jaar','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'KHvm','Jaar','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'KAfstV','Jaar','Alle ritten')

#maak andere grafiek met zo veel categorieen
datpltverplp = mkpltverplp (allodinyr,specvaltab,'VertUur','Jaar','Alle ritten')

#neem alleen ritten naar huis, Aankomst is dan wonen: 
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
datpltverplp = mkpltverplp (naarhuis,specvaltab,'Aank_MXI22_GW','Jaar','Naar huis')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'Vert_MXI22_GW','Aank_MXI22_GW','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','Aank_MXI22_GW','Alle ritten')

datpltverplp = mkpltverplp (allodinyr,specvaltab,'Doel','Vert_MXI22_GW','Alle ritten')

#naar huis: aankomst is wonen: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KHvm','Aank_MXI22_GW','Naar huis')

#naar huis: aankomst is wonen: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KAfstV','Aank_MXI22_GW','Naar huis')

#naar huis: aankomst is wonen, als functie van oppervlakte dichtheid: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KAfstV','Aank_MXI22_GG','Naar huis')

#naar huis: vertrel is utiliteit: 
datpltverplp = mkpltverplp (naarhuis,specvaltab,'KHvm','Vert_MXI22_GW','Naar huis')

#neem alleen ritten naar werk, Aankomst is dan werken: 
naarwerk= allodinyr [allodinyr ['Doel'] ==2 ]
datpltverplp = mkpltverplp (naarwerk,specvaltab,'Aank_MXI22_GW','Jaar','Naar werk')

#neem alleen ritten naar werk, Aankomst is dan wonen
datpltverplp = mkpltverplp (naarwerk,specvaltab,'Vert_MXI22_GW','Jaar','Naar werk')

#neem alleen ritten naar werk, Aankomst is dan werken: 
datpltverplp = mkpltverplp (naarwerk,specvaltab,'KHvm','Aank_MXI22_GW','Naar werk')

#neem alleen ritten naar werk, Aankomst is dan werken: 
datpltverplp = mkpltverplp (naarwerk,specvaltab,'KAfstV','Aank_MXI22_GW','Naar werk')


