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



allodinyr=ODiN2readpkl.allodinyr
len(allodinyr.index)

# +
fietswijk1pc4= ODiN2readpkl.fietswijk1pc4

fietswijk1pc4['S_MXI22_GW'] = pd.qcut(fietswijk1pc4['S_MXI22_BW'], 10)
fietswijk1pc4['PC4'] =  pd.to_numeric(fietswijk1pc4['PC4'] ,errors='coerce')
print(fietswijk1pc4.dtypes)
fwin4 = fietswijk1pc4 [['PC4','S_MXI22_GW','S_MXI22_BW']]
allodinyr2 = allodinyr.merge(fwin4,left_on='AankPC', right_on='PC4',how='left')
allodinyr2 = allodinyr2.rename ( columns= {'S_MXI22_GW': 'Aank_MXI22_GW', 'S_MXI22_BW': 'Aank_MXI22_BW' } )
allodinyr2 = allodinyr2.merge(fwin4,left_on='VertPC', right_on='PC4',how='left')
allodinyr2 = allodinyr2.rename (columns= {'S_MXI22_GW': 'Vert_MXI22_GW', 'S_MXI22_BW': 'Vert_MXI22_BW' } )
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

def mkpltverplp (df,collvar,normgrp):
    dfvrecs = df [df['Verpl']==1]
#    oprecs = df [df['OP']==1]
    pstats = dfvrecs[[normgrp, collvar,'FactorV']].groupby([normgrp, collvar]).sum().reset_index()
    #print(pstats)
    denoms= pstats [[normgrp, 'FactorV']].groupby([normgrp]).sum().reset_index().rename(columns={'FactorV':'Denom'} )
    #print(denoms)
    pstatsn = pstats.merge(denoms,how='left')
    pstatsn['FractV'] = pstatsn['FactorV'] *100.0/ pstatsn['Denom']
    #print(pstatsn)
    sns.catplot(data=pstatsn, y='FractV', x=collvar, hue=normgrp, kind="bar")
    return(pstatsn)
   
    
datpltverplp = mkpltverplp (allodinyr,'Doel','Jaar')
# -

datpltverplp = mkpltverplp (allodinyr,'MotiefV','Jaar')

datpltverplp = mkpltverplp (allodinyr,'KHvm','Jaar')

datpltverplp = mkpltverplp (allodinyr,'KAfstV','Jaar')

datpltverplp = mkpltverplp (allodinyr,'VertUur','Jaar')

#neem alleen ritten naar huis, Aankomst is dan wonen: 
naarhuis = allodinyr [allodinyr ['Doel'] ==1 ]
datpltverplp = mkpltverplp (naarhuis,'Aank_MXI22_GW','Jaar')

#naar huis: aankomst is wonen: 
datpltverplp = mkpltverplp (naarhuis,'KHvm','Aank_MXI22_GW')

#naar huis: aankomst is wonen: 
datpltverplp = mkpltverplp (naarhuis,'KAfstV','Aank_MXI22_GW')

#naar huis: vertrel is utiliteit: 
datpltverplp = mkpltverplp (naarhuis,'KHvm','Vert_MXI22_GW')

#neem alleen ritten naar werk, Aankomst is dan werken: 
naarwerk= allodinyr [allodinyr ['Doel'] ==2 ]
datpltverplp = mkpltverplp (naarwerk,'Aank_MXI22_GW','Jaar')

#neem alleen ritten naar werk, Aankomst is dan wonen
datpltverplp = mkpltverplp (naarwerk,'Vert_MXI22_GW','Jaar')

#neem alleen ritten naar werk, Aankomst is dan werken: 
datpltverplp = mkpltverplp (naarwerk,'KHvm','Aank_MXI22_GW')

#neem alleen ritten naar werk, Aankomst is dan werken: 
datpltverplp = mkpltverplp (naarwerk,'KAfstV','Aank_MXI22_GW')


