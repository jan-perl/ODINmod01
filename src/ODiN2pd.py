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
import os as os

# +
#system(pip install --upgrade pip setuptools wheel)

# +
#system(pip install spss-converter)

# +
#to be run as tooy inside container: docker exec -u 0 -it jupyter03 bash
#system(apt-get install libz-dev)

# +
#system(pip uninstall pyreadstat)
# -

os.system("pip install pandas-ods-reader")

from pandas_ods_reader import read_ods

# +
#system(pip install pyreadstat==0.3.4)
# -

#system(pip install spss-converter)
print (pd.__version__) 

df_2018 = pd.read_csv("../data/ODiN2018_Databestand_v2.0.csv", encoding = "ISO-8859-1", sep=";")  
df_2019 = pd.read_csv("../data/ODiN2019_Databestand_v2.0.csv", encoding = "ISO-8859-1", sep=";")  
df_2020 = pd.read_csv("../data/ODiN2020_Databestand_v2.0.csv", encoding = "ISO-8859-1", sep=";")  
df_2021 = pd.read_csv("../data/ODiN2021_Databestand.csv", encoding = "ISO-8859-1", sep=";")  
df_2022 = pd.read_csv("../data/ODiN2022_Databestand.csv", encoding = "ISO-8859-1", sep=";")  

df_2019

df_2019.dtypes

dbk_2022 = read_ods("../data/ODiN_2022/ODiN2022_Codeboek_v1.0.ods","Codeboek_ODiN_2022") 

dbk_2022_cols = dbk_2022 [~ dbk_2022.Variabele_naam_ODiN_2022.isna()]
dbk_2022_cols [ dbk_2022_cols.Niveau.isna()]

dbk_2022_cols.Niveau.unique()

# +
#note: to clean: copied in ODiN2readpkl
excols= ['Wogem', 'AutoHhl', 'MRDH', 'Utr', 'FqLopen', 'FqMotor', 'WrkVervw', 'WrkVerg', 'VergVast', 
         'VergKm', 'VergBrSt', 'VergOV', 'VergAans', 'VergVoer', 'VergBudg', 'VergPark', 'VergStal', 'VergAnd', 
         'BerWrk', 'RdWrkA', 'RdWrkB', 'BerOnd', 'RdOndA', 'RdOndB', 'BerSup', 'RdSupA', 'RdSupB',
         'BerZiek', 'RdZiekA', 'RdZiekB', 'BerArts', 'RdArtsA', 'RdArtsB', 'BerStat', 'RdStatA', 'RdStatB', 
         'BerHalte', 'RdHalteA', 'RdHalteB', 'BerFam', 'RdFamA', 'RdFamB', 'BerSport', 'RdSportA', 'RdSportB',
          'VertMRDH', 'VertUtr', 'AankMRDH', 'AankUtr' ]

def miscols(df,jaar,dbk,vnamcol):
    chkcols = dbk [  ~ ( dbk[vnamcol].isin( excols) )][vnamcol]
#kolommen zoner beschrijving
    acols = df.columns[ ~ (df.columns.isin(chkcols) )]
    mcols = list(chkcols[ ~ (chkcols.isin(df.columns) )])
    ocols = df.dtypes[df.dtypes == 'object' ]
    for chkcol in ocols.index:
        df[chkcol] = pd.to_numeric(df[chkcol],errors='coerce')
        nas=len(df[df[chkcol].isna()].index)
        if (nas !=0) & False:
            print(chkcol,"has nas:",nas)        
    print(jaar,acols.size,acols,len(mcols),mcols,ocols.size,ocols.index)
    
miscols(df_2022,2022,dbk_2022_cols,'Variabele_naam_ODiN_2022')
miscols(df_2021,2021,dbk_2022_cols,'Variabele_naam_ODiN_2022')
miscols(df_2020,2020,dbk_2022_cols,'Variabele_naam_ODiN_2022')
miscols(df_2019,2019,dbk_2022_cols,'Variabele_naam_ODiN_2022')
miscols(df_2018,2018,dbk_2022_cols,'Variabele_naam_ODiN_2022')


# +
#TODO hernoem wat kolommen

# +
#TODO parse ook data labels
# -

allodinyr=pd.concat([df_2018,df_2019,df_2020,df_2021,df_2022], ignore_index=True)
len(allodinyr.index)


allodinyr.to_pickle("../intermediate/allodinyr.pkl")

dbk_2022_cols.to_pickle("../intermediate/dbk_allyr_cols.pkl")

# +
#nu postcode match hulptabel
# -

pc6gwb2020 = pd.read_csv("../data/pc6-gwb2020.csv", encoding = "ISO-8859-1", sep=";")  
pc6gwb2020['PC4'] = pc6gwb2020['PC6'].str[0:4]

pc6gwb2020

pc4bumatch = pc6gwb2020[['Buurt2020','PC4','PC6']].groupby(['Buurt2020','PC4']).count().reset_index()
#pc4bumatch = pc4bumatch.assign(BU_CODE= pc4bumatch['Buurt2020'])
pc4bumatch['BU_CODE']  = pc4bumatch['Buurt2020'].apply( lambda x: "BU%08i" % x)

pc4bumatch

import sys
print(sys.path)
sys.path.append('/home/jovyan/work/pyshp')
import shapefile

sf = shapefile.Reader("../inputs/fietswijk1.dbf")

sf.fields

fieldnames = [f[0] for f in sf.fields[1:]]
fieldnames

fietswijk1bu = pd.DataFrame( sf.records() )
fietswijk1bu.columns = fieldnames
fietswijk1bufor4 = fietswijk1bu.merge(pc4bumatch,how='left')
fietswijk1bufor4['S_MXI22_BWT'] =fietswijk1bufor4['S_MXI22_B'] * fietswijk1bufor4['O_MXI22T']
fietswijk1bufor4['S_MXI22_BAT'] =fietswijk1bufor4['S_MXI22_B'] * (fietswijk1bufor4['O_MXI22N'] - fietswijk1bufor4['O_MXI22T'] )
fietswijk1bufor4['S_MXI22_BWN'] =fietswijk1bufor4['O_MXI22T']
fietswijk1bufor4['S_MXI22_BAN'] =(fietswijk1bufor4['O_MXI22N'] - fietswijk1bufor4['O_MXI22T'] )
fietswijk1pc4 = fietswijk1bufor4[['PC4','S_MXI22_BWT','S_MXI22_BAT','S_MXI22_BWN','S_MXI22_BAN','PC6']].groupby('PC4').sum().reset_index()
fietswijk1pc4['S_MXI22_BW'] =fietswijk1pc4['S_MXI22_BWT'] / fietswijk1pc4['S_MXI22_BWN']
fietswijk1pc4['S_MXI22_BA'] =fietswijk1pc4['S_MXI22_BAT'] /  fietswijk1pc4['S_MXI22_BAN']                                                                 
fietswijk1pc4['S_MXI22_GW'] = pd.qcut(fietswijk1pc4['S_MXI22_BW'], 10)
fietswijk1pc4['S_MXI22_GA'] = pd.qcut(fietswijk1pc4['S_MXI22_BA'], 10)
fietswijk1pc4

fietswijk1pc4.to_pickle("../intermediate/fietswijk1pc4.pkl")

fietswijk1pc4c2 = fietswijk1pc4.groupby('S_MXI22_GW').mean().reset_index()
fietswijk1pc4c2['MXIGRP']  = fietswijk1pc4c2['S_MXI22_BWN']  / (fietswijk1pc4c2['S_MXI22_BWN']  + fietswijk1pc4c2['S_MXI22_BAN'] )
fietswijk1pc4c2 


