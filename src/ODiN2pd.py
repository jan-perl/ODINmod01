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
#leest ODiN gegevens en schrijft naar pickles
#las eerder ook RUDiFUN en gesmoothde RUDIFUN
# -

import pandas as pd
import numpy as np
import os as os
import re as re
import seaborn as sns

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

ODiN2readpd_skiprd=1
import ODiN2readpkl
print (ODiN2readpkl.fietswijk1pc4)
#de bestanden die ODiN2readpkl in leest worden niet gebruikt, de constanten wel

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

dbk_2022.to_pickle("../intermediate/dbk_allyr.pkl")

print ("Finished")


