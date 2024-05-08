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

import sys
print(sys.path)
sys.path.append('/home/jovyan/work/pyshp')
import shapefile

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


import viewCBS

#import conversion routines buurt to PC4/6
import viewCBS

pc6gwb2020 = pd.read_csv("../data/CBS/PC6HNR/pc6-gwb2020.csv", encoding = "ISO-8859-1", sep=";")  
pc6gwb2020['PC4'] = pc6gwb2020['PC6'].str[0:4].astype('int64')

pc6gwb2020

pc4bumatch = pc6gwb2020[['Buurt2020','PC4','PC6']].groupby(['Buurt2020','PC4']).count().reset_index()
#pc4bumatch = pc4bumatch.assign(BU_CODE= pc4bumatch['Buurt2020'])
pc4bumatch['BU_CODE']  = pc4bumatch['Buurt2020'].apply( lambda x: "BU%08i" % x)

pc4bumatch

sf = shapefile.Reader("../inputs/fietswijk1.dbf")

sf.fields

fieldnames = [f[0] for f in sf.fields[1:]]
fieldnames

fietswijk1bu = pd.DataFrame( sf.records() )
fietswijk1bu.columns = fieldnames
fietswijk1bufor4 = fietswijk1bu.merge(pc4bumatch,how='left')
fietswijk1bufor4['S_MXI22_BWT'] =fietswijk1bufor4['S_MXI22_B'] *  fietswijk1bufor4['O_MXI22T']
fietswijk1bufor4['S_MXI22_BAT'] =fietswijk1bufor4['S_MXI22_B'] * (fietswijk1bufor4['O_MXI22N'] - fietswijk1bufor4['O_MXI22T'] )
fietswijk1bufor4['S_MXI22_BBT'] =fietswijk1bufor4['S_MXI22_B'] * (fietswijk1bufor4['O_MXI22N'] )
fietswijk1bufor4['S_MXI22_BWN'] = fietswijk1bufor4['O_MXI22T']
fietswijk1bufor4['S_MXI22_BAN'] =(fietswijk1bufor4['O_MXI22N'] - fietswijk1bufor4['O_MXI22T'] )
fietswijk1bufor4['S_MXI22_BBN'] =(fietswijk1bufor4['O_MXI22N']  )
fietswijk1bufor4['S_MXI22_BGN'] =(fietswijk1bufor4['AREA_GEO']  )
fietswijk1pc4 = fietswijk1bufor4[['PC4','S_MXI22_BWT','S_MXI22_BAT','S_MXI22_BWN','S_MXI22_BAN',
                                        'S_MXI22_BBT','S_MXI22_BBN','S_MXI22_BGN','PC6']].groupby('PC4').sum().reset_index()
#fietswijk1pc4['S_MXI22_BW'] =fietswijk1pc4['S_MXI22_BWT'] / fietswijk1pc4['S_MXI22_BWN']
#fietswijk1pc4['S_MXI22_BA'] =fietswijk1pc4['S_MXI22_BAT'] / fietswijk1pc4['S_MXI22_BAN']  
fietswijk1pc4['S_MXI22_BB'] = fietswijk1pc4['S_MXI22_BBT'] / fietswijk1pc4['S_MXI22_BBN']  
fietswijk1pc4['S_MXI22_BG'] = fietswijk1pc4['S_MXI22_BBN'] / fietswijk1pc4['S_MXI22_BGN'] 
fietswijk1pc4['S_MXI22_GB'] = pd.qcut(fietswijk1pc4['S_MXI22_BB'], 10)
fietswijk1pc4['S_MXI22_GG'] = pd.qcut(fietswijk1pc4['S_MXI22_BG'], 10)
fietswijk1pc4

fietswijk1pc4.to_pickle("../intermediate/fietswijk1pc4.pkl")

#dit is niet goed: de gemiddelden van 'MXIGRP' liggen niet bij midden in intervallen, maar er onder.
#wat is hier aan de hand ???
#wel goed: het gemiddeld aantal PC6-en loopt op als er meer woningen zijn
fietswijk1pc4c2 = fietswijk1pc4.groupby('S_MXI22_GB').mean().reset_index()
fietswijk1pc4c2['MXIGRP']  = fietswijk1pc4c2['S_MXI22_BWN']  / (fietswijk1pc4c2['S_MXI22_BWN']  + fietswijk1pc4c2['S_MXI22_BAN'] )
fietswijk1pc4c2 

#eigen postcode 4 vs gemoothd met omgeving
fietswijk1pc4['S_MXI22_NS'] = fietswijk1pc4['S_MXI22_BWN']  / (fietswijk1pc4['S_MXI22_BWN']  + fietswijk1pc4['S_MXI22_BAN'] )
sns.scatterplot(data=fietswijk1pc4,x='S_MXI22_NS',y='S_MXI22_BB')

fietswijk1pc4[fietswijk1pc4['S_MXI22_BB']<.05]

#sns.scatterplot(data=fietswijk1pc4,x='S_MXI22_BB',y='S_MXI22_BG')
sns.scatterplot(data=fietswijk1pc4,x='S_MXI22_NS',y='S_MXI22_BG')

#input data: per buurt
fietswijk1bufor4['S_MXI22_NS'] = fietswijk1bufor4['S_MXI22_BWN']  / (fietswijk1bufor4['S_MXI22_BWN']  + fietswijk1bufor4['S_MXI22_BAN'] )
fietswijk1bufor4['S_MXI22_NSAD'] = fietswijk1bufor4['S_MXI22_NS'] -  fietswijk1bufor4['S_MXI22_BWN']  / ( fietswijk1bufor4['S_MXI22_BBN'] )
print(fietswijk1bufor4['S_MXI22_NSAD'].std() )
sns.scatterplot(data=fietswijk1bufor4,x='S_MXI22_NS',y='S_MXI22_B')

# +
#nu eens kijken hoe de nieuwe routines het doen
# -

pc6hnryr =ODiN2readpkl.getpc6hnryr(2021) 
nhnrperbuurt =pc6hnryr[['Huisnummer','Buurt']].groupby(['Buurt']).sum().reset_index().rename(
    columns={'Huisnummer':'nhuisnrbuurt'} )
pc6hnryrmstats= pc6hnryr.merge(nhnrperbuurt,how='left')
sffields_sum =['AREA_GEO','O_MXI22T','O_MXI22N']  
sffields_rel =['S_MXI22_B']  
RFbu2021pc6 = viewCBS.distrpc6(pc6hnryrmstats,fietswijk1bu,'BU_CODE',sffields_sum,sffields_rel )
RFbu2021pc4 = viewCBS.cnvpc4(RFbu2021pc6,sffields_sum,sffields_rel )

RFbu2021pc4.dtypes

fietswijk1bufor4.dtypes

fietswijk1bufor4nona = fietswijk1bufor4[ ~ fietswijk1bufor4['PC4'].isna() ]

fietswijk1bufor4['PC4'] =fietswijk1bufor4nona['PC4'].astype('int64')
RFbu2021pc4cmeth=  RFbu2021pc4.merge(fietswijk1bufor4,how='outer')

RFbu2021pc4cmeth['d-S_MXI22_B'] = RFbu2021pc4cmeth['S_MXI22_B'] - 
RFbu2021pc4cmeth['d-S_MXI22_NS'] = RFbu2021pc4cmeth['O_MXI22T'] / RFbu2021pc4cmeth['O_MXI22TN'] -RFbu2021pc4cmeth['S_MXI22_NS']  
RFbu2021pc4cmeth.sort_values(by=)
