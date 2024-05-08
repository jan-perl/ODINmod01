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
#leesrouteines voor diverse CBS gegevens
#en conversie routines buurt-PC4,6
# -

import pandas as pd
import numpy as np
import os as os
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

import sys
print(sys.path)
sys.path.append('/home/jovyan/work/pyshp')
import shapefile

os.system("pip install geopandas")

os.system("pip install contextily")

import geopandas
import contextily as cx
plt.rcParams['figure.figsize'] = [10, 6]

if(1==1):
    try:
        os.mkdir("../intermediate/CBS")
    except:
        print(" replacing existing ");

# +
#nog te leren van https://github.com/VNG-Realisatie/grenswijzigen

# +
#TODO CBS lezen arbeidsplaatsen en inwoners
#TODO CBS relateren arbeidsplaatsen en inwoners aan BAG getallen

# +
#niet meer gebruikt
#wijkendata = shapefile.Reader("../data/CBS/wijken_2023_v1.dbf")

# +
#wijkendata.fields
# -

#eerste read is alleen nodig om kolom namen op te pakken


# +
def cnvpc4stats(year):
#eerste read is alleen nodig om kolom namen op te pakken    
    if year==2023:
        data_pc4_2023_1 =pd.read_excel('../data/CBS/PC4STATS/pc4_2023_v1.xlsx',
                             skiprows=7)
        data_pc4 =pd.read_excel('../data/CBS/PC4STATS/pc4_2023_v1.xlsx',
                             skiprows=8)
        data_pc4.columns = data_pc4_2023_1.columns
    elif year==2022:
#eerste read is alleen nodig om kolom namen op te pakken
        data_pc4_2022_1 =pd.read_excel('../data/CBS/PC4STATS/pc4_2022_v1.xlsx',
                             skiprows=7)
        data_pc4 =pd.read_excel('../data/CBS/PC4STATS/pc4_2022_v1.xlsx',
                             skiprows=8)
        data_pc4.columns = data_pc4_2022_1.columns
    elif year==2021:
        data_pc4_2021_1 =pd.read_excel('../data/CBS/PC4STATS/pc4_2021_v1.xlsx',
                             skiprows=7)
        data_pc4 =pd.read_excel('../data/CBS/PC4STATS/pc4_2021_v1.xlsx',
                             skiprows=8)
        data_pc4.columns = data_pc4_2021_1.columns
    elif year==2020:
        data_pc4_2020_1 =pd.read_excel('../data/CBS/PC4STATS/pc4_2020_vol.xlsx',
                             skiprows=8)
        data_pc4 =pd.read_excel('../data/CBS/PC4STATS/pc4_2020_vol.xlsx',
                             skiprows=9)
        data_pc4.columns = data_pc4_2020_1.columns
    stryear=str(year)    
    data_pc4.to_pickle("../intermediate/CBS/pc4stats_"+stryear+".pkl")    
    return (data_pc4)

data_pc4_2020 = cnvpc4stats(2020)
data_pc4_2022 = cnvpc4stats(2022)
# -

data_pc4_2022.dtypes

dat_83504 = pd.read_csv('../data/CBS/83504NED/Observations.csv',sep=';')
dat_83504_mc = pd.read_csv('../data/CBS/83504NED/MeasureCodes.csv',sep=';')

dat_83504.dtypes

dat_83504_mc.dtypes

dat_83504_mc

dat_85560 = pd.read_csv('../data/CBS/85560NED/Observations.csv',sep=';')
dat_85560_mc = pd.read_csv('../data/CBS/85560NED/MeasureCodes.csv',sep=';')
dat_85560['Value'] = pd.to_numeric(dat_85560['Value'].str.replace(",","."))

dat_85560_mc

dat_85560


# +
def cnvgwb(year):
    if year==2023:
        g = geopandas.read_file("../data/CBS/wijkbuurt/gemeenten_2023_v1.dbf")
        w = geopandas.read_file("../data/CBS/wijkbuurt/wijken_2023_v1.dbf")
        b = geopandas.read_file("../data/CBS/wijkbuurt/buurten_2023_v1.dbf")
        b.replace(to_replace=-99999999,value=pd.NA,inplace=True)
    elif year==2022:
        g = geopandas.read_file("../data/CBS/wijkbuurt/gemeenten_2022_v2.dbf")
        w = geopandas.read_file("../data/CBS/wijkbuurt/wijken_2022_v2.dbf")
        b = geopandas.read_file("../data/CBS/wijkbuurt/buurten_2022_v2.dbf")
        b.replace(to_replace=-99999999,value=pd.NA,inplace=True)
    elif year==2021:
        g = geopandas.read_file("../data/CBS/wijkbuurt/gemeenten_2021_v3.dbf")
        w = geopandas.read_file("../data/CBS/wijkbuurt/wijken_2021_v3.dbf")
        b = geopandas.read_file("../data/CBS/wijkbuurt/buurten_2021_v3.dbf")
        b.replace(to_replace=-99999999,value=pd.NA,inplace=True)
    elif year==2020:
        g = geopandas.read_file("../data/CBS/WijkBuurtkaart_2020_v3/gemeente_2020_v3.dbf")
        w = geopandas.read_file("../data/CBS/WijkBuurtkaart_2020_v3/wijk_2020_v3.dbf")
        b = geopandas.read_file("../data/CBS/WijkBuurtkaart_2020_v3/buurt_2020_v3.dbf")
        b.replace(to_replace=-99999999,value=pd.NA,inplace=True)  
    stryear=str(year)    
    g.to_pickle("../intermediate/CBS/gwb_gem_"+stryear+".pkl")    
    w.to_pickle("../intermediate/CBS/gwb_wijk_"+stryear+".pkl") 
    b.to_pickle("../intermediate/CBS/gwb_buurt_"+stryear+".pkl") 
    return ([g,w,b])

gemeentendata ,  wijkgrensdata ,    buurtendata = cnvgwb(2020)    
# -

for year in range(2021,2024):
    print(year)
    gemeentendata_c ,  wijkgrensdata_c ,    buurtendata_c = cnvgwb(year)    


# +
def cnvpc6hnryr(year):
    stryear=str(year)
    indat = pd.read_csv("../data/CBS/PC6HNR/pc6hnr"+stryear+
                        ("0801_gwb.csv" if (year!=2018) else "0801_gwb-vs2.csv" ), 
                        encoding = "ISO-8859-1", 
                        sep=(";" if (year<2023) else "," )  )
    indat=indat.rename(columns={"Buurt"+stryear:"Buurt","Wijk"+stryear:"Wijk",
                                "Gemeente"+stryear:"Gemeente"})  
    ngrp = indat.groupby(["PC6","Buurt", "Wijk","Gemeente"]).count().reset_index()
    ngrp['PC4'] = ngrp['PC6'].str[0:4].astype('int64')
#noot: 2023 data heeft vreemde waarden in buurt
    ngrp['Buurt'] = ngrp['Buurt'].astype('int64')
    ngrp['BU_CODE']  = ngrp['Buurt'].apply( lambda x: "BU%08i" % x)
    ngrp.to_pickle("../intermediate/CBS/pc6hnryr_"+stryear+".pkl") 
    return(ngrp)
                       
pc6hnryr =cnvpc6hnryr(2020) 
pc6hnryr.dtypes
# -

#2023 werkt nog niet
for year in range(2018,2023):
    pc6hnryr =cnvpc6hnryr(year) 
    print(pc6hnryr.dtypes)

print("Finished")
