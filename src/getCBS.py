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
        os.mkdir("../data/CBS")
    except:
        print(" replacing existing ");

fetchweb=False

#Kerncijfers per postcode
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/gegevens-per-postcode
pc4lnks = ["https://download.cbs.nl/postcode/2023-CBS_pc4_2022_v1.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc4_2021_v2.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc4_2020_vol.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc4_2019_vol.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2018-v3.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2017-v3.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2016-v2.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2015-v2.zip"
          ] 
print(pc4lnks[1:3])


def getcbspc4(link):
    of=re.sub("^.*/","",link)
    try:
        os.remove ("../data/CBS/"+of)
    except:
        print(" first time download ");        
    str1= "wget -q -O ../data/CBS/"+of+" "+link
    os.system(str1)
    str2= "bash -c 'cd ../data/CBS ; unzip -u "+of + "'"
    os.system(str2)
if fetchweb:
    for link in pc4lnks:
        getcbspc4(link)


#Huishoudens; huishoudenssamenstelling en viercijferige postcode, 1 januari
#from https://www.cbs.nl/nl-nl/cijfers/detail/83505NED?q=postcode
#https://datasets.cbs.nl/CSV/CBS/nl/83505NED
def getcbsset(link):
    of=re.sub("^.*/","",link)    
    try:
        os.mkdir ("../data/CBS/"+of)
    except:
        print(" replacing existing ");
    str1= "wget -q -O ../data/CBS/"+of+"/"+of+".zip"+" "+link
    os.system(str1)
    str2= "bash -c 'cd ../data/CBS/"+of+" ; unzip -u "+of + ".zip'"
    os.system(str2)
if fetchweb:
    getcbsset("https://datasets.cbs.nl/CSV/CBS/nl/83505NED")

#Bevolking; geslacht, positie huishouden, viercijferige postcode, 1 januari
#from https://www.cbs.nl/nl-nl/cijfers/detail/83504NED?q=postcode
#https://datasets.cbs.nl/CSV/CBS/nl/83504NED
if fetchweb:
    getcbsset("https://datasets.cbs.nl/CSV/CBS/nl/83504NED")

#Nabijheid voorzieningen; afstand locatie, wijk- en buurtcijfers 2022
#from https://dataportal.cbs.nl/detail/CBS/85560NED
if fetchweb:
    getcbsset("https://datasets.cbs.nl/CSV/CBS/nl/85560NED")

#from https://www.cbs.nl/nl-nl/maatwerk/2020/24/nabijheid-voorzieningen-buurtcijfers-2019
#https://www.cbs.nl/nl-nl/cijfers/detail/84718NED?dl=348F7
if fetchweb:
    getcbsset("https://datasets.cbs.nl/CSV/CBS/nl/84718NED")

#wijken en buurten kaarten komen via
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2023
#https://download.cbs.nl/regionale-kaarten/wijkbuurtkaart_2023_v1.zip
if fetchweb:
    getcbspc4("https://download.cbs.nl/regionale-kaarten/wijkbuurtkaart_2023_v1.zip")
#data from
#https://www.cbs.nl/nl-nl/maatwerk/2020/39/buurt-wijk-en-gemeente-2020-voor-postcode-huisnummer
#https://www.cbs.nl/-/media/_excel/2020/39/2020-cbs-pc6huisnr20200801-buurt.zip
if fetchweb:
    getcbspc4("https://www.cbs.nl/-/media/_excel/2020/39/2020-cbs-pc6huisnr20200801-buurt.zip")

# +
#TODO CBS lezen arbeidsplaatsen en inwoners
#TODO CBS relateren arbeidsplaatsen en inwoners aan BAG getallen
# -

wijkendata = shapefile.Reader("../data/CBS/wijken_2023_v1.dbf")

wijkendata.fields

data_pc4_2022_1 =pd.read_excel('../data/CBS/pc4_2022_v1.xlsx',
                             skiprows=7)
data_pc4_2022 =pd.read_excel('../data/CBS/pc4_2022_v1.xlsx',
                             skiprows=8)
data_pc4_2022.columns = data_pc4_2022_1.columns

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

gemeentendata = geopandas.read_file("../data/CBS/gemeenten_2023_v1.dbf")
wijkgrensdata = geopandas.read_file("../data/CBS/wijken_2023_v1.dbf")

#wijkendata = geopandas.read_file("../data/CBS/wijken_2023_v1.dbf")
wijkendata = geopandas.read_file("../data/CBS/buurten_2023_v1.dbf")
wijkendata.replace(to_replace=-99999999,value=pd.NA,inplace=True)

# +
#print(wijkgrensdata.crs.to_string() )
#gemeentendata.to_crs(epsg=3857)
# -

wijkendata.dtypes

wijkendata

#examples from https://geopandas.org/en/stable/docs/user_guide/mapping.html
wijkendata.plot()

wijkendata.plot(column="BEV_DICHTH",legend=True, cmap='OrRd')

gemset = ['Houten']
gemset = ['Houten','Bunnik','Nieuwegein']
gemset2 = ['Houten','Bunnik','Nieuwegein', 
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','Utrechtse Heuvelrug',
          'De Bilt']
gemset2 = ['Houten','Bunnik','Nieuwegein', 'Vijfheerenlanden',
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','De Bilt','Utrechtse Heuvelrug',
           'Stichtse Vecht' ]
wijkdata_sel=wijkendata[wijkendata['GM_NAAM'].isin (gemset)]
gemdata_sel=gemeentendata[gemeentendata['GM_NAAM'].isin(gemset)]
wijkgrdata_sel=wijkgrensdata[wijkgrensdata['GM_NAAM'].isin(gemset)]

base=wijkgrdata_sel.boundary.plot(color='green');
#cx.add_basemap(base, crs=wijkgrdata_sel.crs.to_string(), source=cx.providers.Stamen.TonerLite)

#base=gemdata_sel.boundary.plot(color='green');
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
wijkdata_sel.plot(ax=base ,column="BEV_DICHTH",legend=True, cmap='OrRd',
                  legend_kwds={"label": "Bevolkingsdichtheid"})

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
#wijkdata_sels=wijkdata_sel[wijkdata_sel['STED'] !=5]
wijkdata_sels=wijkdata_sel[wijkdata_sel['BEV_DICHTH'] >1000]
wijkdata_sels.plot(ax=base ,column="STED",legend=True, cmap='RdYlBu',
                  legend_kwds={"label": "Stedelijkheid"})

wijkdata_selfield='BU_CODE'
adcol01= ['D000025','A045790_3','A045791_3','A045792_5']
def adddta(df1,df2,addmeas):
    df1s=df1[['geometry','BU_CODE','BU_NAAM','GM_NAAM','AANT_INW']]
    df2flds = df2[df2['Measure'].isin(addmeas)]    
    df2tab= df2flds.pivot_table(index='WijkenEnBuurten', columns='Measure',values='Value')
    seljoin = df1s.merge(df2tab,left_on=wijkdata_selfield, right_on='WijkenEnBuurten' ,how="left")    
    return(seljoin)
wijkdata_withfield01= adddta(wijkdata_sels,dat_85560,adcol01)

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
wijkdata_withfield01.plot(ax=base,column=adcol01[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
wijkdata_withfield01['SUPER_LOOP'] = wijkdata_withfield01[adcol01[1]] !=0
wijkdata_withfield01.plot(ax=base,column='SUPER_LOOP',legend=True, cmap='RdYlBu',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

wijkdata_withfield01.groupby(['GM_NAAM','SUPER_LOOP']).agg({'AANT_INW':'sum'})

adcol02= ['D000038', 'A045790_4', 'A045791_4','A045792_6' ]
wijkdata_withfield02= adddta(wijkdata_sel,dat_85560,adcol02)
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
wijkdata_withfield02.plot(ax=base,column=adcol02[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal dagelijkse levensmiddelen < 1 km"}
           )          

adcol03= ['D000020',
'A045790_5',
'A045791_5',
'A045792_8',
'D000021',
'A045790_6',
'A045791_6',
'A045792_9',
'D000043',
'A045790_7',
'A045791_7',
'A045792_10' ]
wijkdata_withfield03= adddta(wijkdata_sel,dat_85560,adcol03)
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
wijkdata_withfield03.plot(ax=base,column=adcol03[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal cafes < 1 km"}
           )          


