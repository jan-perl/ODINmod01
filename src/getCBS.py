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


def getcbspc4(link,uzdir):
    of=re.sub("^.*/","",link)
    try:
        os.remove ("../data/CBS/"+of)
    except:
        print(" first time download ");        
    str1= "wget -q -O ../data/CBS/"+of+" "+link
    os.system(str1)
    if uzdir=='':
        str2= "bash -c 'cd ../data/CBS ; unzip -u "+of + "'"
    else:
        try:
            os.mkdir ("../data/CBS/"+uzdir)
        except:
            print(" replacing existing ");
        str2= "bash -c 'cd ../data/CBS/"+uzdir+" ; unzip -u ../"+of + "'"
    os.system(str2)
if fetchweb :
    for link in pc4lnks:
        getcbspc4(link,'PC4STATS')


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
if fetchweb & False:
    getcbspc4("https://download.cbs.nl/regionale-kaarten/wijkbuurtkaart_2023_v1.zip",'')
#data from
#https://www.cbs.nl/nl-nl/maatwerk/2020/39/buurt-wijk-en-gemeente-2020-voor-postcode-huisnummer
#https://www.cbs.nl/-/media/_excel/2020/39/2020-cbs-pc6huisnr20200801-buurt.zip
if fetchweb & False:
    getcbspc4("https://www.cbs.nl/-/media/_excel/2020/39/2020-cbs-pc6huisnr20200801-buurt.zip",'')

#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2019
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2020
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2021
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2022
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2023
geolst = [
    'https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/wijk-en-buurtstatistieken/wijkbuurtkaart_2019_v3.zip'
    'https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/wijk-en-buurtstatistieken/wijkbuurtkaart_2020_v3.zip',
#    'https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/wijk-en-buurtstatistieken/wijkbuurtkaart_2020_v3.zip',
    'https://download.cbs.nl/regionale-kaarten/wijkbuurtkaart_2021_v3.zip',
    'https://download.cbs.nl/regionale-kaarten/wijkbuurtkaart_2022_v2.zip',
    'https://download.cbs.nl/regionale-kaarten/wijkbuurtkaart_2023_v1.zip'
]
if fetchweb:
    for link in geolst:
        getcbspc4(link,'wijkbuurt')

#https://www.cbs.nl/nl-nl/maatwerk/2018/36/buurt-wijk-en-gemeente-2018-voor-postcode-huisnummer
# https://www.cbs.nl/nl-nl/maatwerk/2019/42/buurt-wijk-en-gemeente-2019-voor-postcode-huisnummer
#https://www.cbs.nl/nl-nl/maatwerk/2020/39/buurt-wijk-en-gemeente-2020-voor-postcode-huisnummer
#https://www.cbs.nl/nl-nl/maatwerk/2021/36/buurt-wijk-en-gemeente-2021-voor-postcode-huisnummer
#https://www.cbs.nl/nl-nl/maatwerk/2022/37/buurt-wijk-en-gemeente-2022-voor-postcode-huisnummer
#https://www.cbs.nl/nl-nl/maatwerk/2023/35/buurt-wijk-en-gemeente-2023-voor-postcode-huisnummer
adrwijkpclst = [
 'https://www.cbs.nl/-/media/_excel/2018/36/2018-cbs-pc6huisnr20180801_buurt--vs2.zip',
 'https://www.cbs.nl/-/media/_excel/2019/42/2019-cbs-pc6huisnr20190801_buurt.zip',
 'https://www.cbs.nl/-/media/_excel/2020/39/2020-cbs-pc6huisnr20200801-buurt.zip',
 'https://www.cbs.nl/-/media/_excel/2021/36/2021-cbs-pc6huisnr20200801_buurt.zip',
 'https://www.cbs.nl/-/media/_excel/2022/37/2022-cbs-pc6huisnr20210801_buurt.zip',
 'https://www.cbs.nl/-/media/_excel/2023/35/2023-cbs-pc6huisnr20230801_buurt.zip' ]
if  fetchweb:
    for link in adrwijkpclst:
        getcbspc4(link,'PC6HNR')

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
data_pc4_2020_1 =pd.read_excel('../data/CBS/PC4STATS/pc4_2020_vol.xlsx',
                             skiprows=8)
data_pc4_2020 =pd.read_excel('../data/CBS/PC4STATS/pc4_2020_vol.xlsx',
                             skiprows=9)
data_pc4_2020.columns = data_pc4_2020_1.columns
data_pc4_2020

#eerste read is alleen nodig om kolom namen op te pakken
data_pc4_2022_1 =pd.read_excel('../data/CBS/PC4STATS/pc4_2022_v1.xlsx',
                             skiprows=7)
data_pc4_2022 =pd.read_excel('../data/CBS/PC4STATS/pc4_2022_v1.xlsx',
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


# +
def getgwb(year):
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
    return ([g,w,b])

gemeentendata ,  wijkgrensdata ,    buurtendata = getgwb(2020)    
# -

gemeentendata_c ,  wijkgrensdata_c ,    buurtendata_c = getgwb(2021)    

# +
#print(wijkgrensdata.crs.to_string() )
#gemeentendata.to_crs(epsg=3857)
# -

buurtendata.dtypes

buurtendata

#Okay: postcode is alleen soort van indicator
recspb =buurtendata[['BU_CODE','BU_NAAM']].groupby(['BU_CODE']).count().reset_index()
recspb[recspb['BU_NAAM']>1]

#examples from https://geopandas.org/en/stable/docs/user_guide/mapping.html
buurtendata.plot()

buurtendata.plot(column="BEV_DICHTH",legend=True, cmap='OrRd')

gemset = ['Houten']
gemset = ['Houten','Bunnik','Nieuwegein']
gemset2 = ['Houten','Bunnik','Nieuwegein', 
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','Utrechtse Heuvelrug',
          'De Bilt']
gemset2 = ['Houten','Bunnik','Nieuwegein', 'Vijfheerenlanden',
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','De Bilt','Utrechtse Heuvelrug',
           'Stichtse Vecht' ]
buurtdata_sel=buurtendata[buurtendata['GM_NAAM'].isin (gemset)]
gemdata_sel=gemeentendata[gemeentendata['GM_NAAM'].isin(gemset)]
wijkgrdata_sel=wijkgrensdata[wijkgrensdata['GM_NAAM'].isin(gemset)]

base=wijkgrdata_sel.boundary.plot(color='green');
#cx.add_basemap(base, crs=wijkgrdata_sel.crs.to_string(), source=cx.providers.Stamen.TonerLite)

#base=gemdata_sel.boundary.plot(color='green');
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_sel.plot(ax=base ,column="BEV_DICHTH",legend=True, cmap='OrRd',
                  legend_kwds={"label": "Bevolkingsdichtheid"})

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
#buurtdata_sels=buurtdata_sel[buurtdata_sel['STED'] !=5]
buurtdata_sels=buurtdata_sel[buurtdata_sel['BEV_DICHTH'] >1000]
buurtdata_sels.plot(ax=base ,column="STED",legend=True, cmap='RdYlBu',
                  legend_kwds={"label": "Stedelijkheid"})

buurtdata_selfield='BU_CODE'
adcol01= ['D000025','A045790_3','A045791_3','A045792_5']
def adddta(df1,df2,addmeas):
    df1s=df1[['geometry','BU_CODE','BU_NAAM','GM_NAAM','AANT_INW']]
    df2flds = df2[df2['Measure'].isin(addmeas)]    
    df2tab= df2flds.pivot_table(index='WijkenEnBuurten', columns='Measure',values='Value')
    seljoin = df1s.merge(df2tab,left_on=buurtdata_selfield, right_on='WijkenEnBuurten' ,how="left")    
    return(seljoin)
buurtdata_withfield01= adddta(buurtdata_sels,dat_85560,adcol01)

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield01.plot(ax=base,column=adcol01[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield01['SUPER_LOOP'] = buurtdata_withfield01[adcol01[1]] !=0
buurtdata_withfield01.plot(ax=base,column='SUPER_LOOP',legend=True, cmap='RdYlBu',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

buurtdata_withfield01.groupby(['GM_NAAM','SUPER_LOOP']).agg({'AANT_INW':'sum'})

adcol02= ['D000038', 'A045790_4', 'A045791_4','A045792_6' ]
buurtdata_withfield02= adddta(buurtdata_sel,dat_85560,adcol02)
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield02.plot(ax=base,column=adcol02[1],legend=True, cmap='OrRd',
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
buurtdata_withfield03= adddta(buurtdata_sel,dat_85560,adcol03)
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield03.plot(ax=base,column=adcol03[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal cafes < 1 km"}
           )          


# +
def getpc6hnryr(year):
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
    return(ngrp)
                       
pc6hnryr =getpc6hnryr(2020) 
pc6hnryr.dtypes
# -

#2023 werkt nog niet
for year in range(2018,2023):
    pc6hnryr =getpc6hnryr(year) 
    print(pc6hnryr.dtypes)



# +
#now start algoritm to make pc4 values from buurten
#first fetch matchigng table, special one for 2020
# -

pc6gwb2020 = pd.read_csv("../data/CBS/PC6HNR/pc6-gwb2020.csv", encoding = "ISO-8859-1", sep=";")  
pc6gwb2020['PC4'] = pc6gwb2020['PC6'].str[0:4]
pc6gwb2020['BU_CODE']  = pc6gwb2020['Buurt2020'].apply( lambda x: "BU%08i" % x)

pc6gwb2020.dtypes

pc4perbuurt = pc6gwb2020[['Buurt2020','PC4','PC6']].groupby(['Buurt2020','PC4']).count().reset_index()
pc6perpc4 = pc6gwb2020[['Buurt2020','PC4','PC6']].groupby(['PC4']).count().reset_index()
pc4perbuurt
nbuurtperpc6 =pc6gwb2020[['Buurt2020','PC6']].groupby(['PC6']).count().reset_index().rename(
    columns={'Buurt2020':'nbuurtpc6'} )
#somminge PC6-en liggen ook nog eens in meerdere buurten
numPC6perbuurtverd = nbuurtperpc6 .groupby(['nbuurtpc6']).count().reset_index()
print(numPC6perbuurtverd)


#maar dat hoeft niet erg te zijn als IN DAT GEVAL de PC4 codes van die PC6 gelijk zijn
PC6metmeerderbuurten = nbuurtperpc6 [nbuurtperpc6['nbuurtpc6']>1]
buurteninPC6mm= pc6gwb2020 [ pc6gwb2020['PC6'].isin( PC6metmeerderbuurten['PC6'])]
#nu meer dan 2 x het aantal records als niet-1
buurteninPC6mm
pc4buurtcombiPC6mm = buurteninPC6mm[['Buurt2020','PC4','PC6']].groupby(['Buurt2020','PC4']
                                                ).count().reset_index()
pc4buurtcombiPC6mmaant = pc4buurtcombiPC6mm.groupby(['Buurt2020']).count().reset_index()
pc4buurtcombiPC6mmverd = pc4buurtcombiPC6mmaant.groupby(['PC4']).count().reset_index()
pc4buurtcombiPC6mmverd
#dat is dus niet het geval, dus verdeling is nodig

pc6gwb2020c1 =pc6gwb2020.merge(nbuurtperpc6,how='left')
pc6gwb2020c1['pc6buwgt'] = 1.0/pc6gwb2020c1['nbuurtpc6']
buurtvelover = pc6gwb2020c1[['Buurt2020','pc6buwgt']].groupby('Buurt2020').sum(
    ).reset_index().rename (  columns={'pc6buwgt':'buurtwgsum'} )
pc6gwb2020c1l  = pc6gwb2020c1.merge(buurtvelover,how='left')
buurtvelover['BU_CODE']  = buurtvelover['Buurt2020'].apply( lambda x: "BU%08i" % x)
buurtveloverstats =buurtvelover.groupby('buurtwgsum').count().reset_index()
buurtveloverstats

pc6gwb2020c1l [ pc6gwb2020c1l['BU_CODE' ] .isna()]
pc6gwb2020c1l 

print(buurtendata.columns)
bufields_sum=[ 'AANT_INW',
       'AANT_MAN', 'AANT_VROUW', 'P_00_14_JR', 'P_15_24_JR', 'P_25_44_JR',
       'P_45_64_JR', 'P_65_EO_JR', 
#niet in PC4        'P_ONGEHUWD', 'P_GEHUWD', 'P_GESCHEID', 'P_VERWEDUW', 
       'AANTAL_HH', 'P_EENP_HH', 'P_HH_Z_K', 'P_HH_M_K',
#niet in PC4       'P_NL_ALL', 'P_EUR_ALL', 'P_NEU_ALL', 
        'P_GEBNL_NL', 'P_GEBNL_EU', 'P_GEBNL_NE', 'P_GEBBL_EU', 'P_GEBBL_NE', 'OPP_TOT',
       'OPP_LAND', 'OPP_WATER']
bufields_rel=['GEM_HH_GR', 'DEK_PERC', 'OAD', 'STED', 'BEV_DICHTH']

pc4fields_sum=[ 'Totaal', 'Man', 'Vrouw', 'tot 15 jaar', '15 tot 25 jaar',
       '25 tot 45 jaar', '45 tot 65 jaar', '65 jaar en ouder',
        'Totaal.1','Eenpersoons','Meerpersoons \nzonder kinderen',
        'Eenouder','Tweeouder',
'Geboren in Nederland met een Nederlandse herkomst',
'Geboren in Nederland met een Europese herkomst (excl. Nederland)',
'Geboren in Nederland met herkomst buiten Europa',
'Geboren buiten Nederland met een Europese herkomst (excl. Nederland)',
'Geboren buiten Nederland met een herkomst buiten Europa'
                ]
pc4fields_rel=['Huishoudgrootte']
print(data_pc4_2020.columns)
woondata="\
Totaal.2                                                                  int64\
voor 1945                                                                 int64\
1945 tot 1965                                                             int64\
1965 tot 1975                                                             int64\
1975 tot 1985                                                             int64\
1985 tot 1995                                                             int64\
1995 tot 2005                                                             int64\
2005 tot 2015                                                             int64\
2015 en later                                                             int64\
Meergezins                                                                int64\
Koopwoning                                                                int64\
Huurwoning                                                                int64\
Huurcoporatie                                                             int64\
Niet bewoond                                                              int64\
WOZ-waarde\nwoning                                                        int64\
Personen met WW, Bijstand en/of AO uitkering\nBeneden AOW-leeftijd        int64"


# +
def checkkeyuniek(df,keys):
    recspb =df[keys].groupby(keys).count().reset_index()
    luni = len(recspb.index)
    lin  = len(df.index)
    if lin!=luni:
        print('duplicate entries in input',lin,luni)
        stop()
    
checkkeyuniek(buurtendata,['BU_CODE'])


# +
#dus eerste verdelen naar PC6, controleren dat totalen behouden zijn
#todist mag maar 1 record per buurt hebben ! check dit eerst
def distrpc6v2020(bverd,todist,bucol,cols):
    checkkeyuniek(todist,[bucol])
    exand=bverd[['PC6','BU_CODE','pc6buwgt','buurtwgsum']].merge(todist[['BU_CODE']+cols],
                        left_on ='BU_CODE', right_on = bucol ,how='outer')
    #print(exand)
    print(len(exand.index))
    nonm = exand[exand[cols[0]].isna() ]
    print(len(nonm.index))
    if (len(nonm.index) !=0):
        print (nonm)
    nonm = exand[exand['PC6'].isna() ]
    print(len(nonm.index))
    if (len(nonm.index) !=0):
        print (nonm)        
    for t in cols:
        exand[t] = exand[t] * exand['pc6buwgt'] / exand['buurtwgsum']
    perpc6=exand.groupby('PC6').sum().reset_index()
    isums = todist[cols].sum()
    osums = perpc6[cols].sum()
    odiff = osums - isums
    print(isums)
    print(osums)
    print(odiff)
    
tstbu='BU19923200'
tstbu='BU03630403'
buurtendatasel=buurtendata[buurtendata['BU_CODE']==tstbu]
pc6gwb2020c1ls=pc6gwb2020c1l[pc6gwb2020c1l['BU_CODE']== tstbu]

distrpc6v2020(pc6gwb2020c1l,buurtendata,'BU_CODE',bufields_sum[1:3] )   
# -
pc6hnryr =getpc6hnryr(2020) 


nbuurtperpc6 =pc6hnryr[['Buurt','PC6']].groupby(['PC6']).count().reset_index().rename(
    columns={'Buurt':'nbuurtpc6'} )
#somminge PC6-en liggen ook nog eens in meerdere buurten
numPC6perbuurtverd = nbuurtperpc6 .groupby(['nbuurtpc6']).count().reset_index()
numPC6perbuurtverd

nhnrperpc6 =pc6hnryr[['Huisnummer','PC6']].groupby(['PC6']).sum().reset_index().rename(
    columns={'Huisnummer':'nhuisnrpc6'} )
#somminge PC6-en liggen ook nog eens in meerdere buurten
nhnrperpc6v = nhnrperpc6 .groupby(['nhuisnrpc6']).count().reset_index()
nhnrperpc6v

nhnrperbuurt =pc6hnryr[['Huisnummer','Buurt']].groupby(['Buurt']).sum().reset_index().rename(
    columns={'Huisnummer':'nhuisnrbuurt'} )
#somminge PC6-en liggen ook nog eens in meerdere buurten
nhnrperbuurtv  = nhnrperbuurt  .groupby(['nhuisnrbuurt']).count().reset_index()
nhnrperbuurtv 

pc6hnryrmstats= pc6hnryr.merge(nhnrperbuurt,how='left')
pc6hnryrmstats


# +
#dus eerste verdelen naar PC6, controleren dat totalen behouden zijn
#todist mag maar 1 record per buurt hebben ! check dit eerst
def distrpc6(bverd,todist,bucol,sumcols,avgcols):
    checkkeyuniek(todist,[bucol])
    exand=bverd[['PC6','BU_CODE','Huisnummer','nhuisnrbuurt']].merge(todist[['BU_CODE']+
                                        sumcols+avgcols],
                        left_on ='BU_CODE', right_on = bucol ,how='outer')
    #print(exand)
    print(len(exand.index))
    nonm = exand[exand[sumcols[0]].isna() ]
    print(len(nonm.index))
    if (len(nonm.index) !=0):
        print (nonm[['PC6','BU_CODE']])
    nonm = exand[exand['PC6'].isna() & ~ exand[sumcols[0]].isna() ]
    print(len(nonm.index))
    print(nonm[sumcols+avgcols].sum())
    if (len(nonm.index) !=0):
        print (nonm[['BU_CODE']+  sumcols+avgcols])        
    normdf= exand[['PC6','BU_CODE']]
    for t in sumcols:
        exand[t] = exand[t] * exand['Huisnummer'] / exand['nhuisnrbuurt']
    for t in avgcols:
        exand[t] = exand[t] * exand['Huisnummer'] 
        normdf[t] =  exand['Huisnummer'] 
    perpc6=exand.groupby('PC6').sum().reset_index()
    normpc6=normdf.groupby('PC6').sum().reset_index()
    for t in avgcols:
        perpc6[t] =  perpc6[t] / normdf[t]
    isums = todist[sumcols].sum()
    osums = perpc6[sumcols].sum()
    odiff = osums - isums
    print(isums)
    print(osums)
    print(odiff)
    return (perpc6)
    
tstbu='BU19923200'
tstbu='BU03630403'
buurtendatasel=buurtendata[buurtendata['BU_CODE']==tstbu]
pc6gwb2020c1ls=pc6gwb2020c1l[pc6gwb2020c1l['BU_CODE']== tstbu]

match6bu2020 = distrpc6(pc6hnryrmstats,buurtendata,'BU_CODE',bufields_sum[1:3],bufields_rel[1:3] )
# -

data_pc4_2020.replace(to_replace=-99997,value=pd.NA,inplace=True)  
data_pc4_2020.sort_values(by=['tot 15 jaar'])

print(data_pc4_2020.sum())


# +
def cmppc4(cnvdat,dirdat,sumcolg,avgcolg,sumcold,avgcold):    
    cnvdat['PC4']= cnvdat['PC6'].str[0:4].astype('int64')
    normdf= cnvdat[['PC4','Huisnummer']]
    for t in avgcolg:
        cnvdat[t] = cnvdat[t] * cnvdat['Huisnummer'] 
    cnv4dat = cnvdat.groupby(['PC4']).sum().reset_index()
    for t in avgcolg:
        cnv4dat[t] = cnv4dat[t] / cnvdat['Huisnummer'] 
    cmp4dat = cnv4dat.merge(dirdat,left_on='PC4',right_on="Postcode-4")
    cmpres= cmp4dat[['PC4','Huisnummer']]
    for tc,td in zip(sumcolg,sumcold):
        print (tc,td)
        cmpres[tc] = cmp4dat[tc] - cmp4dat[td]

    return(cmpres)
    
cmp4bu2020 = cmppc4(match6bu2020,data_pc4_2020,
                     bufields_sum[1:3],bufields_rel[1:3],
                     pc4fields_sum[1:3],pc4fields_rel[1:3] )
print(cmp4bu2020.sum())
cmp4bu2020
# -

cmp4bu2020.sort_values(by=['AANT_MAN'])


# +
def matchyr(year):
    gemeentendata_l ,  wijkgrensdata_l,    buurtendata_l = getgwb(year)  
    pc6hnryr_l =getpc6hnryr(year) 
    nhnrperbuurt_l =pc6hnryr_l[['Huisnummer','Buurt']].groupby(['Buurt']).sum().reset_index().rename(
        columns={'Huisnummer':'nhuisnrbuurt'} )
    #somminge PC6-en liggen ook nog eens in meerdere buurten
    pc6hnryrmstats_l= pc6hnryr_l.merge(nhnrperbuurt_l,how='left')
    distrpc6(pc6hnryrmstats_l,buurtendata_l,'BU_CODE',bufields_sum[1:3] )
    
matchyr(2021)    
# -

matchyr(2022)    


