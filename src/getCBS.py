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

if(1==1):
    try:
        os.mkdir("../data/CBS")
    except:
        print(" replacing existing ");

fetchweb=True

#Kerncijfers per postcode
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/gegevens-per-postcode
pc4lnks = ["https://download.cbs.nl/postcode/2024-cbs_pc4_2023_v1.zip",
           "https://download.cbs.nl/postcode/2024-cbs_pc4_2022_v2.zip",
           "https://download.cbs.nl/postcode/2024-cbs_pc4_2021_vol.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc4_2020_vol.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc4_2019_vol.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2018-v3.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2017-v3.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2016-v2.zip",
           "https://download.cbs.nl/postcode/CBS-PC4-2015-v2.zip"
          ] 
print(pc4lnks[1:3])

#Kerncijfers per postcode
#https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/gegevens-per-postcode
pc6lnks = ["https://download.cbs.nl/postcode/2024-cbs_pc6_2023_v1.zip",
           "https://download.cbs.nl/postcode/2024-cbs_pc6_2022_v2.zip",
           "https://download.cbs.nl/postcode/2024-cbs_pc6_2021_vol.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc6_2020_vol.zip",
           "https://download.cbs.nl/postcode/2023-cbs_pc6_2019_vol.zip",
           "https://download.cbs.nl/postcode/CBS-PC6-2018-v3.zip",
           "https://download.cbs.nl/postcode/CBS-PC6-2017-v3.zip",
           "https://download.cbs.nl/postcode/CBS-PC6-2016-v2.zip",
           "https://download.cbs.nl/postcode/CBS-PC6-2015-v2.zip"
          ] 
print(pc6lnks[1:3])


def getcbspc4(link,uzdir):
    of=re.sub("^.*/","",link)
    try:
        os.remove ("../data/CBS/"+of)
    except:
        print(" first time download "+uzdir);        
    str1= "wget -q -O ../data/CBS/"+of+" "+link
    os.system(str1)
    if uzdir=='':
        str2= "bash -c 'cd ../data/CBS ; unzip -u "+of + "'"
    else:
        try:
            os.mkdir ("../data/CBS/"+uzdir)
        except:
            print(" replacing existing "+uzdir);
        str2= "bash -c 'cd ../data/CBS/"+uzdir+" ; unzip -u ../"+of + "'"
    os.system(str2)
if fetchweb :
    for link in pc4lnks:
        getcbspc4(link,'PC4STATS')

if fetchweb | True:
    for link in pc6lnks:
        getcbspc4(link,'PC6STATS')


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


