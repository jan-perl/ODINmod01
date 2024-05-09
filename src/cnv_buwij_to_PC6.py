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

#import ODiN2pd
import ODiN2readpkl

os.system("pip install geopandas")

os.system("pip install contextily")

import geopandas
import contextily as cx
plt.rcParams['figure.figsize'] = [10, 6]

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
data_pc4_2020 = ODiN2readpkl.getpc4stats(2020)
data_pc4_2020

data_pc4_2022 = ODiN2readpkl.getpc4stats(2022)
data_pc4_2022

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

gemeentendata ,  wijkgrensdata ,    buurtendata = ODiN2readpkl.getgwb(2020)    

# +
#gemeentendata_c ,  wijkgrensdata_c ,    buurtendata_c = getgwb(2021)    

# +
#print(wijkgrensdata.crs.to_string() )
#gemeentendata.to_crs(epsg=3857)
# -

buurtendata.dtypes

print(gemeentendata.dtypes)



# +
#now start algoritm to make pc4 values from buurten
#first fetch matchigng table, special one for 2020
# -

pc6gwb2020 = pd.read_csv("../data/CBS/PC6HNR/pc6-gwb2020.csv", encoding = "ISO-8859-1", sep=";")  
pc6gwb2020['PC4'] = pc6gwb2020['PC6'].str[0:4].astype('int64')
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
pc6hnryr =ODiN2readpkl.getpc6hnryr(2020) 


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

voorb_pc4=3897
voorb_pc4_BUs = pc6hnryrmstats[pc6hnryrmstats['PC6'].str[0:4].astype('int64') ==voorb_pc4]['BU_CODE'].unique()
print(voorb_pc4_BUs)
pc6hnryrmstats[pc6hnryrmstats['PC6'].str[0:4].astype('int64') ==voorb_pc4]

# +
#om nrhuisnrbuurt te controleren: pc6hnryrmstats[pc6hnryrmstats['BU_CODE'].isin(voorb_pc4_BUs)].groupby(['BU_CODE']).sum()
# -

pc6hnryrmstats[pc6hnryrmstats['BU_CODE'].isin(voorb_pc4_BUs)].sort_values(by=['PC6'])

lastsumfield=5
lastrelfield=2
fchk=bufields_sum[0:lastsumfield]+bufields_rel[0:lastrelfield]
buurtendata[buurtendata['BU_CODE'].isin(voorb_pc4_BUs)][['BU_CODE']+fchk]


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
        print('Missende matches ')
        print (nonm[['PC6','BU_CODE']])
    nonm = exand[exand['PC6'].isna() & ~ exand[sumcols[0]].isna() ]
    print(len(nonm.index))
    print(nonm[sumcols+avgcols].sum())
    if (len(nonm.index) !=0):
        print ('Missende matches ')
        print ( nonm[['BU_CODE']+  sumcols+avgcols])        
    normdf= exand[['PC6','BU_CODE']]
    for t in sumcols:
        exand[t] = exand[t] * exand['Huisnummer'] *1.0/ exand['nhuisnrbuurt']
    for t in avgcols:
        exand[t] = exand[t] * exand['Huisnummer'] 
        exand['avgnorm'] =  exand['Huisnummer'] 
    perpc6=exand.groupby('PC6').sum().reset_index()
    for t in avgcols:
        perpc6[t] =  perpc6[t] / perpc6['avgnorm']
    isums = todist[sumcols].sum()
    osums = perpc6[sumcols].sum()
    odiff = osums - isums
    print("Totaal controles, alleen voor som velden")
    print(isums)
    print(osums)
    print(odiff)
    return (perpc6)
    
tstbu='BU19923200'
tstbu='BU03630403'
buurtendatasel=buurtendata[buurtendata['BU_CODE']==tstbu]
pc6gwb2020c1ls=pc6gwb2020c1l[pc6gwb2020c1l['BU_CODE']== tstbu]

match6bu2020 = distrpc6(pc6hnryrmstats,buurtendata,'BU_CODE',bufields_sum[0:lastsumfield],bufields_rel[0:lastrelfield] )
# -

match6bu2020

data_pc4_2020.replace(to_replace=-99997,value=pd.NA,inplace=True)  
data_pc4_2020.sort_values(by=['tot 15 jaar'])

print(data_pc4_2020.sum())

match6bu2020[match6bu2020['PC6'].str[0:4].astype('int64') ==voorb_pc4]


# +
def cnvpc4(cnvdat,sumcolg,avgcolg):    
    cnvdat['PC4']= cnvdat['PC6'].str[0:4].astype('int64')
    normdf= cnvdat[['PC4','Huisnummer']]
    for t in avgcolg:
        cnvdat[t] = cnvdat[t] * cnvdat['Huisnummer'] 
    cnv4dat = cnvdat.groupby(['PC4']).sum().reset_index()
    for t in avgcolg:
        cnv4dat[t] = cnv4dat[t] / cnv4dat['Huisnummer'] 
    return(cnv4dat)

def cmppc4(cnvdat,dirdat,sumcolg,avgcolg,sumcold,avgcold):  
    cnv4dat = cnvpc4(cnvdat,sumcolg,avgcolg)
    cmp4dat = cnv4dat.merge(dirdat,left_on='PC4',right_on="Postcode-4")
    cmpres= cmp4dat[['PC4','Huisnummer']+sumcolg+sumcold+avgcolg+avgcold]
    for tc,td in zip(sumcolg,sumcold):
        print (tc,td)
        cmpres[tc+"diff"] = cmp4dat[tc] - cmp4dat[td]
    for tc,td in zip(avgcolg,avgcold):
        print (tc,td)
        cmpres[tc+"diff"] = cmp4dat[tc] - cmp4dat[td]        

    return(cmpres)
    
cmp4bu2020 = cmppc4(match6bu2020,data_pc4_2020,
                     bufields_sum[0:lastsumfield],bufields_rel[0:lastrelfield],
                     pc4fields_sum[0:lastsumfield],pc4fields_rel[0:lastrelfield] )
print(cmp4bu2020.sum())
cmp4bu2020
# -

cmp4bu2020[cmp4bu2020['PC4'].isin([voorb_pc4,voorb_pc4-1])]

cmp4bu2020.sort_values(by=['AANT_MANdiff'],ascending=False)


# +
def matchyr(year):
    gemeentendata_l ,  wijkgrensdata_l,    buurtendata_l = ODiN2readpkl.getgwb(year)  
    pc6hnryr_l =ODiN2readpkl.getpc6hnryr(year) 
    nhnrperbuurt_l =pc6hnryr_l[['Huisnummer','Buurt']].groupby(['Buurt']).sum().reset_index().rename(
        columns={'Huisnummer':'nhuisnrbuurt'} )
    #somminge PC6-en liggen ook nog eens in meerdere buurten
    pc6hnryrmstats_l= pc6hnryr_l.merge(nhnrperbuurt_l,how='left')
    distrpc6(pc6hnryrmstats_l,buurtendata_l,'BU_CODE',bufields_sum[0:lastsumfield],bufields_rel[0:3] )
    
matchyr(2021)    
# -

matchyr(2022)    

print("Finished")


