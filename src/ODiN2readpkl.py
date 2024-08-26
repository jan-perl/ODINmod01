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
#leest geconverteerde pickle data tabellen
#goede basis voor verdere berekeningen
# -

import pandas as pd
import numpy as np
import os as os

# +
#TODO hernoem wat kolommen

# +
#TODO parse ook data labels

# +
#nu postcode match hulptabel
# -

allodinyr =pd.read_pickle("../intermediate/allodinyr.pkl")

dbk_allyr= pd.read_pickle("../intermediate/dbk_allyr.pkl")

fietswijk1pc4=pd.read_pickle("../intermediate/fietswijk1pc4.pkl")

#note: to clean: copied in ODiN2pd
excols= ['Wogem', 'AutoHhl', 'MRDH', 'Utr', 'FqLopen', 'FqMotor', 'WrkVervw', 'WrkVerg', 'VergVast', 
         'VergKm', 'VergBrSt', 'VergOV', 'VergAans', 'VergVoer', 'VergBudg', 'VergPark', 'VergStal', 'VergAnd', 
         'BerWrk', 'RdWrkA', 'RdWrkB', 'BerOnd', 'RdOndA', 'RdOndB', 'BerSup', 'RdSupA', 'RdSupB',
         'BerZiek', 'RdZiekA', 'RdZiekB', 'BerArts', 'RdArtsA', 'RdArtsB', 'BerStat', 'RdStatA', 'RdStatB', 
         'BerHalte', 'RdHalteA', 'RdHalteB', 'BerFam', 'RdFamA', 'RdFamB', 'BerSport', 'RdSportA', 'RdSportB',
          'VertMRDH', 'VertUtr', 'AankMRDH', 'AankUtr' ]

# +
largranval = -9999999999
ODINmissint = -99997 
def mkspecvaltab(indbk): 
    lastlbl=''
    nrec=int(len(indbk))
    c0=[]
    c1=[]
    c2=[]
    for irec in range(0,nrec):
        nxtlbl=indbk.iloc[[irec]]
#        print (nxtlbl)
        if nxtlbl['Variabele_naam_ODiN_2022'].isna().item():
            if ".." in str(nxtlbl['Code_ODiN_2022'].item()):
                vrng = str(nxtlbl['Code_ODiN_2022'].item()).split("..")
                vrng= [int(vrng[0]),int(vrng[1])+1]
                if (vrng[1] - vrng[0]) >15:
                    print ('Setting (unchecked) large range',vrng,lastlbl,
                           nxtlbl['Code_label_ODiN_2022'].item())
                    c0.append(lastlbl)
                    c1.append(largranval)
                    c2.append(nxtlbl['Code_label_ODiN_2022'].item() )
                else:
                    for num in range(vrng[0],vrng[1]):
                        c0.append(lastlbl)
                        c1.append(num)
                        c2.append(num)
            else:
                c0.append(lastlbl)
                c1.append(nxtlbl['Code_ODiN_2022'].item())
                c2.append(nxtlbl['Code_label_ODiN_2022'].item() )
        else:
            lastlbl=nxtlbl['Variabele_naam_ODiN_2022'].item()
    outcol_names =  ['Variabele_naam', 'Code', 'Code_label'] 
    outdf=pd.DataFrame(list(zip(c0,c1,c2)),columns=outcol_names)
    return(outdf)

specvaltab = mkspecvaltab(dbk_allyr)
specvaltab


# -
def getgwb(year):
    stryear=str(year)    
    g=pd.read_pickle("../intermediate/CBS/gwb_gem_"+stryear+".pkl")    
    w=pd.read_pickle("../intermediate/CBS/gwb_wijk_"+stryear+".pkl") 
    b=pd.read_pickle("../intermediate/CBS/gwb_buurt_"+stryear+".pkl") 
    return ([g,w,b])
#test code
#gemeentendata ,  wijkgrensdata ,    buurtendata = getgwb(2020)    


def getpc4stats(year):
    stryear=str(year)    
    data_pc4=pd.read_pickle("../intermediate/CBS/pc4stats_"+stryear+".pkl")    
    return (data_pc4)
#test code
#data_pc4_2020 = getpc4stats(2020)


def getpc6hnryr(year):
    stryear=str(year)    
    ngrp=pd.read_pickle("../intermediate/CBS/pc6hnryr_"+stryear+".pkl") 
    return(ngrp)
pc6hnryr =getpc6hnryr(2020) 
pc6hnryr.dtypes


