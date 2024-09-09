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
#visualisaties van data
# -

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#het inlezen van odinverplgr loopt in deze versie via ODINcatVNuse
import ODINcatVNuse

useKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
xlatKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <20].copy()
maxcuse= np.max(useKAfstV[useKAfstV ["MaxAfst"] !=0] ['KAfstCluCode'])
xlatKAfstV  = xlatKAfstVa [(xlatKAfstVa['KAfstCluCode']<=maxcuse ) |
                           (xlatKAfstVa['KAfstCluCode']==np.max(useKAfstV[ 'KAfstCluCode']) )].copy()
#print(xlatKAfstV)   
print(useKAfstV)   

print(useKAfstV) 
maskKAfstV= list(useKAfstV['KAfstCluCode'])
maskKAfstV

odinverplklinfo = ODINcatVNuse.odinverplklinfo_o[np.isin(ODINcatVNuse.odinverplklinfo_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
odinverplgr =ODINcatVNuse.odinverplgr_o[np.isin(ODINcatVNuse.odinverplgr_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
odinverplflgs =ODINcatVNuse.odinverplflgs_o[np.isin(ODINcatVNuse.odinverplflgs_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)

import ODiN2readpkl

dbk_2022 = ODiN2readpkl.dbk_allyr
dbk_2022_cols = dbk_2022 [~ dbk_2022.Variabele_naam_ODiN_2022.isna()]
dbk_2022_cols [ dbk_2022_cols.Niveau.isna()]

specvaltab = ODiN2readpkl.mkspecvaltab(dbk_2022)
specvaltab

odindiffflginfo= ODINcatVNuse.convert_diffgrpsidat(odinverplflgs,
                ODINcatVNuse.fitgrpse,[],ODINcatVNuse.kflgsflds, [],"_c",ODINcatVNuse.landcod,False)

odindiffflginfo


def mkmotsumdiff1(diffin,myspecvals):
    motafstgrp =  diffin.groupby(['MotiefV','KAfstCluCode'])[['FactorV_c','FactorKm_c','FactorActiveV_c']].agg('sum').reset_index()
    totgrp = motafstgrp.groupby(['MotiefV'])[['FactorV_c']].agg('sum').reset_index(). rename (columns={'FactorV_c':'MotiefSum'})
    motafstgrp= motafstgrp.merge(totgrp)
    explhere1 = myspecvals [myspecvals['Variabele_naam'] == 'MotiefV'].copy()
    explhere1['Code'] = pd.to_numeric(explhere1['Code'],errors='coerce')
    motafstgrp = (motafstgrp.merge(explhere1,left_on='MotiefV', right_on='Code', how='left').rename(
               columns={'Code_label':'MotiefV_label'}) )
    motafstgrp['MotiefS'] = (motafstgrp['MotiefV'].astype(int) .astype(str) ) + " "+ motafstgrp['MotiefV_label']
    motafstgrp['FractAct'] = motafstgrp['FactorActiveV_c']/ motafstgrp['FactorV_c'] 
    motafstgrp['GemAfst'] = motafstgrp['FactorKm_c']/ motafstgrp['FactorV_c'] 
    motafstgrp['FractMot'] = motafstgrp['FactorV_c']/ motafstgrp['MotiefSum'] 
    
    return motafstgrp
motafstgrp_glb = mkmotsumdiff1(odindiffflginfo,specvaltab)
#motafstgrp_glb

def mkmotsum1(diffin,myspecvals,myKAfstV):
    motafstgrp =  diffin.groupby(['MotiefV','KAfstCluCode'])[['FactorV','FactorKm','FactorActiveV']].agg('sum').reset_index()
    totgrp = motafstgrp[motafstgrp['KAfstCluCode']==15][['MotiefV','FactorV']]. rename (columns={'FactorV':'MotiefSum'})
    motafstgrp= motafstgrp.merge(totgrp).merge(myKAfstV)
    explhere1 = myspecvals [myspecvals['Variabele_naam'] == 'MotiefV'].copy()
    explhere1['Code'] = pd.to_numeric(explhere1['Code'],errors='coerce')
    motafstgrp = (motafstgrp.merge(explhere1,left_on='MotiefV', right_on='Code', how='left').rename(
               columns={'Code_label':'MotiefV_label'}) )
    motafstgrp['MotiefS'] = (motafstgrp['MotiefV'].astype(int) .astype(str) ) + " "+ motafstgrp['MotiefV_label']
    motafstgrp['FractAct'] = motafstgrp['FactorActiveV']/ motafstgrp['FactorV'] 
    motafstgrp['GemAfst'] = np.where(motafstgrp['MaxAfst'] ==0,100 , motafstgrp['MaxAfst'] )
    motafstgrp['FractMot'] = motafstgrp['FactorV']/ motafstgrp['MotiefSum'] 
    
    return motafstgrp
motafstgrpc_glb = mkmotsum1(odinverplflgs,specvaltab,useKAfstV)
#motafstgrp_glb

def motplt(df0,tit):
    df = df0[df0['MotiefSum']>5e9]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df,x='GemAfst',y="FactorV",hue="MotiefS",ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Afstand (km)')
    ax.set_ylabel('Aantal ritten')
    ax.set_title(tit)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig
motplt(motafstgrpc_glb,'Totalen over jaren, cumulatief over afstand')


def motpltrel(df0,tit):
    df = df0[df0['MotiefSum']>5e9]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df,x='GemAfst',y="FractMot",hue="MotiefS",ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Afstand (km)')
    ax.set_ylabel('Fractie van reizen voor motief')
    ax.set_title(tit)
    ax.axhline(0)
    ax.axhline(0.5)
    ax.axhline(1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig
motpltrel(motafstgrpc_glb,'Deel reizen per motief, cumulatief over afstand')


def motactfplt(df0,tit):
    df = df0[df0['MotiefSum']>5e9]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df,x='GemAfst',y="FractAct",hue="MotiefS",ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('Afstand (km)')
    ax.set_ylabel('Fractie active modes')
    ax.set_title(tit)
    ax.axhline(0)
    ax.axhline(0.5)
    ax.axhline(1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig
motactfplt(motafstgrpc_glb,'Deel actief, alle reizen binnen afstand')

motactfplt(motafstgrp_glb,'Deel actief, EXTRA reizen op die afstand')


