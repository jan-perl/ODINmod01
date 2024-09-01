# -*- coding: utf-8 -*-
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
# analyse van pkl in intermediates van ODIN van/naar categorieen
#
# -

# todo
# actieve modes per afstandsklasse en type
# gemiddelde afstand per afstandsklasse en type
# active mode kms en passive mode kms per PC naar & van (let op: dubbeltelling)
# referentie reiskms actieve modes
# vergelijking co2 uitstoot active modes met kentallen


# +
#todo
#OA klasse ook via convolutie -> voor bij grafieken
#vannaar per postcode combinaite maken voor geselecteerde postcodes
#  voor visualisatie, uit convolutie. Exporteer alleen !=0, bijvoorbeeld PCs in Utrecht

# +
#splits code op; DIT DEEL is:

#- analyse top PC4 per Motief voor 2 ritten op dag en niet woon kant
#- t.o.v. gebieds oppervlak
#- bij top PC4 maak annotatie naar 1 of meerdere PC6 mogelijk
# maak ook afstandsklassen tabel (met index per indeling naam) en schrijf naar pkl

# -

# scenarios
# #+ 10 % wonen en +10 werken (8% en 10 % oppervlak resp)
# A) beide op plaatsen waar het al is
# b) tegengesteld
# C) werk op huidige plek , wonen alleen waar nu lage dichtheid is (+20 % daar)


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import re
import time
import io   

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"

from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
from sklearn import linear_model
import seaborn

stryear='2020'
cbspc4data =pd.read_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl")
cbspc4data= cbspc4data.sort_values(by=['postcode4']).reset_index()

cbspc4data['oppervlak'] = cbspc4data.area

cbspc4data.dtypes

#providers = cx.providers.flatten()
#providers
prov0=cx.providers.nlmaps.grijs.copy()
print( cbspc4data.crs)
print (prov0)
plot_crs=3857
#plot_crs=28992
if 1==1:
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['min_zoom']= 0
    prov0['max_zoom'] =12
    print (prov0)

pland= cbspc4data.to_crs(epsg=plot_crs).plot()
cx.add_basemap(pland, source= prov0)

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot()
cx.add_basemap(phtn, source= prov0)

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']==3995)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot()
cx.add_basemap(phtn, source= prov0)

pc4tifname=calcgdir+'/cbs2020pc4-NL.tif'
pc4excols= ['aantal_inwoners','aantal_mannen', 'aantal_vrouwen']
pc4inwgrid= rasterio.open(pc4tifname)


def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
pc4inwgcache = getcachedgrids(pc4inwgrid)

# nu nog MXI overzetten naar PC4 ter referentie




# +
#useKAfstV=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
#xlatKAfstV=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")

# +
odinverplgr_o=pd.read_pickle("../intermediate/ODINcatVN01db.pkl")

def deffactorv(rv):
    onlyok=False
    if onlyok:
        rv['FactorV'] = np.where ((rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ,
               np.nan,rv['FactorVGen'] + 0* rv['FactorVSpec'] )
    else:
        rv['FactorV'] = rv['FactorVGen'] + rv['FactorVSpec'] 
        
deffactorv(odinverplgr_o)
# -

fitgrps=['MotiefV','isnaarhuis']

odinverplflgs_o=pd.read_pickle("../intermediate/ODINcatVN03db.pkl")
kflgsflds=['FactorV',"FactorKm","FactorAutoKm","FactorActiveKm"]

odinverplklinfo_o=pd.read_pickle("../intermediate/ODINcatVN02db.pkl")
landcod=np.max(odinverplklinfo_o['KAfstCluCode'])
landcod

oriKAfstVcods= np.array(range(landcod))+1
maxcuse= 3
maskKAfstV  = oriKAfstVcods [(oriKAfstVcods<=maxcuse ) |
                             (oriKAfstVcods==landcod )].copy()
maskKAfstV

odinverplklinfo = odinverplklinfo_o[np.isin(odinverplklinfo_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
odinverplgr =odinverplgr_o[np.isin(odinverplgr_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
odinverplflgs =odinverplflgs_o[np.isin(odinverplflgs_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)

odinverplgr[['KAfstCluCode','GeoInd']].groupby('KAfstCluCode').agg('count')
#odinverplklinfo[['KAfstCluCode','FactorV']].groupby('KAfstCluCode').agg('count')

# +
nfogrps=['KHvm','AankUur','VertUur','Jaar']
# definition in ODINcatVN: allodinyr['FactorKm']= allodinyr['FactorV'] * allodinyr['AfstS'] *10
infoflds=['FactorV','FactorKm']

fitgrpse=fitgrps+['GrpExpl']
kinfoflds=["GrpTyp", "GrpVal","GrpV_label"]
odinverplklinfo
# -

odinverplgr[['FactorVGen','FactorVSpec']].sum()


def mkverplsum1(indf,landcod):    
    totdf=indf[indf['KAfstCluCode']==landcod]    
    rv=totdf.groupby(['KAfstCluCode',"GrpTyp"])[infoflds].agg('sum')
    return rv
mkverplsum1(odinverplklinfo,landcod)


# +
def mkverplsum1metlab(indf,kf,landcod):
    totdf=indf[indf['KAfstCluCode']==landcod]    
    rv=totdf.groupby(['KAfstCluCode']+kf)[infoflds].agg('sum')
    return rv
t2=(mkverplsum1metlab(odinverplklinfo,kinfoflds,landcod)/1e9/5).reset_index()

def convert_duurzaam_slice(t3):
    t2= t3[t3['GrpTyp']=='KHvm'].copy(deep=False)
    TTW_kgco2pkm=.149    
    t2['CO2GT']=t2['FactorKm'] * np.where(t2['GrpVal']==1,TTW_kgco2pkm,0)
    t2['KmActive']=t2['FactorKm'] * np.where(np.isin(t2['GrpVal'],[5,6]),1,0)
    return t2
#aantallen in miljarden per jaar, laatste kolom in kms
#t2
t2=convert_duurzaam_slice(t2)
t2['GemAfst']=t2['FactorKm']/ t2['FactorV']
t2

# +
#https://www.cbs.nl/nl-nl/visualisaties/verkeer-en-vervoer/verkeer/verkeersprestaties-personenautos
#In 2022 legden alle Nederlandse personenauto’s samen 114,3 miljard kilometer af.
# -

mkverplsum1metlab(odinverplklinfo[odinverplklinfo['GrpTyp']=='Jaar'],kinfoflds,landcod)/1e9


# +
 #nu kenmerken per opvolgend slice

def _mkratios2 (indfs,indfgr):
    indfr=indfs.merge(indfgr,how='left')
    indfr['FactorKm']=np.where( indfr['FactorSum'] ==0,0,indfr['KmC']/indfr['FactorSum'])
    indfr['FactorC']=np.where( indfr['FactorSum'] ==0,0,indfr['FactorC']/indfr['FactorSum'] )
    return indfr

def _chkratios2 (indfr,grp):
#check data alle sommen kloppen
    indftst  =indfr.groupby(grp).agg('sum').reset_index()
    indfail2 = indfr[(np.isnan(indfr['FactorC'])) | (np.isnan(indfr['FactorKm']))]
    if len(indfail2)>0:
        print(len(indfail2),len(indfr),indfail2)
        raise("programming error")
    indfail1 = indftst[(abs(indftst['FactorC']-1)> 1e-6) & (indftst['FactorSum'] !=0)]
    if len(indfail1)>0:
        print(indfail1)
        raise("programming error")


def convert_diffgrpsidat2(indf,fg,kf,fclu,infflds,landcod,relative):
    indfs =indf.sort_values(by=fg+kf+['KAfstCluCode']).reset_index()    
    grp1=indfs.groupby(fg+kf)
    for if in infflds:
        indfs[if+"_c"] = indfs['FactorV']- (grp1['FactorV'].shift(1,fill_value=0.0) )
    ifcflds = (if+"_c" for if in infflds)    
    ifsflds = {if+"_c":if+"Sum" for if in infflds}
    if relative:
    #    print(indfgr)
    #controleer data dit de landerlijke waarden worden
        indfr = indfs.rename(columns={'FactorC':'FactorSum','KmC': 'KmSum' })        
        totdf=indf[indf['KAfstCluCode']==landcod].rename(columns={'FactorV':'FactorC','FactorKm':'KmC' })
        indfgrtot=indfr.groupby(fg+fclu)[['FactorSum','KmSum']].agg('sum').reset_index()
        tottst = _mkratios (totdf,indfgrtot,infflds)
        _chkratios (tottst,fg+fclu,infflds)

    #verdeel slices
        indfgr=indfr.groupby(fg+fclu+['KAfstCluCode'])[['FactorSum','KmSum']].agg('sum').reset_index()
        indfr = _mkratios (indfs,indfgr)    
        _chkratios (indfr,fg+fclu+['KAfstCluCode'])
        rv= indfr [fg+kf+ ['KAfstCluCode','FactorC','FactorKm'] ]
    else:
        rv= indfs [fg+kf+ ['KAfstCluCode','FactorC','FactorKm'] ]
    return rv 
odindiffgrpinfo2 = convert_diffgrpsidat2(odinverplklinfo,fitgrpse,kinfoflds,infoflds,
                                       ['GrpTyp'],landcod,True)
odindiffgrpinfo2


# +
#nu kenmerken per opvolgend slice

def _mkratios (indfs,indfgr):
    indfr=indfs.merge(indfgr,how='left')
    indfr['FactorKm']=np.where( indfr['FactorSum'] ==0,0,indfr['FactorKm']/indfr['FactorSum'])
    indfr['FactorC']=np.where( indfr['FactorSum'] ==0,0,indfr['FactorC']/indfr['FactorSum'] )
    return indfr

def _chkratios (indfr,grp):
#check data alle sommen kloppen
    indftst  =indfr.groupby(grp).agg('sum').reset_index()
    indfail2 = indfr[(np.isnan(indfr['FactorC'])) | (np.isnan(indfr['FactorKm']))]
    if len(indfail2)>0:
        print(len(indfail2),len(indfr),indfail2)
        raise("programming error")
    indfail1 = indftst[(abs(indftst['FactorC']-1)> 1e-6) & (indftst['FactorSum'] !=0)]
    if len(indfail1)>0:
        print(indfail1)
        raise("programming error")


def convert_diffgrpsidat(indf,fg,kf,fclu,landcod,relative):
    indfs=indf.sort_values(by='KAfstCluCode').reset_index()    
    indfs['FactorP'] = indfs.groupby(fg+kf)['FactorV'].shift(1,fill_value=0.0)
    indfs['KmP']     = indfs.groupby(fg+kf)['FactorKm'].shift(1,fill_value=0.0)
    indfs['FactorC']=indfs['FactorV']-indfs['FactorP']
    indfs['FactorKm']=indfs['FactorKm']-indfs['KmP']
    indfr = indfs.rename(columns={'FactorC':'FactorSum','FactorKm': 'KmSum' })
#    print(indfgr)
#controleer data dit de landerlijke waarden worden
    if relative:        
        totdf=indf[indf['KAfstCluCode']==landcod].rename(columns={'FactorV':'FactorC','FactorKm':'KmC' })
        indfgrtot=indfr.groupby(fg+fclu)[['FactorSum','KmSum']].agg('sum').reset_index()
        tottst = _mkratios (totdf,indfgrtot)
        _chkratios (tottst,fg+fclu)

    #verdeel slices
        indfgr=indfr.groupby(fg+fclu+['KAfstCluCode'])[['FactorSum','KmSum']].agg('sum').reset_index()
        indfr = _mkratios (indfs,indfgr)    
        _chkratios (indfr,fg+fclu+['KAfstCluCode'])
        rv= indfr [fg+kf+ ['KAfstCluCode','FactorC','FactorKm'] ]
    else:
        rv= indfs [fg+kf+ ['KAfstCluCode','FactorC','FactorKm'] ]
    return rv 
odindiffgrpinfo = convert_diffgrpsidat(odinverplklinfo,fitgrpse,kinfoflds,['GrpTyp'],landcod,True)
odindiffgrpinfo 
# -

#eens kijken of de Factor C en factorKm ergens op slaan
odindiffgrpinfo[odindiffgrpinfo['GrpTyp']=='Jaar'].sort_values(by=fitgrpse+['KAfstCluCode']+kinfoflds)

#eens kijken of de Factor C en factorKm ergens op slaan
odinverplgr.sort_values(by=fitgrpse+['GeoInd','PC4','KAfstCluCode'])


# +
def mkduurzconvert(lf,fg):
    ds= convert_duurzaam_slice(lf)
    ds=ds.drop(columns=["GrpTyp", "GrpVal","GrpV_label"])
    dssu=ds.groupby(['KAfstCluCode']+fg).agg('sum').reset_index()    
    return dssu
    
duurzrefnrs = mkduurzconvert(odindiffgrpinfo,fitgrpse)
# -

seaborn.lineplot(data=duurzrefnrs,x="FactorKm",y="KmActive",hue="GrpExpl")

seaborn.lineplot(data=duurzrefnrs,x="FactorKm",y="CO2GT",hue="GrpExpl")



# +
odinverplflgs_o=pd.read_pickle("../intermediate/ODINcatVN03db.pkl")
kflgsflds=['FactorV',"FactorKm","FactorAutoKm","FactorActiveKm"]

def _normflgvals (vg,kenmua,fg,gt):
    print (("_normflgvals"),gt)
    kenmu = kenmua[np.isin(kenmua["GrpTyp"],gt)].copy(deep=False)
    print(('kenmu',len(kenmu)))
    ds=vg.merge(kenmu,how='left',on=fg+['KAfstCluCode'])
    print(('ds',len(ds),ds.dtypes))
    ds['FactorV'] = ds['FactorC'] * ds['FCone']
    ds['FactorKm'] = ds['FactorC'] * ds['RitAfst']
    ds=ds.drop(columns=[ "GrpVal","GrpV_label"])
    debug =True
    if debug:
        ds['Checkn2'] = 1 * ds['FCone']
        indftst=ds.groupby(fg+['KAfstCluCode'] +  ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
        indfail1 = indftst[(abs(indftst['Checkn2']-1)> 1e-6) & (indftst['Checkn2'] !=0)]
        if len(indfail1)>0:
            print(indfail1)
            raise("programming error")   
    dssu=ds.groupby(fg+ ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
    return dssu


#pak nu database als odinverplgr, differentieer per slice, en plak 
#daar gegevens aan uit per groep genormeerde database als
def mksumperklas(verpl,kenm,fg,landcod):
    print(('verpl',len(verpl),verpl.dtypes) )
    v2=verpl.copy(deep=False).drop(columns='Variabele_naam')
    v2['FactorV']= v2['FactorVGen']+ v2['FactorVSpec']
    #in deze totalen zijn afstanden zinloos
    v2['FactorKm']=v2['FactorV']
    #deze dus niet normaliseren
    vg= convert_diffgrpsidat(v2,fg,['PC4','GeoInd'],['GeoInd'],landcod,False)   
    print(('vg',len(vg),vg.dtypes))
    kenmu=kenm.rename(columns={'FactorC':'FactorNorm'})
    print(('kenmu',len(kenmu),kenmu.dtypes))
    dssu=  _normflgvals (vg,kenmu,fg,[gt] ) 
    return dssu
    
cattots2 = mksumperklas(odinverplgr,odinverplflgs,kflgsflds,landcod)
#cattots2

# +
def _sumtogrpvals (vg,kenmua,fg,gt):
    print (("_sumtogrpvals"),gt)
    kenmu = kenmua[np.isin(kenmua["GrpTyp"],gt)].copy(deep=False)
    print(('kenmu',len(kenmu)))
    ds=vg.merge(kenmu,how='left',on=fg+['KAfstCluCode'])
    print(('ds',len(ds),ds.dtypes))
    ds['FactorV'] = ds['FactorC'] * ds['FCone']
    ds['FactorKm'] = ds['FactorC'] * ds['RitAfst']
    ds=ds.drop(columns=[ "GrpVal","GrpV_label"])
    debug =True
    if debug:
        ds['Checkn2'] = 1 * ds['FCone']
        indftst=ds.groupby(fg+['KAfstCluCode'] +  ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
        indfail1 = indftst[(abs(indftst['Checkn2']-1)> 1e-6) & (indftst['Checkn2'] !=0)]
        if len(indfail1)>0:
            print(indfail1)
            raise("programming error")   
    dssu=ds.groupby(fg+ ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
    return dssu


#pak nu database als odinverplgr, differentieer per slice, en plak 
#daar gegevens aan uit per groep genormeerde database als
def mksumperklas(verpl,kenm,fg,landcod):
    print(('verpl',len(verpl),verpl.dtypes) )
    v2=verpl.copy(deep=False).drop(columns='Variabele_naam')
    v2['FactorV']= v2['FactorVGen']+ v2['FactorVSpec']
    #in deze totalen zijn afstanden zinloos
    v2['FactorKm']=v2['FactorV']
    #deze dus niet normaliseren
    vg= convert_diffgrpsidat(v2,fg,['PC4','GeoInd'],['GeoInd'],landcod,False)   
    print(('vg',len(vg),vg.dtypes))
    kenmu=kenm.rename(columns={'FactorC':'FCone','FactorKm':'RitAfst'})
    print(('kenmu',len(kenmu),kenmu.dtypes))
    kenmgr = kenmu.groupby(["GrpTyp"])[['GrpVal']].agg('count').reset_index()
    print (kenmgr)
    hasret=False
    #nu ontstaat de combinatie. Deze in geheugen opslaan maakt het erg groot, en daarmee traag
    for gt in list(kenmgr["GrpTyp"]) :
        dssp=  _sumtogrpvals (vg,kenmu,fg,[gt] ) 
        if hasret:
            dssu= dssu.append(dssp)
        else:
            dssu=dssp
        hasret=True
    return dssu
    
cattots2 = mksumperklas(odinverplgr,odindiffgrpinfo,fitgrpse,landcod)
#cattots2
# -

# mooi, nu analyses welke raar zijn


# +
#cattots2
# -

o2=cattots2.groupby(["GrpTyp","GeoInd"]).agg('sum')
o2/74170863993


# +
def odinltot(indf,kf,infoflds,landcod):    
    totdf=indf[indf['KAfstCluCode']==landcod]    
    rv=totdf.groupby(['KAfstCluCode']+kf)[infoflds].agg('sum')
    return rv

ltot = odinltot (odinverplgr,['GeoInd'],['FactorV'],landcod)
ltot
# -

t2=mkverplsum1(odinverplklinfo)
t2['GemAfst']=t2['FactorKm']/ t2['FactorV']            
t2

print("Finished")

