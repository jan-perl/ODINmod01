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
#odinverplgr per PC aank of vertr
#odinverplklinfo: in diverse groepen (tijd, uur, jaar) mogelijk toe te voegen data
#odinverplflgs: vlaggen die gejoind kunnen worden aan oorspronkelijke data


# +
#useKAfstV=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
#xlatKAfstV=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
# -

fitgrps=['MotiefV','isnaarhuis']

odinverplflgs_o=pd.read_pickle("../intermediate/ODINcatVN03db.pkl")
kflgsflds=['FactorV',"FactorKm","FactorKmAuto","FactorKmActive","FactorVActive"]

odinverplklinfo_o=pd.read_pickle("../intermediate/ODINcatVN02db.pkl")
landcod=np.max(odinverplklinfo_o['KAfstCluCode'])
landcod


def chkvalues(chkdf,value,label):
    selvec= np.any(np.abs(chkdf - value) > 1e-6,axis=1)
#    print(selvec)
    selrecs = chkdf [selvec]
    if len(selrecs):
        print ("ERROR chkvalues ",label," Offending records not ",value)
        print (selrecs)
        assert len(selrecs)==0
    else:
        print ("OK chkvalues ",label)



oriKAfstVcods= np.array(range(landcod))+1
maxcuse= 3
maskKAfstV  = oriKAfstVcods [(oriKAfstVcods<=maxcuse ) |
                             (oriKAfstVcods==landcod )].copy()
maskKAfstV

# +
odinverplgr_o=pd.read_pickle("../intermediate/ODINcatVN01db.pkl")
#bij onlyOK=True: zet waarden van gemaskeerde postcodes op NaN
#todo: onlyOK wordt string: "ALL","Gen","Spec"
def deffactorv(rv,useKAfstV,UseSelFactorV): 
    if UseSelFactorV =='FactorVGen':
        rv['FactorV'] = np.where ((rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ,
               np.nan,rv['FactorVGen'] + 0* rv['FactorVSpec'] )
    elif UseSelFactorV =='FactorV':    
        rv['FactorV'] = rv['FactorVGen'] + rv['FactorVSpec'] 
    else:
        assert(0)
    return rv[np.isin(rv['KAfstCluCode'],useKAfstV) ].copy (deep=False)

MainUseSelFactorV='FactorVGen'
#voor de default run worden ALLE data gebruikt   
#hierop zijn dan ook de asserts in de code afgesteld
totaalmotief  =      74170863993
if MainUseSelFactorV=='FactorVGen':
    totaalmotief  =  69115694090
odinverplgr= deffactorv(odinverplgr_o,maskKAfstV,MainUseSelFactorV )
print(odinverplgr[odinverplgr['KAfstCluCode'] ==landcod]['FactorVGen'].sum()/totaalmotief/2)

chkvalues(pd.DataFrame( odinverplgr[odinverplgr['KAfstCluCode'] ==landcod][['FactorV']].agg('sum'))/totaalmotief/2,1.0,"som FactorV modaliteiten convert_duurzaam_slice")


# +
def selKafst_odin_o(odf,useKAfstV,UseSelFactorV):
    return odf[np.isin(odf['KAfstCluCode'],useKAfstV)
                                   & (odf['SelFactorV']==UseSelFactorV ) ] .copy (deep=False)

odinverplklinfo = selKafst_odin_o(odinverplklinfo_o,maskKAfstV,MainUseSelFactorV)
odinverplflgs =selKafst_odin_o(odinverplflgs_o,maskKAfstV,MainUseSelFactorV)
#odinverplklinfo = odinverplklinfo_o[np.isin(odinverplklinfo_o['KAfstCluCode'],maskKAfstV)
#                                   & (odinverplklinfo_o['SelFactorV']==MainUseSelFactorV ) ] .copy (deep=False)
#odinverplflgs =odinverplflgs_o[np.isin(odinverplflgs_o['KAfstCluCode'],maskKAfstV)
#                              & (odinverplflgs_o['SelFactorV']==MainUseSelFactorV) ].copy (deep=False)
# -

odinverplgr[['KAfstCluCode','GeoInd']].groupby('KAfstCluCode').agg('count')
#odinverplklinfo[['KAfstCluCode','FactorV']].groupby('KAfstCluCode').agg('count')



# +
#variabele nfogrps was not used; infogrps is in ODINcatVN.py
# nfogrps=['KHvm','AankUur','VertUur','Jaar']
# definition in ODINcatVN: allodinyr['FactorKm']= allodinyr['FactorV'] * allodinyr['AfstV'] *10
infoflds=['FactorV','FactorKm']

fitgrpse=fitgrps+['GrpExpl']
kinfoflds=["GrpTyp", "GrpVal","GrpV_label"]
odinverplklinfo


# -

#hulpfuncties, vergelijkbaar functie in ODINcatVN
#voor normalisatie van Factor V per jaar
def getcats(indf):
    ntab = indf.groupby('GrpVal')[['GrpVal']].agg('count')
    return ntab
welkejaren = getcats (odinverplklinfo[odinverplklinfo['GrpTyp']=='Jaar']).index
aantaljaren=len(welkejaren)
#print(welkejaren)
print(aantaljaren)

#synchronize with ODINcatVN, note: values of totals depend on MainUseSelFactorV
FactorVincols=['FactorVGen','FactorVSpec','FactorVGenActive','FactorVSpecActive']
odinverplgr[odinverplgr['KAfstCluCode'] ==landcod][FactorVincols].sum() / totaalmotief/2


#controleer de initiele database odinverplklinfo die afstandsommen / motief vertaalt
#naar afgeleide waardes van GrpTyp
def mkverplsum1(indf,landcod):    
    totdf=indf[indf['KAfstCluCode']==landcod]    
    rv=totdf.groupby("GrpTyp")[infoflds].agg('sum')
    return rv
o2=mkverplsum1(odinverplklinfo,landcod)
o2 *aantaljaren / totaalmotief
#assert niet eenvoudig mogelijk i.v.m. afwijkende normalisatie Jaar

# +
def mkverplsum1metlab(indf,kf,landcod):
    totdf=indf[indf['KAfstCluCode']==landcod]    
    rv=totdf.groupby(['KAfstCluCode']+kf)[infoflds].agg('sum')
    return rv
t2=(mkverplsum1metlab(odinverplklinfo,kinfoflds,landcod)*aantaljaren/totaalmotief).reset_index()

def convert_duurzaam_slice(t3):
    t2= t3[t3['GrpTyp']=='KHvm'].copy(deep=False)
    TTW_kgco2pkm=.149    
    amodes= [5,6]
    t2['CO2GT']=t2['FactorKm'] * np.where(t2['GrpVal']==1,TTW_kgco2pkm,0)
    t2['KmActive']=t2['FactorKm'] * np.where(np.isin(t2['GrpVal'],amodes),1,0)
    return t2
#aantallen in miljarden per jaar, laatste kolom in kms
#t2
t2=convert_duurzaam_slice(t2)
t2['GemAfst']=t2['FactorKm']/ t2['FactorV']
#print(pd.DataFrame( t2[['FactorV']].agg('sum') ) )
chkvalues(pd.DataFrame( t2[['FactorV']].agg('sum') ),1.0,"som FactorV modaliteiten convert_duurzaam_slice")
t2

# +
#https://www.cbs.nl/nl-nl/visualisaties/verkeer-en-vervoer/verkeer/verkeersprestaties-personenautos
#In 2022 legden alle Nederlandse personenauto’s samen 114,3 miljard kilometer af.
# -

t2=mkverplsum1metlab(odinverplklinfo[odinverplklinfo['GrpTyp']=='Jaar'],kinfoflds,landcod)/1e9
t2['GemAfst']=t2['FactorKm']/ t2['FactorV']   
t2


# +
#nu kenmerken per opvolgend slice,verouderde versie
#deze code wordt niet meer gebruikt

def _mkratioso (indfs,indfgr):
    indfr=indfs.merge(indfgr,how='left')
    indfr['FactorKm']=np.where( indfr['FactorSum'] ==0,0,indfr['FactorKm']/indfr['FactorSum'])
    indfr['FactorC']=np.where( indfr['FactorSum'] ==0,0,indfr['FactorC']/indfr['FactorSum'] )
    return indfr

def _mkratios (indfs,indfgr):
    indfr=indfs.merge(indfgr,how='left')
    normvals = indfr['FactorSum']  
    normzero = normvals==0
    indfr['FactorKm']=np.where(normzero,0,indfr['FactorKm']/normvals ) 
    indfr['FactorC'] =np.where(normzero,0,indfr['FactorC']/normvals )
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


def convert_diffgrpsidatold(indf,fg,kf,fclu,landcod,relative):
    indfs=indf.sort_values(by=fg+kf+['KAfstCluCode'] ).reset_index()    
    indfs['FactorP'] = indfs.groupby(fg+kf)['FactorV'].shift(1,fill_value=0.0)
    indfs['KmP']     = indfs.groupby(fg+kf)['FactorKm'].shift(1,fill_value=0.0)
    indfs['FactorC']=indfs['FactorV']-indfs['FactorP']
    indfs['FactorKm']=indfs['FactorKm']-indfs['KmP']
    indfr = indfs.rename(columns={'FactorC':'FactorSum','FactorKm': 'KmSum' })
#    print(indfgr)
#controleer data dit de landerlijke waarden worden
    if relative:        
        totdf=indf[indf['KAfstCluCode']==landcod].rename(
            columns={'FactorV':'FactorSum','FactorKm':'KmSum' }).drop(
            columns=['KAfstCluCode','index','AvgAfst'])
        ifsflds=['FactorSum','KmSum']
#        print((totdf.columns),len(totdf),totdf[ifsflds].sum())
#        indfgrtot=indfr.groupby(fg+kf)[ifsflds].agg('sum').reset_index()
#        print((indfgrtot.columns),len(indfgrtot),indfgrtot[ifsflds].sum())
        tottst = _mkratios (indfs,totdf)
        _chkratios (tottst,fg+kf)

    #verdeel slices
        indfgr=indfr.groupby(fg+fclu+['KAfstCluCode'])[['FactorSum','KmSum']].agg('sum').reset_index()
        indfr = _mkratios (indfs,indfgr)    
        _chkratios (indfr,fg+fclu+['KAfstCluCode'])
        rv= indfr [fg+kf+ ['KAfstCluCode','FactorC','FactorKm'] ]
    else:
        rv= indfs [fg+kf+ ['KAfstCluCode','FactorC','FactorKm'] ]
    return rv 
odindiffgrpinfo0 = convert_diffgrpsidatold(odinverplklinfo,fitgrpse,kinfoflds,['GrpTyp'],landcod,True)
odindiffgrpinfo0


# +
 #nu kenmerken per opvolgend slice

def _mkratios2 (indfs,indfgr,onflds,cflds):
#    print((indfs.columns),len(indfs))
#    print((indfgr.columns),len(indfgr))
    indfr=indfs.merge(indfgr,how='left')   
#    print((indfr.columns),len(indfr))
    normvals = indfr['FactorVSum']
    normzero = (normvals==0)
    for fld in cflds:
        indfr[fld] = np.where( normzero,0, indfr[fld] /normvals)       
    return indfr

def _chkratios2 (indfr,grp,cflds):
#check data alle sommen kloppen
    indftst  =indfr.groupby(grp).agg('sum').reset_index().sort_values(by=grp)
    chkvsumrange= True
    if chkvsumrange:
        indftstmin  =indfr.groupby(grp).agg('min').reset_index().sort_values(by=grp)
        indftstmax  =indfr.groupby(grp).agg('max').reset_index().sort_values(by=grp)
        indftst['Vsumrange '] = indftstmax['FactorVSum'] - indftstmin['FactorVSum'] 
    
    indfail2 = indfr[(np.isnan(indfr[cflds[0]])) | (np.isnan(indfr[cflds[1]]))]
    if len(indfail2)>0:
        print(grp,len(indfail2),len(indfr),indfail2)
        raise("programming error")
    indfail1 = indftst[(abs(indftst[cflds[0]]-1)> 1e-6) & (indftst['FactorVSum'] !=0)]
    if len(indfail1)>0:
        print(grp,len(indfail1),len(indfr),indfail1)
        raise("programming error")


def convert_diffgrpsidat(indf,fg,kf,infflds,fclu,pf,landcod,relative):
    indfs =indf [ ~np.isnan (indf['FactorV'])].sort_values(by=fg+kf+['KAfstCluCode']).reset_index(drop=True)    
    grp1=indfs.groupby(fg+kf)
    for fld in infflds:
#        indfs[fld+"_p"] = (grp1[fld].shift(1,fill_value=0.0) )
#        indfs[fld+pf] = indfs[fld]- indfs[fld+"_p"]        
        indfs[fld+pf] = indfs[fld]- (grp1[fld].shift(1,fill_value=0.0) )
    ifcrena = dict(( (fld+pf,fld+"Sum" )  for fld in infflds))    
    insrena = dict(( (fld,fld+"Sum" )  for fld in infflds))    
    ifcflds = list(ifcrena.keys())
    ifsflds = list(ifcrena.values())
    if relative:
    #    print(indfgr)
    #controleer data dit de landerlijke waarden worden
        indsumin = indfs.rename(columns=ifcrena)        
        totdf=indf[indf['KAfstCluCode']==landcod].rename(columns=insrena) .drop(
            columns=['KAfstCluCode','index','AvgAfst'])
#        print((totdf.columns),len(totdf),totdf[ifsflds].sum())
        indfgrtot=indsumin.groupby(fg+kf)[ifsflds].agg('sum').reset_index()
#        print((indfgrtot.columns),len(indfgrtot),indfgrtot[ifsflds].sum())
        if False:
            totdfo=totdf.rename(columns={'FactorVSum':'FactorSum','FactorKmSum':'KmSum'}).copy()
            indfso=indfs.rename(columns={'FactorV_c':'FactorC','FactorKm_c':'FactorKm'}).copy()
            tottst = _mkratios (indfso,totdfo)
            _chkratios (tottst,fg+kf)
        else:
            tottst = _mkratios2 (indfs,totdf,fg+kf,ifcflds)
            _chkratios2 (tottst,fg+kf,ifcflds)

        print(fg,fclu)
    #verdeel slices
        indfgr=indsumin.groupby(fg+fclu+['KAfstCluCode'])[ifsflds].agg('sum').reset_index()
        indfrat = _mkratios2 (indfs,indfgr,fg+kf,ifcflds)    
        _chkratios2 (indfrat,fg+fclu+['KAfstCluCode'],ifcflds)
        rv= indfrat [fg+kf+ ['KAfstCluCode'] + ifcflds ]
    else:
        rv= indfs [fg+kf+ ['KAfstCluCode'] + ifcflds ]
    return rv 

odindiffgrpinfo2 = convert_diffgrpsidat(odinverplklinfo,fitgrpse,kinfoflds,infoflds,
                                       ['GrpTyp'],"_c",landcod,True)
odindiffgrpinfo2


# +
def _dbg (df):
    s1 = df[df['GrpExpl'] =='1.0 4 : Van en naar het werk ronde elders']
    s2= s1[s1['GrpTyp']=='Jaar']
    return s2

s2r = _dbg(odinverplklinfo)    
s2r.groupby('KAfstCluCode').sum()
# -

#dit stukje code vergelijkt de data van de oude en nieuwe methode
#en laat de verschillend zien in FactorVd. Deze blijken nul te zijn
odindiffgrpinfo = odindiffgrpinfo2
odindiffgrpinfodifa = odindiffgrpinfo2.merge(odindiffgrpinfo0)
odindiffgrpinfodifa['FactorVd']= odindiffgrpinfodifa['FactorC'] - odindiffgrpinfodifa['FactorV_c']
odindiffgrpinfodifa['FactorKmd']= odindiffgrpinfodifa['FactorKm'] - odindiffgrpinfodifa['FactorKm_c']
odindiffgrpinfodifa.sort_values(by='FactorVd')

odindiffgrpinfodifa.groupby("GrpTyp").agg('sum')

#eens kijken of de Factor C en factorKm ergens op slaan
odindiffgrpinfo[odindiffgrpinfo['GrpTyp']=='Jaar'].sort_values(by=fitgrpse+['KAfstCluCode']+kinfoflds)

#eens kijken of de Factor C en factorKm ergens op slaan
odinverplgr.sort_values(by=fitgrpse+['GeoInd','PC4','KAfstCluCode'])


# +
def mkduurzconvert(lf,fg):
    ds= convert_duurzaam_slice(lf.rename(columns={'FactorV_c':'FactorV', 'FactorKm_c':'FactorKm'}))
    ds=ds.drop(columns=["GrpTyp", "GrpVal","GrpV_label"])
    dssu=ds.groupby(['KAfstCluCode']+fg).agg('sum').reset_index()    
    return dssu
    
duurzrefnrs = mkduurzconvert(odindiffgrpinfo,fitgrpse)
# -

seaborn.lineplot(data=duurzrefnrs,x="FactorKm",y="KmActive",hue="GrpExpl")

seaborn.lineplot(data=duurzrefnrs,x="FactorKm",y="CO2GT",hue="GrpExpl")

odinverplflgs

kflgsflds

odindiffflginfo = convert_diffgrpsidat(odinverplflgs,fitgrpse,[],kflgsflds, [],"_c",landcod,False)
odindiffflginfo


def mkdatadiff0(verpl,fg,landcod):    
#    print(('verpl',len(verpl),verpl.dtypes) )
    v2=verpl.copy(deep=False)
#    v2['FactorV']= v2['FactorVGen']+ v2['FactorVSpec']
    #in deze totalen zijn afstanden zinloos
    v2['FactorKm']=v2['FactorV']
    #deze dus niet normaliseren
    vg= convert_diffgrpsidat(v2,fg,['PC4','GeoInd'],infoflds,['GeoInd'],"_v",landcod,False) 
#    print(('vg',len(vg),vg.dtypes))
    return vg
#datadiffcache =    mkdatadiff0(odinverplgr,fitgrpse,landcod)
#datadiffcache.dtypes


# +
#Maak per afstandsgroep oplopende verschillen zodat die 
#voor een deel categorie (PC4/ Geoind en fitgrpse ) oplopen tot het totaal in landcod

def mkdatadiff(verpl,fg,infof,grpind,landcod):    
#    print(('verpl',len(verpl),verpl.dtypes) )
    v2=verpl.copy(deep=False)
#    v2['FactorV']= v2['FactorVGen']+ v2['FactorVSpec']
    #in deze totalen zijn afstanden zinloos
    v2['FactorKm']=v2['FactorV']
    #deze dus niet normaliseren
    vg= convert_diffgrpsidat(v2,fg,[grpind,'GeoInd'],
                                          infof,['GeoInd'],"_v",landcod,False) 
#    print(('vg',len(vg),vg.dtypes))
    return vg
datadiffcache = mkdatadiff(odinverplgr,fitgrpse,infoflds,'PC4',landcod)
datadiffcache.dtypes
# -

c2 =datadiffcache.groupby("GeoInd")[['FactorV_v','FactorKm_v']].agg('sum')/totaalmotief
chkvalues( c2,1.0,"mkdatadiff(odinverplgr")
c2

#zoeken naar missende data als bovenstaande d2 uit datadiffcache niet op 0 uit komt
datadiferr=False
if datadiferr:
    deagg=['GeoInd','PC4','MotiefV','isnaarhuis']
    stot= odinverplgr[odinverplgr['KAfstCluCode'] ==landcod].drop(columns=["KAfstCluCode"]). groupby(deagg).agg('sum')
    print(stot.dtypes)
    print(len(stot))
    ctot= datadiffcache.groupby(deagg).agg('sum')
    print(ctot.dtypes)
    print(len(ctot))
    dtot = stot.merge(ctot,on=deagg,how='outer')
    dtot['Fdiff'] = abs(dtot['FactorV']-dtot['FactorV_v'])
    print(len(dtot))
    dtot2 = dtot[dtot['Fdiff']> 1e-6]
    print(len(dtot2))
    dtot2


#en nog emer deep-dive om te pinpointen
def gtinsp1(df):
    return df[ (df['GeoInd']=='AankPC') & (df['PC4']==1012) &
                (df['MotiefV']==6) &  (df['isnaarhuis']==6) ].sort_values('KAfstCluCode')
if datadiferr:
    print( gtinsp1(odinverplgr) )
    print( gtinsp1(datadiffcache) )
    print( gtinsp1(datadiffcache).sum() )
    print( gtinsp1(dtot.reset_index()) )


# +
#routine wordt ook aangegoepen om totalen per PC4 te maken
def normflgvals (vg,kenmu,fg,cflds,outkeeppart):    
    ds=vg.merge(kenmu,how='left',on=fg+['KAfstCluCode'])
#    print(('ds',len(ds),ds.dtypes))
    #normaliseren doen we hier, omdat er steeds precies 1 match is
    for fld in cflds:
        ds[fld] = ds['FactorV_v'] * ds[fld+'_c']/ ds['FactorV_c']   
    debug = False
    if debug:
        ds['Checkn2'] = 1 * ds['FCone']
        indftst=ds.groupby(fg+['KAfstCluCode'] +  ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
        indfail1 = indftst[(abs(indftst['Checkn2']-1)> 1e-6) & (indftst['Checkn2'] !=0)]
        if len(indfail1)>0:
            print(indfail1)
            raise("programming error")   
#    todrop1= list( (fld+"_v" for fld in cflds) )
    todrop2= list ( (fld+"_c" for fld in cflds) ) 
#    print(todrop2)
    dsc= ds.drop(columns=todrop2)               
    dssu=dsc.groupby(fg+outkeeppart+ ['GeoInd'] ).agg('sum').reset_index()    
    return dssu


#pak nu database als odinverplgr, differentieer per slice, en plak 
#daar gegevens aan uit per groep genormeerde database als
def mkinfosums(vg,kenmu,fg,kenmcols,landcod):    
#    print(('vg',len(vg),vg.dtypes))    
#    print(('kenmu',len(kenmu),kenmu.dtypes))
    dssu=  normflgvals (vg,kenmu,fg,kenmcols,[]) 
    return dssu
    
infotots2 = mkinfosums(datadiffcache ,odindiffflginfo,fitgrpse,kflgsflds,landcod)
infotots2.dtypes
# -

infotots2 

# +
#dubbel check: komen FactorV_v en FactorV precies uit (resultaat 1.0) op de totalen in ODINcatVN ?

o2=infotots2.groupby(["GeoInd"]).agg('sum')
if True:
    print(o2/totaalmotief)    
chkvalues( (o2[['FactorV_v','FactorKm_v','FactorV']] )/totaalmotief,1.0,"Factors infotots2")
# -

# nu maskeren en tof wegschrijven per PC4


# +
#dit is kopie uit _normflgvals, die kennelijk nog niet heel consistent is

#merge de data groep voor groep op gt 
#controleer ook of er geen data kwijt zijn geraakt
def _sumtogrpvals (vg,kenmua,fg,gt):
    debug =False
    if debug:
        print (("debug: _sumtogrpvals"),gt)
    kenmu = kenmua[np.isin(kenmua["GrpTyp"],gt)].copy(deep=False)
    if debug:
        print(('len kenmu',len(kenmu)))
    ds=vg.merge(kenmu,how='left',on=fg+['KAfstCluCode'])
    if debug:
        print(('len ds',len(ds),ds.dtypes))
    ds['FactorV'] = ds['FactorV_v'] * ds['FactorV_c']
    ds['FactorKm'] = ds['FactorV_v'] * ds['FactorKm_c']
    ds=ds.drop(columns=[ "GrpVal","GrpV_label"])
    if debug:
        ds['Checkn2'] = 1 *  ds['FactorV_c']
        indftst=ds.groupby(fg+['KAfstCluCode'] +  ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
        indfail1 = indftst[(abs(indftst['Checkn2']-1)> 1e-6) & (indftst['Checkn2'] !=0)]
        #TODO: uitzoeken wat deze check doet en waarom deze waarschuwt
        if len(indfail1)>0:
            print("Interne consistentie: Checkn2 hoort 1 of 0 te zijn, niets er tussen in")   
            print(indfail1)
            print("programming error")   
#            raise("programming error")   
    dssu=ds.groupby(fg+ ["GrpTyp",'GeoInd'] ).agg('sum').reset_index()    
    return dssu


#pak nu database als odinverplgr, differentieer per slice, en plak 
#daar gegevens aan uit per groep genormeerde database als
def mksumperklas(vg,kenmu,fg):
#    print(('vg',len(vg),vg.dtypes))
#    kenmu=kenm.rename(columns={'FactorC':'FCone','FactorKm':'RitAfst'})
#    print(('kenmu',len(kenmu),kenmu.dtypes))
    kenmgr = kenmu.groupby(["GrpTyp"])[['GrpVal']].agg('count').reset_index()
#    print (kenmgr)
    #nu ontstaat de combinatie. Deze in geheugen opslaan maakt het erg groot, en daarmee traag
    #gebruik map om deze een voor een uit te voeren
    dssu0 = map(lambda gt: _sumtogrpvals (vg,kenmu,fg,[gt] )  ,list(kenmgr["GrpTyp"])) 
    dssu1 = pd.concat(dssu0)
    return dssu1
    
if True:
    cattots2 = mksumperklas(datadiffcache,odindiffgrpinfo,fitgrpse)
    o2=cattots2.groupby(["GrpTyp","GeoInd"]).agg('sum')
    print(o2/totaalmotief)    
    #resultaten in o2 FactorV horen te normaliseren tot 1
    chkvalues(o2[['FactorV']]/totaalmotief,1.0, "resultaten in o2 FactorV ")
    cattots2
# -

# mooi, nu analyses welke raar zijn


# +
def odinltot(indf,kf,infoflds,landcod):    
    totdf=indf[indf['KAfstCluCode']==landcod]    
    rv=totdf.groupby(['KAfstCluCode']+kf)[infoflds].agg('sum')
    return rv

ltot = odinltot (odinverplgr,['GeoInd'],['FactorV'],landcod)
chkvalues(ltot/totaalmotief,1.0, "Factor odinltot (odinverplgr ")
ltot
# -

t2=mkverplsum1(odinverplklinfo,landcod)/totaalmotief
t2=t2.reset_index()
t2['grpfact']= np.where (t2['GrpTyp']=='Jaar',1,aantaljaren)
t2=t2.set_index('GrpTyp')
t2['FactorV']=t2['FactorV']*t2['grpfact']
t2['FactorKm']=t2['FactorKm']*t2['grpfact']
#print(d2)
t2['GemAfst']=t2['FactorKm']/ t2['FactorV']            
chkvalues(t2[['FactorV']] ,1.0, "Factor mkverplsum1(odinverplklinfo ")
t2

o2= datadiffcache.groupby(['GeoInd']).sum()/totaalmotief
#ook hier hoort FactorV_v / totaalmotief weer op 1.0 uit te komen.
print(o2)
chkvalues(o2[['FactorV_v','FactorKm_v']],1.0, "datadiffcache FactorV_v")

print("Finished")


