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
#Vergelijk resultaten van modellen (en origineel)

# +
#beschrijf kenmerken
#volledig additief
# binnen cirkels
# -

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import re
import time
import glob

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"

#voor gemeentegrenzen; kost hier wel heel veel geheugen voor. Kijken hoe dit te vermijden
import ODiN2readpkl

from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
from sklearn import linear_model
import seaborn

import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

stryear='2020'
cbspc4data =pd.read_pickle("../intermediate/CBS/pc4data_"+stryear+".pkl")
cbspc4data= cbspc4data.sort_values(by=['postcode4']).reset_index()

cbspc4data['oppervlak'] = cbspc4data.area
cbspc4data['aantal_inwoners'] = np.where(cbspc4data['aantal_inwoners'] <0,0,
                                         cbspc4data['aantal_inwoners'] )

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

#rudifunset, heb originele data niet nodig, alleen grid
#Rf_net_buurt=pd.read_pickle("../intermediate/rudifun_Netto_Buurt_o.pkl") 
#Rf_net_buurt.reset_index(inplace=True,drop=True)
#gemaakt in ROfietsbalans2
rudifuntifname=calcgdir+'/oriTN2-NL.tif'
rudifungrid= rasterio.open(rudifuntifname)


def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst
pc4inwgcache = getcachedgrids(pc4inwgrid)
rudifungcache = getcachedgrids(rudifungrid)

# nu nog MXI overzetten naar PC4 ter referentie




# +
#nu ODIN ranges opzetten
#we veranderen NIETS aan odin data
#wel presenteren we het steeds als cumulatieve sommen tot een bepaalde bin
# -

useKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
xlatKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <20].copy()
maxcuse= np.max(useKAfstV[useKAfstV ["MaxAfst"] !=0] ['KAfstCluCode'])
xlatKAfstV  = xlatKAfstVa [(xlatKAfstVa['KAfstCluCode']<=maxcuse ) |
                           (xlatKAfstVa['KAfstCluCode']==np.max(useKAfstV[ 'KAfstCluCode']) )].copy()
#print(xlatKAfstV)   
print(useKAfstV)   

#dit was alleen voor ODIN1KAFmo om met kleine sets te werken.
#deze variabele niet gebruiken voor verwerken hele sets, wel voor regressie tests op Q set
useKAfstVQ  = useKAfstV [useKAfstV ["MaxAfst"] <4]

# +
#import ODiN2pd
#import ODiN2readpkl
# -

usePC4MXI=True


def mkfietswijk3pc4(pc4data,pc4grid,rudigrid):
    #pc4lst=pc4grid.read(1)
    pc4lst=pc4grid[1]
    outdf=pc4data[['postcode4','aantal_inwoners']].rename(columns={'postcode4':'PC4'} )
#    outdf['aantal_inwoners_gr2'] = rasteruts1.sumpixarea(pc4lst,pc4grid.read(3) )
#    outdf['S_MXI22_BWN'] = rasteruts1.sumpixarea(pc4lst,rudifungrid.read(3) )
#    outdf['S_MXI22_BAT'] = rasteruts1.sumpixarea(pc4lst,rudifungrid.read(5) )
    outdf['aantal_inwoners_gr2'] = rasteruts1.sumpixarea(pc4lst,pc4grid[3] )
    outdf['S_MXI22_BWN'] = rasteruts1.sumpixarea(pc4lst,rudigrid[3] )
    outdf['S_MXI22_BAT'] = rasteruts1.sumpixarea(pc4lst,rudigrid[5] )
    outdf['S_MXI22_BAN'] = outdf['S_MXI22_BWN'] - outdf['S_MXI22_BAT'] 
    if usePC4MXI:
        outdf['S_MXI22_NS'] = outdf['S_MXI22_BWN']  / (outdf['S_MXI22_BWN']  + outdf['S_MXI22_BAN'] )
        outdf['S_MXI22_BB'] = outdf['S_MXI22_NS']

    outdf['S_MXI22_BG'] = outdf['S_MXI22_BWN'] / pc4data['oppervlak']        
    outdf['S_MXI22_GB'] = pd.qcut(outdf['S_MXI22_BB'], 10)
    outdf['S_MXI22_GG'] = pd.qcut(outdf['S_MXI22_BG'], 10)
    outdf['aantal_inwoners_d2'] = outdf['aantal_inwoners_gr2'] -outdf['aantal_inwoners']
    return outdf
#fietswijk3pc4=mkfietswijk3pc4(cbspc4data,pc4inwgrid,rudifungrid)
fietswijk3pc4=mkfietswijk3pc4(cbspc4data,pc4inwgcache,rudifungcache)
bd=fietswijk3pc4 [abs(fietswijk3pc4['aantal_inwoners_d2'] ) > 1 ]

expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0}

from importlib import reload  # Python 3.4+
if False:
        foo = reload(ODINcatVNuse)

#het inlezen van odinverplgr loopt in deze versie via ODINcatVNuse
import ODINcatVNuse

print(useKAfstV) 
maskKAfstV= list(useKAfstV['KAfstCluCode'])
maskKAfstV

# +
#odinverplklinfo = ODINcatVNuse.odinverplklinfo_o[np.isin(ODINcatVNuse.odinverplklinfo_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
#odinverplgr =ODINcatVNuse.odinverplgr_o[np.isin(ODINcatVNuse.odinverplgr_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
#odinverplflgs =ODINcatVNuse.odinverplflgs_o[np.isin(ODINcatVNuse.odinverplflgs_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)

MainUseSelFactorV='FactorVGen'
odinverplgr= ODINcatVNuse.deffactorv(ODINcatVNuse.odinverplgr_o,maskKAfstV,MainUseSelFactorV )
odinverplklinfo = ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplklinfo_o,maskKAfstV,MainUseSelFactorV)
odinverplflgs =ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplflgs_o,maskKAfstV,MainUseSelFactorV)


# -

#was odinverplgr=pd.read_pickle("../intermediate/ODINcatVN01db.pkl")
def findskippc(rv):
    rv['FactorV'] = np.where ((rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ,
               0,rv['FactorVGen'] + 0* rv['FactorVSpec'] )
    skipsdf = rv [(rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ] [['PC4','MotiefV']].copy()
    return skipsdf
skipPCMdf = findskippc(odinverplgr)
skipPCMdf

# +
#top . nu eerst kijken naar kengetallen van basis fit


# +
#check 1 reisigerskm auto kilometers

# +
def mkdatadiff3(verpl,fg,infof,landcod):    
    return ODINcatVNuse.mkdatadiff(verpl,fg,landcod)

def mkdatadiff2(verpl,fg,infof,grpind,landcod):    
#    print(('verpl',len(verpl),verpl.dtypes) )
    v2=verpl.copy(deep=False)
#    v2['FactorV']= v2['FactorVGen']+ v2['FactorVSpec']
    #in deze totalen zijn afstanden zinloos
    v2['FactorKm']=v2['FactorV']
    #deze dus niet normaliseren
    vg= ODINcatVNuse.convert_diffgrpsidat(v2,fg,[grpind,'GeoInd'],
                                          infof,['GeoInd'],"_v",landcod,False) 
#    print(('vg',len(vg),vg.dtypes))
    return vg
#ddc_indat =  mkdatadiff2(fitdatverplgr,ODINcatVNuse.fitgrpse,
#                         DINcatVNuse.infoflds,ODINcatVNuse.landcod)


# -

odindiffflginfo= ODINcatVNuse.convert_diffgrpsidat(odinverplflgs,
                ODINcatVNuse.fitgrpse,[],ODINcatVNuse.kflgsflds, [],"_c",ODINcatVNuse.landcod,False)

lblo='origchk'
fitdatverplgr = pd.read_pickle("../output/fitdf_"+lblo+".pd")

fitdatverplgr.dtypes

#ddc_indat =  ODINcatVNuse.mkdatadiff(fitdatverplgr,ODINcatVNuse.fitgrpse,ODINcatVNuse.landcod)
ddc_indat =  mkdatadiff2(fitdatverplgr,
                         ODINcatVNuse.fitgrpse,ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)
totinf_indat = ODINcatVNuse.mkinfosums(ddc_indat,odindiffflginfo,
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
totinf_indat

ddc_fitdat =  mkdatadiff2(fitdatverplgr.rename (
       columns={'FactorV':'FactorO', 'FactorEst':'FactorV' }),
            ODINcatVNuse.fitgrpse,  ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)

totinf_fitdat = ODINcatVNuse.mkinfosums(ddc_fitdat,odindiffflginfo,
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
totinf_fitdat.groupby(["GeoInd"]).agg('sum')

totinf_indat.groupby(["GeoInd"]).agg('sum')


#let op dit kan niet, want hierover hebben we gesommeerd !
def permotief(totin,totfit,kflgs):
    renai = dict ( ( (flg,flg+"_i") for flg in kflgs) ) 
    agrpi=totin.groupby(["MotiefV","GeoInd"])[kflgs].agg('sum') /1e6
    agrpi['Gemafst_i'] = agrpi ['FactorKm'] /agrpi ['FactorV'] 

    renaf = dict ( ( (flg,flg+"_f") for flg in kflgs) )
    agrpf=totfit.groupby(["MotiefV","GeoInd"])[kflgs].agg('sum') /1e6
    agrpf['Gemafst_f'] = agrpf ['FactorKm'] /agrpf ['FactorV'] 
    agrp=pd.merge(agrpi.reset_index().rename(columns=renai),  
                  agrpf.reset_index().rename(columns=renaf) )
    return agrp
motlst= permotief(totinf_indat,totinf_fitdat,ODINcatVNuse.kflgsflds)  
motlst.to_excel("../output/orif_permot2.xlsx")


def woonbalans1(totin,kflgs):
    ause = totin[np.isin(totin['isnaarhuis'],[5,6]) &
                 np.isin(totin['MotiefV'],[1]) ].copy(deep=False)
    ause['Ri2']=np.where(ause['isnaarhuis']==5, 
                        np.where(ause['GeoInd']=='AankPC', 'actzijde' , 'huiszijde' ),
                        np.where(ause['GeoInd']!='AankPC', 'actzijde' , 'huiszijde' )    
                           )
    agrp = ause .groupby(['MotiefV','GeoInd','isnaarhuis'])[kflgs].agg('sum')    /1e6   
    agrp ['Gemafst'] = agrp ['FactorKm'] /agrp ['FactorV'] 
    agrp = agrp.reset_index()
#    agrp ['gcol'] = "Mot_"+ np.array(agrp ['MotiefV'] .astype('string'))+"_nh_"+ np.array(agrp ['isnaarhuis'] .astype('string'))
    agrp ['p1'] = "Mot_"
    agrp ['p2'] = "_nh_"
    mc= agrp['MotiefV'].astype(int).astype(str)
    nc= agrp['isnaarhuis'].astype(str)
    p1= agrp['p1'].astype(str)
    p2= agrp['p2'].astype(str)
    agrp ['gcol'] =p1 + mc +p2 + nc 
    agrp2 = agrp.pivot(index="GeoInd", columns="gcol", values="FactorV")
    return agrp2
woonbalans1(totinf_indat,ODINcatVNuse.kflgsflds)                       


def woonbalans(totin,kflgs):
    iso=False
    for mot in [1,7]:
        for vn in [5,6]:
            colnm = ('Mot_{}_nh_{}').format(mot,vn)
            fruse= totin [(totin['MotiefV']==mot )&(totin['isnaarhuis']==vn*1.0 )]
            fruse = fruse[['GeoInd','FactorV']].rename(columns={'FactorV': colnm})
            fruse=fruse.copy()
            if iso:
                outf=outf.merge(fruse)
            else:
                outf=fruse
            iso=True
    return outf
woonbalans(totinf_indat,ODINcatVNuse.kflgsflds)

woonbalans(totinf_fitdat,ODINcatVNuse.kflgsflds).reset_index() # .drop(columns='gcol')    

rdf00=fitdatverplgr

#eerste proging: bouw elst precies zo op als in ODIN1KAfmo
#eigenlijk goede manier van regressie
globset="e0904a"
flst = glob.glob ("../intermediate/addgrds/"+globset+"*[a-z]_00*.tif")
elst = list(re.sub(".tif$",'',re.sub('^.*/','',f) ) for f in flst) 
elst


#vergelijkbaar met origineel; gebruikt gelezen data
def gropc4stats(dfm,runname,lbl,myuseKAfstV,normfr):
    dfm=pd.read_pickle("../output/fitdf_"+runname+"_"+lbl+".pd")
    mymaskKAfstV= list(myuseKAfstV['KAfstCluCode'])
    if lbl=='brondat':
        dfmu=dfm[np.isin(dfm['KAfstCluCode'],mymaskKAfstV)].copy (deep=False)
    else:
        dfmu=dfm.rename (columns={'FactorV':'FactorO', 'FactorEst':'FactorV' })
    ddc_fitdat =  mkdatadiff2(dfmu, ODINcatVNuse.fitgrpse,  ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)

    # myodinverplflgs / myodindiffflginfo kunnen ook buiten loop worden berekend, maar dit borgt consisitente
    # voor relatief weinig extra rekentijd
    myodinverplflgs =ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplflgs_o,maskKAfstV,MainUseSelFactorV)
    myodindiffflginfo= ODINcatVNuse.convert_diffgrpsidat(myodinverplflgs,
                ODINcatVNuse.fitgrpse,[],ODINcatVNuse.kflgsflds, [],"_c",ODINcatVNuse.landcod,False)
    totinf_fitdat = ODINcatVNuse.mkinfosums(ddc_fitdat,myodindiffflginfo,                                            
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
    rv =totinf_fitdat.groupby(["GeoInd"]).agg('sum')
    wb =woonbalans(totinf_fitdat,ODINcatVNuse.kflgsflds).groupby(["GeoInd"]).agg('sum')
    rv=rv.join(wb)
#    rv=rv.set_index(['GeoInd'])
    if (len (normfr) >0):
        rv = rv/ normfr
        rv['label']=lbl
    return rv 
gs00=gropc4stats(rdf00,"Dbg01","orig",useKAfstV,[])
#print(gs00)
gs00T = gropc4stats(rdf00,"Dbg01","origchk",useKAfstV,gs00)
gs00T


#vergelijkbaar met origineel; gebruikt gelezen data
def grosumm(dfmdummy,runname,lbl,myuseKAfstV,normfr):
    dfm=pd.read_pickle("../output/fitdf_"+runname+"_"+lbl+".pd")
    mymaskKAfstV= list(myuseKAfstV['KAfstCluCode'])
    if lbl=='brondat':
        dfmu=dfm[np.isin(dfm['KAfstCluCode'],mymaskKAfstV)].copy (deep=False)
    else:
        dfmu=dfm.rename (columns={'FactorV':'FactorO', 'FactorEst':'FactorV' })
    ddc_fitdat =  mkdatadiff2(dfmu, ODINcatVNuse.fitgrpse,  ODINcatVNuse.infoflds,'mxigrp',ODINcatVNuse.landcod)

    # myodinverplflgs / myodindiffflginfo kunnen ook buiten loop worden berekend, maar dit borgt consisitente
    # voor relatief weinig extra rekentijd
    myodinverplflgs =ODINcatVNuse.odinverplflgs_o[np.isin(
         ODINcatVNuse.odinverplflgs_o['KAfstCluCode'],mymaskKAfstV)].copy (deep=False)
    myodindiffflginfo= ODINcatVNuse.convert_diffgrpsidat(myodinverplflgs,
                ODINcatVNuse.fitgrpse,[],ODINcatVNuse.kflgsflds, [],"_c",ODINcatVNuse.landcod,False)
    totinf_fitdat = ODINcatVNuse.mkinfosums(ddc_fitdat,myodindiffflginfo,                                            
                       ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,ODINcatVNuse.landcod)
    rv =totinf_fitdat.groupby(["GeoInd"]).agg('sum')
    wb =woonbalans(totinf_fitdat,ODINcatVNuse.kflgsflds).groupby(["GeoInd"]).agg('sum')
    rv=rv.join(wb)
#    rv=rv.set_index(['GeoInd'])
    if (len (normfr) >0):
        rv = rv/ normfr
        rv['label']=lbl
    return rv 
gs00=grosumm(rdf00,"Dbg01","orig",useKAfstV,[])
#print(gs00)
gs00T = grosumm(rdf00,"Dbg01","origchk",useKAfstV,gs00)
gs00T


#vergelijkbaar met origineel; gebruikt gelezen data
def grosres (explst,incache0dummy,mult,fitpdummy,oridat,myuseKAfst,runname,setname):
    rdf00N="dummydonotuse"
    gs00N = grosumm(rdf00N,runname,"orig",myuseKAfst,[])
    st = ( grosumm("dummy2",runname,exp,myuseKAfst ,gs00N)  for exp in explst )
    st = pd.concat (st)
    dto= grosumm(oridat,runname,'brondat',myuseKAfst ,gs00N)
    #print(dto)
    st=st.append(dto)
    st.reset_index().to_excel("../output/fitrelres2-"+runname+setname+".xlsx")
    return st
stQ = grosres (elst[0:3],'rudifungcachedummy',1,"fitparadummy", 'fitdatverplgr',useKAfstVQ,'Dbg02Q-',globset)
stQ


# +
def getStori(runname,setname):
    stqori= pd.read_excel("../output/fitrelres_"+runname+setname+".xlsx")
    return stqori 

def chkStori(stqcalc,stqori):
    ci=  stqcalc.reset_index().set_index(["label","GeoInd"])
    oi= stqori.set_index(["label","GeoInd"])
    stqdiff= ci -oi
    return stqdiff
rv= chkStori(stQ,getStori('Dbg02Q-',globset))
np.max(np.abs(rv))
# -

if 1==1:
    StoriQ=getStori('Set04Q-',globset)
    EloriQ=list(StoriQ[StoriQ['GeoInd']=="AankPC"]['label'])
    EloriQ=EloriQ[0:-1]
    EloriQ
    StorcQ=grosres (EloriQ,'rudifungcachedummy',1,"fitparadummy", 'fitdatverplgr',useKAfstVQ,'Set04Q-',globset)
rv= chkStori(StorcQ,StoriQ)
np.max(np.abs(rv))
#succes: all < 1e-8

if 1==1:
    StoriN=getStori('Set04N-',globset)
    EloriN=list(StoriN[StoriN['GeoInd']=="AankPC"]['label'])
    EloriN=EloriN[0:-1]
    EloriN
    StorcN=grosres (EloriN,'rudifungcachedummy',1,"fitparadummy", 'fitdatverplgr',useKAfstV,'Set04N-',globset)
rv= chkStori(StorcN,StoriN)
np.max(np.abs(rv))
#succes: all < 1e-8

# +
#check eens alles
#stQa = grosres (elst,rudifungcache,1,fitpara, fitdatverplgr,useKAfstVQ,'DBgf01Q-'+globset)
#stQa
# -
#einde van poging om fitdatres te herhalen
#nu weer naar oorspronkelijke data en ative mode vergelijingken


# +
#allerlei active modes vergelijkingen

# +
MainUseSelFactorV='FactorVGen'
def deffactorvin(rv,UseSelFactorV): 
    if UseSelFactorV =='FactorVGen':
        rv['FactorVInActive'] =rv['FactorVGenActive']            
    elif UseSelFactorV =='FactorV':    
        rv['FactorVInActive'] =rv['FactorVGenActive'] + rv['FactorVSpecActive']       
    else:
        assert(0)
    rv= rv.reset_index().rename(columns={"FactorV":"FactorVin"})
    return rv

#ng zonder geoind: dan komt iedere PC precies een keer voor
pc4orisumng = (odinverplgr[odinverplgr['KAfstCluCode'] == ODINcatVNuse.landcod] .groupby (
    ['PC4'] ).agg('sum')*0.5).drop (columns=['index','MotiefV','isnaarhuis','KAfstCluCode'])
pc4orisumng = deffactorvin(pc4orisumng,MainUseSelFactorV)
pc4orisum =   (odinverplgr[odinverplgr['KAfstCluCode'] == ODINcatVNuse.landcod] .groupby (
    ['PC4','GeoInd'] ).agg('sum')).drop(columns=['index','MotiefV','isnaarhuis','KAfstCluCode'])
pc4orisum =   deffactorvin(pc4orisum,  MainUseSelFactorV)
pc4orisum
# -

odinverplgr= ODINcatVNuse.deffactorv(ODINcatVNuse.odinverplgr_o,maskKAfstV,MainUseSelFactorV )
odinverplklinfo = ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplklinfo_o,maskKAfstV,MainUseSelFactorV)
odinverplflgs =ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplflgs_o,maskKAfstV,MainUseSelFactorV)

MainUseSelFactorVmspec='FactorV'
odinverplgrmspec= ODINcatVNuse.deffactorv(ODINcatVNuse.odinverplgr_o,maskKAfstV,MainUseSelFactorVmspec )
odinverplklinfomspec = ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplklinfo_o,maskKAfstV,MainUseSelFactorVmspec)
odinverplflgsmspec =ODINcatVNuse.selKafst_odin_o(ODINcatVNuse.odinverplflgs_o,maskKAfstV,MainUseSelFactorVmspec)

datadiffcachemspec = ODINcatVNuse.mkdatadiff(odinverplgrmspec,ODINcatVNuse.fitgrpse,
                                        ODINcatVNuse.infoflds,'PC4',ODINcatVNuse.landcod)
datadiffcache = ODINcatVNuse.mkdatadiff(odinverplgr,ODINcatVNuse.fitgrpse,
                                        ODINcatVNuse.infoflds,'PC4',ODINcatVNuse.landcod)

o2=datadiffcachemspec.groupby(['GeoInd']).sum()/ODINcatVNuse.totaalmotief
#ODINcatVNuse.chkvalues(o2[['FactorV_v','FactorKm_v']],1.0, "datadiffcache FactorV_v")
o2

totaalVgen=69115694090
datadiffcache.groupby(['GeoInd']).sum()/totaalVgen

datadiffcache.groupby(['KAfstCluCode','GeoInd']).sum()


#maak een frame met FactorVActive geschat op basis van FactorV, gesommeerd over astandsklasses
#maak nog apart per motief
def calcFactVPC4(ddc,perGeoInd):
    if perGeoInd:
        return (ODINcatVNuse.normflgvals(ddc ,odindiffflginfo,
                        ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,['PC4'] 
                    ).      groupby(['PC4','GeoInd'] ).agg('sum')).reset_index().drop (
     columns=['MotiefV','isnaarhuis','KAfstCluCode'])
    else:
        return (ODINcatVNuse.normflgvals(ddc ,odindiffflginfo,
                                        ODINcatVNuse.fitgrpse,ODINcatVNuse.kflgsflds,['PC4'],
            ).      groupby(['PC4'] ).agg('sum')*0.5).reset_index().drop (
                columns=['MotiefV','isnaarhuis','KAfstCluCode'])
calcFactVPC4(datadiffcache,False)
calcFactVPC4(datadiffcache,True)


# +
#OK:: Hier gebeurt de magic voor het vergelijken van de active modes per PC4
#de output uit calcFactVPC4 (zie hierboven) wordt als verklarende variabele gebruikt.
#deze wordt gemerged met pc4orisum die uit de oorspronkelijke data komt
#Er zijn even minder naam conflicten omdat de oorspronkelijke data nog Gen en Spec kennen

def mrgpcactdiffr(in2pc,inpc4ori):
    infotots2pcdiff=  in2pc.merge(inpc4ori,how='inner')
    infotots2pcdiff['FactorVChk'] =infotots2pcdiff['FactorV']- infotots2pcdiff['FactorVin']
    infotots2pcdiff['RatActiveVIn'] = infotots2pcdiff['FactorVInActive']/infotots2pcdiff['FactorVin']
    infotots2pcdiff['RatActiveV'] = infotots2pcdiff['FactorVActive']/infotots2pcdiff['FactorV']
    infotots2pcdiff['FitRatVActive'] = infotots2pcdiff['RatActiveV'] 
    infotots2pcdiff['FactorVChkActive'] =infotots2pcdiff['FactorVActive'] -infotots2pcdiff['FactorVInActive']
    #infotots2pcdiff[infotots2pcdiff['FactorVChk']  !=0]
    return(infotots2pcdiff)
infotots2pcdiff=mrgpcactdiffr(calcFactVPC4(datadiffcache,True),pc4orisum)
t2= infotots2pcdiff.groupby(['GeoInd']).sum()/ODINcatVNuse.totaalmotief
t2.T
#/totaalVgen
# -

infotots2pcdiffmspecng=mrgpcdiffr(calcFactVPC4(datadiffcachemspec,False),pc4orisumng)
infotots2pcdiffmspecng.sum()/ODINcatVNuse.totaalmotief

infotots2pcdiffng=mrgpcdiffr(calcFactVPC4(datadiffcache,False),pc4orisumng)
infotots2pcdiffng.sum()/totaalVgen

#deze scatter plot ziet er goed uit
minFactorVplot=5e6
def mkactpccmpfig(indf0,title):
#    indf0['RatActiveVSc'] = 1.75*indf0['RatActiveV']**2 +.05
    indf=indf0[(indf0['FactorVin']>minFactorVplot ) & (indf0['RatActiveVIn']>1e-2)].copy(deep=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    indf['sizser'] = np.abs(indf['FactorVin']) 
    c2=sns.lineplot(data=indf, x='RatActiveV',y='FitRatVActive',ax=ax)
    c1=sns.scatterplot(data=indf, x='RatActiveV',y='RatActiveVIn',hue='sizser',size='sizser',ax=ax)
    fig.suptitle(title)
    ax.set_xlabel('Schatting active modes a.h.v. land gem. motief en afstand')
    ax.set_ylabel('Punten: aandeel active per PC4')
#    ax.set_xscale('log')
#    ax.set_yscale('log')
    return(fig)
r1=mkactpccmpfig(infotots2pcdiffng,'Data center functions excluded')


# +
#deze scatter plot ziet er goed uit
minFactorVplot=5e6
def mkactpccmpfig(indf0,title):
#    indf0['RatActiveVSc'] = 1.75*indf0['RatActiveV']**2 +.05
    indf=indf0[(indf0['FactorVin']>minFactorVplot ) & (indf0['RatActiveVIn']>1e-2)].copy(deep=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    indf['sizser'] = np.abs(indf['FactorVin']) 
    nlev=50
    mybin=np.linspace(0,1,10)
#    print(mybin)
    indf['RatActiveVl']=np.round (indf['RatActiveV'] *nlev)*1.0/nlev
    indf['RatActiveVInl']=np.round (indf['RatActiveVIn'] *nlev)*1.0/nlev
#    indflev = indf[['RatActiveVl','RatActiveVInl','sizser']].groupby(['RatActiveVl','RatActiveVInl']).agg('sum').reset_index()
    indflev= np.histogram2d(indf['RatActiveV'],indf['RatActiveVIn'],bins=mybin,weights=indf['sizser'] )
    indflevh= indf.pivot_table(index='RatActiveVl',columns='RatActiveVInl',values='sizser'
                            ,aggfunc='sum'  ,fill_value=0)
    indflevh=np.sqrt(indflevh)
#    print(indflev)
    c2=sns.lineplot(data=indf, x='RatActiveV',y='FitRatVActive',ax=ax)
    indf['sizsers'] = (indf['sizser']) 
    c1=plt.hist2d(indf['RatActiveV'],indf['RatActiveVIn'],bins=100,weights=indf['sizsers'],
                 cmap="rocket_r" ,alpha=0.3)
    
#    c1=sns.heatmap(data=indflev, ax=ax,cmap="rocket_r" ,cbar=True,alpha=0.3)
#    ax.invert_yaxis()
#cmap="mako",     

#    c1=sns.scatterplot(data=indf, x='RatActiveV',y='RatActiveVIn',hue='sizser',size='sizser',ax=ax)
    fig.suptitle(title)
    ax.set_xlabel('Schatting active modes a.h.v. land gem. motief en afstand')
    ax.set_ylabel('Punten: aandeel active per PC4')
#    ax.set_xscale('log')
#    ax.set_yscale('log')
    return(fig)
r1=mkactpccmpfig(infotots2pcdiffng,'Data center functions excluded')
# -



# +
def _regressgrp(indf, yvar, xvars,pcols):  
#        reg_nnls = LinearRegression(fit_intercept=False )
#        print(('o',len(indf)) )
        y_train=indf[yvar]
        X_train=indf[xvars]
        if 1==1:
            #smask = ~ (np.isnan(y_train) | np.isnan(np.sum(X_train)) )
            smask =  (y_train >0) & (np.sum(X_train,axis=1) >0) 
            indf= indf[smask]
            y_train=indf[yvar]
            X_train=indf[xvars]
#            print(('f', len(indf)))
        else:
            y_train[np.isnan(y_train)]=0.1
        if(len(indf)==0) :
            rv=np.zeros(len(xvars))
        else:
            fit1 = nnls(X_train, y_train)    
            rv=pd.DataFrame(fit1[0],index=pcols).T
        return(rv)

#fit op 'FactorVInActive': weeg naar indf['FactorVin']

def fitactivem(indf,fitmode):
    indf['EstActiveV'] = indf['FactorVin'] *indf['FactorVActive']/indf['FactorV']
    indf['EstActiveVS'] = indf['FactorVActive']/indf['FactorV']
    indf['EstActiveVS'] = indf['EstActiveVS'] * indf['EstActiveVS'] * indf['FactorVin']
    if fitmode=='VGenlin':
        fcols=['FactorVin','EstActiveV']
        pcols=['ParamV','ParamVActive']
    elif fitmode=='VGensq':
        fcols=['FactorVin','EstActiveVS']
        pcols=['ParamV','ParamVActiveS']
    else:
        fcols=['FactorVin','EstActiveV','EstActiveVS']
        pcols=['ParamV','ParamVActive','ParamVActiveS']
    rf= _regressgrp (indf, 'FactorVInActive', fcols, pcols)   
    if fitmode=='VGenlin':
        rf['ParamVActiveS']=0
    elif fitmode=='VGensq':
        rf['ParamVActive']=0        
    print (rf)
    #fitdf.merge(rf,how='outer',on=[])
    indf['FitVActive'] = indf['FactorVin'] * rf['ParamV'][0] +  \
            indf['EstActiveV'] * rf['ParamVActive'][0] +\
            indf['EstActiveVS'] * rf['ParamVActiveS'][0]
    diffs = indf['FitVActive'] - indf['FactorVInActive']    
    chisq = np.sum(diffs*diffs) / np.sum(indf['FactorVInActive']* indf['FactorVInActive'])
    indf['FitRatVActive'] = indf['FitVActive'] /indf['FactorVin']
    print(chisq)
    return (chisq,rf)

#tofit=infotots2pcdiffng.copy(deep=False)
#noot: VGensq mag er wel mooier uit zien, maar de chi^2 is een factor 4 slechter
(chisq,rf)= fitactivem(infotots2pcdiffng,'VGensq')
r1=mkactpccmpfig(infotots2pcdiffng,'fitted VGensq')
(chisq,rf)= fitactivem(infotots2pcdiffng,'VGenlin')
r1=mkactpccmpfig(infotots2pcdiffng,'fitted VGenlin')
(chisq,rf)= fitactivem(infotots2pcdiffng,'VGenpara')
r1=mkactpccmpfig(infotots2pcdiffng,'fitted VGenpara')
# -

infotots2pcdiffng [(infotots2pcdiffng ['RatActiveVIn']>.8 )& (infotots2pcdiffng['FactorV']>minFactorVplot ) ]

(chisq,rf)= fitactivem(infotots2pcdiffmspecng,'VGensq')
r1=mkactpccmpfig(infotots2pcdiffmspecng,'All ODIN data incl center functions')
(chisq,rf)= fitactivem(infotots2pcdiffmspecng,'VGenlin')
r1=mkactpccmpfig(infotots2pcdiffmspecng,'All ODIN data incl center functions')

# +
showgdir="../intermediate/showgrids"
act1tifname=showgdir+'/actmo01.tif'
#let op: gaat uit van gesorteerd en reset_index dataframa
def make1stgridgeorel (tifname,indf,usecols,nanval):
    if (np.max(indf.index -np.array(range(len(indf)))) !=0):
        raise(GridArgError("make1stgridgeo: Index not in order (sorting not checked)"))        
    grid = rasteruts1.createNLgrid(100,tifname,8,'')
    dfrefs= rasteruts1.makegridcorr (indf,grid)
    #veel niet gevonden uit landelijk !
    indf['area_geo'] = indf.area
    missptdf= rasteruts1.findmiss(indf,dfrefs)
    for col in usecols:
        indf[col]=np.where((indf[col] == nanval ) | (indf['FactorV'] < minFactorVplot),0,indf[col])
    imagelst=rasteruts1.mkimgpixavgs(grid,dfrefs,False,False, indf[usecols],False)  
    grid.close()
    grid = rasterio.open(tifname)
    return grid

act1tifcols= ['RatActiveVIn','FitRatVActive']
act1grid= make1stgridgeorel (act1tifname,cbspc4data.merge(infotots2pcdiffng,how='left',
                                left_on=['postcode4'],right_on='PC4'),act1tifcols,999)
# -

#common code with Mkaddgrids
gemeentendata ,  wijkgrensdata ,    buurtendata = ODiN2readpkl.getgwb(2020)

#common code with Mkaddgrids
grgem = gemeentendata[(gemeentendata['H2O']=='NEE') & (gemeentendata['AANT_INW']>1e5) ]


#common code with Mkaddgrids
def setaxreg(ax,reg):
    if reg=='htn':
        ax.set_xlim(left=137000, right=143000)
        ax.set_ylim(bottom=444000, top=452000)
    elif reg=='utr':    
        ax.set_xlim(left=113000, right=180000)
        ax.set_ylim(bottom=480000, top=430000)


act1grid= rasterio.open(act1tifname)
actcache={}
actcache[3]= act1grid.read(3) 
actcache[5]= act1grid.read(4) 

# +
#code modified from Mkaddgrids
nlextent=[0,280000,300000, 625000]
def actpltland(ecache,fld,selextent,fname,txt):
    minv=1e-7
    mos=np.log(minv)/ np.log(10)
    image1= ecache[3]
    image1[0][0]=1
    image1[0][1]=0
    image2= ecache[5]
    image2[0][0]=1
    image2[0][1]=0
    nv = ecache[3] - ecache[5]    
    image3= nv

    lststr =  'Values for {}, min1 log10 W {}, max log10 W {}, min O {} , max log10 O {}  , max log10 negs {}'. format (txt, np
                    .min(image1)-mos,np.max(image1)-mos , np.min(image2)-mos,np.max(image2)-mos,np.max(image3)-mos)
    print (lststr)
    fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(50, 20))
    #image = np.isnan(image)
    ax1.imshow(image1,cmap='jet',alpha=.6,extent=selextent)
    ax2.imshow(image2,cmap='jet',alpha=.6,extent=selextent)
    ax3.imshow(np.where(image3>=0, image3,np.nan),cmap='Reds',alpha=.6,extent=selextent)
    ax3.imshow(np.where(image3<=0, -image3,np.nan),cmap='Blues',alpha=.6,extent=selextent)
#    ax3.imshow(image3,cmap='Greens',alpha=.6,extent=selextent)
    grgem.boundary.plot(color='green',ax=ax1,alpha=.8)
    grgem.boundary.plot(color='green',ax=ax2,alpha=.8)
    grgem.boundary.plot(color='green',ax=ax3,alpha=.8)
    if (selextent[0] == 113000):
        setaxreg(ax1,'utr')
        ax1.invert_yaxis()
        setaxreg(ax2,'utr')
        ax2.invert_yaxis()
        setaxreg(ax3,'utr')
        ax3.invert_yaxis()
#   ax1.colorbar()
#    ax2.colorbar()
#    axes[0].plot(x1, y1)
#    axes[1].plot(x2, y2)
    ax2.set_title(lststr)
    fig.tight_layout()
    figname = "../intermediate/addgrds/fig_"+fname+'.png';
    fig.savefig(figname) 
    return fig

r1=actpltland(actcache,3,nlextent,'actexample','actexample')
# -

#common code with Mkaddgrids
utrextent=[113000,180000,430000,480000 ]
def mkloccach(ecache,selextent,oriextent):
    oridim = ecache[3].shape
    xmul= oridim[1]/(oriextent[1]-oriextent[0])
    ymul= oridim[0]/(oriextent[2]-oriextent[3])
    #print([oridim,oriextent,xmul,ymul])
    
    xmin=int((selextent[0]-oriextent[0])*xmul)
    xmax=int((selextent[1]-oriextent[0])*xmul)
    ymin=int((selextent[2]-oriextent[3])*ymul)
    ymax=int((selextent[3]-oriextent[3])*ymul)
    #print([xmin,xmax,ymin,ymax])
    ocache=dict()
    for imgidx in [3,5]:
        #print(ecache[imgidx].shape)
        sli= ecache[imgidx][ymax:ymin,xmin:xmax]
        odim = sli.shape
        #print(odim)
        ocache[imgidx]= sli
    return ocache
actcacheutr=mkloccach(actcache,utrextent,nlextent)
r1=actpltland(actcacheutr,3,utrextent,'actexampleut','actexampleut')

#goede check, maar geen actieve mode info:
datadiffcache.groupby(['KAfstCluCode','GeoInd']).sum().rename(columns= {"FactorV_v":"FactorV" })


#bijdrage actieve mode per afstandsklasse, uit odinverplflgs (Gemaakt met VGen - dus zonder VSpec)
def mkverplAsftsu(inflgs):
    odinverplAsftsu=inflgs.groupby(['KAfstCluCode']).sum().reset_index()
    #.rename(       columns= {"KAfstCluCode":"Kafst"})
    odinverplAsftsu["FactorV"] = odinverplAsftsu["FactorV"] - odinverplAsftsu["FactorV"].shift(1,fill_value=0)
    odinverplAsftsu["FactorVActive"] = odinverplAsftsu["FactorVActive"] - odinverplAsftsu["FactorVActive"].shift(1,fill_value=0)
    odinverplAsftsu["ActFractOri"] = odinverplAsftsu["FactorVActive"] / odinverplAsftsu["FactorV"]
    return(odinverplAsftsu)
odinverplAsftsu=mkverplAsftsu(odinverplflgs)


# +
#maakt een database voor active mode vergelijking
#neemt een database db1, gesommeerd naar KAfstV en tabel myKAfstV
#voegt samen, en maakt velden voor berekende waarden en labels
def prepactsdb(db1,myKAfstV):
    KafstActiveVori = db1.merge(myKAfstV,how='left')
    KafstActiveVori['FactorVr'] = KafstActiveVori['FactorV'] / np.max(KafstActiveVori['FactorV'] )
    KafstActiveVori['FactorVCum'] = KafstActiveVori['FactorV'].cumsum()
    KafstActiveVori['FactorVCum'] = KafstActiveVori['FactorVCum']/ np.max( KafstActiveVori['FactorVCum'])
    KafstActiveVori['FactorPCum'] = KafstActiveVori['FactorVCum'].shift(1,fill_value=0)+1e-6
    KafstActiveVori['KAfstVFmt'] = np.where(KafstActiveVori['MaxAfst'] ==0,"verder",
                                            KafstActiveVori['MaxAfst'].map(lambda x:"%3g"%(x)) )
    return(KafstActiveVori)

KafstActiveVori = prepactsdb(odinverplAsftsu,useKAfstV)


# +
#neemt een database  myActiveVori en maakt standaard plot
# gebruikt kolommen FactorVCum ,ActFractOri , FactorV en KAfstVFmt
def pltactsdb(myActiveVori,savtag,title):
    KafstActiveVorid= myActiveVori.copy(deep=True)
    KafstActiveVorid['FactorPCum']=KafstActiveVorid['FactorVCum']
    KafstActiveVorid = pd.concat([ myActiveVori,KafstActiveVorid] ) .sort_values(by='FactorPCum')                       
    KafstActiveVorid  

    chart= sns.relplot(data=KafstActiveVorid, x='FactorPCum',y='ActFractOri',kind='line')
    totavgact= sum(myActiveVori['ActFractOri'] * myActiveVori['FactorV'] ) / \
                      sum( myActiveVori['FactorV'] )
    #totavgact= sum( KafstActiveVori['FactorVr'] )
    chart.fig.suptitle(title)
    #chart.fig.suptitle('Totaal aandeel actieve mobiliteit %.3f'%(totavgact))            
    chart.set_xlabels('Aandeel van de afstandklasse')
    chart.set_ylabels('Fractie van reizen, per afstandsklasse' )
    labcolor="#3498db" # choose a color
    for x, y, name in zip(myActiveVori['FactorVCum'],myActiveVori['ActFractOri'],
                          myActiveVori['KAfstVFmt']):
        chart.ax.text(x+.02, y , name, color=labcolor)
    chart.ax.text(.2,.2 , 'aandeel actieve\nmodes %.3f'%(totavgact), color=labcolor) 
    chart.ax.text(.7,.7 , 'aandeel gemotoriseerde\nmodes %.3f'%(1-totavgact), color=labcolor) 
    chart.ax.set_xlim(0,1)
    chart.ax.set_ylim(0,1)
    figname = "../output/act_reg_"+savtag+"_"+'m1.svg';
    chart.fig.savefig(figname, bbox_inches="tight")

pltactsdb(KafstActiveVori,'orisel','DIN data Gen - afstanden > 15 geclusterd')  
# -

odinverplAsftsumspec=mkverplAsftsu(odinverplflgsmspec)
KafstActiveVorimspec = prepactsdb(odinverplAsftsumspec,useKAfstV)
pltactsdb(KafstActiveVorimspec,'orisel','Originele ODIN data (incl Spec) - afstanden > 15 geclusterd')  

# +
#OK , we weten nu dat we 
#a fitdatres2- kunnen maken, gelijk aan fitdatres-
#b geo vergelijkingen active modes kunnen maken
#nu op de manier van 
# -

flst = glob.glob ("../intermediate/addgrds/"+globset+"*base_00*.tif")
bname = list(re.sub(".tif$",'',re.sub('^.*/','',f) ) for f in flst) [0]
bname


#vergelijkbaar met origineel; gebruikt gelezen data
def getddcInAct(dfmdummy,runname,lbl,myuseKAfstV,normfr):
    dfm=pd.read_pickle("../output/fitdf_"+runname+"_"+lbl+".pd")
    mymaskKAfstV= list(myuseKAfstV['KAfstCluCode'])
    if lbl=='brondat':
        dfmu=dfm[np.isin(dfm['KAfstCluCode'],mymaskKAfstV)].copy (deep=False)
    else:
        dfmu=dfm.rename (columns={'FactorV':'FactorOrig', 'FactorEst':'FactorV' ,'mxigrp':'PC4'})
    ddc_fitdat =  mkdatadiff2(dfmu, ODINcatVNuse.fitgrpse,  ODINcatVNuse.infoflds,'PC4',ODINcatVNuse.landcod)    
    return ddc_fitdat
#ddc base will be used in several aggregates
ddc_base = getddcInAct("dummy2","Set05N-",bname,useKAfstV ,[]) 


# +
#PC4 data active modes in zelfde format als pc4orisum
def getPC4InAct(ddc_fitdat):
    cfact=calcFactVPC4(ddc_fitdat,True)
    rv = cfact[['PC4','GeoInd','FactorV','FactorVActive']].rename(
        columns={"FactorV":"FactorVin",'FactorVActive':'FactorVInActive'})         
    return rv 

basePC4act = getPC4InAct(ddc_base ) 
#basePC4act = getPC4InAct("dummy2","Dbg01Q","origPC4chk",useKAfstV ,[]) 
baseactpcdiff=mrgpcactdiffr(calcFactVPC4(datadiffcache,True),basePC4act)
t2= baseactpcdiff.groupby(['GeoInd']).sum()/ODINcatVNuse.totaalmotief
t2.T
#/totaalVgen
# -

(chisq,rf)= fitactivem(baseactpcdiff,'VGenpara')
r1b=mkactpccmpfig(baseactpcdiff,'Base fit RUDIDUN geo')

print("Finished")


