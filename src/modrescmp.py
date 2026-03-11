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
from matplotlib import colors 
import matplotlib.ticker as ticker

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
#data_crs="epsg:28992"
if 1==1:
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['min_zoom']= 0
    prov0['max_zoom'] =12
    print (prov0)

pland= cbspc4data.plot(alpha=0.4)
cx.add_basemap(pland, source= prov0,crs=cbspc4data.crs)

#alternatief: gebruik webcoordinaten in plot
cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
phtn = cbspc4datahtn.to_crs(epsg=plot_crs).plot(alpha=0.4)
cx.add_basemap(phtn, source= prov0)


# +
#en nu netjes, met schaal in km
def plaxkm(x, pos=None):
      return '%.0f'%(x/1000.)

def addbasemkmsch(ax,mapsrc):
    cx.add_basemap(ax,source= mapsrc,crs="epsg:28992")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))
    
fig, ax = plt.subplots(figsize=(6, 4))
cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']>3990) & (cbspc4data['postcode4']<3999)]
phtn = cbspc4datahtn.plot(ax=ax,alpha=0.4)
addbasemkmsch(ax,prov0)
# -

cbspc4datahtn = cbspc4data[(cbspc4data['postcode4']==3995)]
phtn = cbspc4datahtn.plot()
cx.add_basemap(phtn, source= prov0,crs=cbspc4data.crs)

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
useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <180].copy()
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

expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0,'SP':1.0}

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
flst = glob.glob ("../intermediate/addgrds/"+globset+"*[a-z0-9]_00*.tif")
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

#eerste proging: bouw elst precies zo op als in ODIN1KAfmo
#eigenlijk goede manier van regressie
globset="e0904a"
flst = glob.glob ("../output/fitdf_Dbg02Q-_"+globset+"*.pd")
elstDbg02Q = list(re.sub(".pd$",'',re.sub('^.*Dbg02Q-_','',f) ) for f in flst) 
elstDbg02Q


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
stQ = grosres (elstDbg02Q,'rudifungcachedummy',1,"fitparadummy", 'fitdatverplgr',useKAfstVQ,'Dbg02Q-',globset)
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
#gebruik nu active mode grafieken uit validate
import actmodval

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


# + endofcell="--"
#PC4 data active modes in zelfde format als pc4orisum
def getPC4InAct(ddc_dat,perGeoInd):
    cfact=actmodval.calcFactVPC4(ddc_dat,perGeoInd)
    keepcols=['PC4','FactorV','FactorVActive']
    if perGeoInd:
        keepcols=keepcols+['GeoInd']
    rv = cfact[keepcols].rename(
        columns={"FactorV":"FactorVin",'FactorVActive':'FactorVInActive'})         
    return rv 

basePC4act = getPC4InAct(ddc_base ,False) 
#basePC4act = getPC4InAct("dummy2","Dbg01Q","origPC4chk",useKAfstV ,[]) 
baseactpcdiffng=actmodval.mrgpcdiffr(actmodval.calcFactVPC4(actmodval.datadiffcache,False),basePC4act)
baseactpcdiffng.sum()/actmodval.totaalVgen

#t2= baseactpcdiffng.groupby(['GeoInd']).sum()/ODINcatVNuse.totaalmotief
#t2.T
#/totaalVgen
# -
#t2.T
#
# --

(chisq,rf)= actmodval.fitactivem(baseactpcdiffng,'VGenpara')
r1b=actmodval.mkactpccmpfig(baseactpcdiffng,'Base fit RUDIDUN geo tov afstandsklasse resultaat')

showgdir=actmodval.showgdir
base1tifname=showgdir+'/basemo01.tif'
def mkgeofig1(mytifname,mytots2pcdiffng,figbase,im1tit,im2tit,im3tit):
    act1grid= actmodval.make1stgridgeorel (mytifname,cbspc4data.merge(mytots2pcdiffng,how='left',
                                left_on=['postcode4'],right_on='PC4'),actmodval.act1tifcols,999)
    act1grid= rasterio.open(mytifname)
    actcache={}
    actcache[3]= act1grid.read(3) 
    actcache[5]= act1grid.read(4) 
    act1grid.close()
    r1=actmodval.actpltland(actcache,3,actmodval.nlextent,figbase+"-nl",figbase+"-nl",im1tit,im2tit,im3tit)
    actcacheutr=actmodval.mkloccach(actcache,actmodval.utrextent,actmodval.nlextent)
    r1=actmodval.actpltland(actcacheutr,3,actmodval.utrextent,figbase+"-ut",figbase+"-ut",im1tit,im2tit,im3tit)
    return r1
r1=mkgeofig1(base1tifname,baseactpcdiffng,'actcmp-base0-dcls',
              'Aandeel actieve modes obv RUDIFUN dichtheden',
              'Aandeel actieve modes obv ODIN afstanden',
              'Rood: reizen voor meer keuze, Blauw: minder keuze vanwege reizen')

basePC4actright= basePC4act.rename(
        columns={"FactorV":"FactorVOrig"}).rename(columns={"FactorVin":"FactorV",
                 'FactorVInActive':'FactorVActive'})
baseactrawmpcdiffng=actmodval.mrgpcdiffr(basePC4actright ,actmodval.pc4orisumng)
#baseactrawmpcdiffng
t2=baseactrawmpcdiffng.sum()/actmodval.totaalVgen
t2.T

(chisq,rf)= actmodval.fitactivem(baseactrawmpcdiffng,'VGenlin')
r1b=actmodval.mkactpccmpfig(baseactrawmpcdiffng,'Base fit RUDIDUN geo tov ODIN metingen')

r1=mkgeofig1(base1tifname,baseactrawmpcdiffng,'actcmp-base0-raw',
              'Aandeel actieve modes in ODIN',
              'Aandeel actieve modes obv RUDIFUN dichtheden',
              'Rood: verder gefietst dan gemiddeld, Blauw: minder gefietst dan gemiddeld')


# +
def geoactch(explst,basedatright):
    for exp in explst:
        chg_base = getddcInAct("dummy2","Set05N-",exp,useKAfstV ,[]) 
        chgPC4act = getPC4InAct(chg_base ,False) 
        chgactrawmpcdiffng=actmodval.mrgpcdiffr(basedatright , chgPC4act)
        t2=chgactrawmpcdiffng.sum()/actmodval.totaalVgen
        print (t2.T)
        r1=mkgeofig1(base1tifname,chgactrawmpcdiffng,'actcmp-base0-'+exp,
              'Aandeel actieve modes experiment '+exp,
              'Aandeel actieve modes obv RUDIFUN dichtheden',
              'Rood: verder gefietst dan gemiddeld, Blauw: minder gefietst dan gemiddeld')
    return 1
    
geoactch(elst, basePC4actright)   
# -

print("Finished")


