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

useKAfstVQ  = useKAfstV [useKAfstV ["MaxAfst"] <4]
#print(xlatKAfstV)   
print(useKAfstVQ)   

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

odinverplklinfo = ODINcatVNuse.odinverplklinfo_o[np.isin(ODINcatVNuse.odinverplklinfo_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
odinverplgr =ODINcatVNuse.odinverplgr_o[np.isin(ODINcatVNuse.odinverplgr_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)
odinverplflgs =ODINcatVNuse.odinverplflgs_o[np.isin(ODINcatVNuse.odinverplflgs_o['KAfstCluCode'],maskKAfstV)].copy (deep=False)


#was odinverplgr=pd.read_pickle("../intermediate/ODINcatVN01db.pkl")
def deffactorv(rv):
    rv['FactorV'] = np.where ((rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ,
               0,rv['FactorVGen'] + 0* rv['FactorVSpec'] )
    skipsdf = rv [(rv['FactorVGen'] ==0 ) & ( rv['FactorVSpec']>0) ] [['PC4','MotiefV']].copy()
    return skipsdf
skipPCMdf = deffactorv(odinverplgr)
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


# +
def runexperiment(expname,incache0,mult,fitp,myuseKAfst):
    print("runexperiment: start processing "+expname)
    incache=dict()
    for fld in [3,5]:
        incache[fld]=np.where(np.isnan( incache0[fld]),0, incache0[fld])
    fname = "../intermediate/addgrds/"+expname+'.tif';
    ogrid= rasterio.open(fname)
    addcache = getcachedgrids(ogrid)

    mycache=dict()
    mycache[3] = incache[3] + mult*addcache[3]
    mycache[5] = incache[5] + mult * (addcache[3] + addcache[5])
    mycache[5]= np.where(mycache[5] < mycache[3] , mycache[3],  mycache[5])    
    ogrid.close()

#gebuik de parameters       
    rdf=predictnewdistr(cbspc4data,pc4inwgcache,mycache,myuseKAfst,xlatKAfstV,
                skipPCMdf,fitgrps, expdefs,fitp)
    return rdf

#rdf01=runexperiment('e0903a___swap_0010_02500',rudifungcache,1,fitpara,useKAfstVQ)


# -

globset="e0904a"
flst = glob.glob ("../intermediate/addgrds/"+globset+"*[a-z]_00*.tif")
elst = list(re.sub(".tif$",'',re.sub('^.*/','',f) ) for f in flst) 
elst


def grosumm(dfmdummy,lbl,myuseKAfstV,normfr):
    dfm=pd.read_pickle("../output/fitdf_"+lbl+".pd")
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
rdf00="dummydonotuse"
gs00=grosumm(rdf00,"orig",useKAfstVQ,[])
#print(gs00)
gs00T = grosumm(rdf00,"origchk",useKAfstVQ,gs00)
gs00T


def grosres (explst,incache0,mult,fitp,oridat,myuseKAfst,setname):
    rdf00N="dummydonotuse"
    gs00N = grosumm(rdf00N,"orig",myuseKAfst,[])
    st = ( grosumm("dummy2",exp,myuseKAfst ,gs00N)  for exp in explst )
    st = pd.concat (st)
    dto= grosumm(oridat,'brondat',myuseKAfst ,gs00N)
    #print(dto)
    st=st.append(dto)
    st.reset_index().to_excel("../output/fitrelres2_"+setname+".xlsx")
    return st
stQ = grosres (elst[0:3],rudifungcache,1,"fitparadummy", fitdatverplgr,useKAfstVQ,'Dbg01Q-'+globset)
stQ

stQ

     

# +
#check eens alles
#stQa = grosres (elst,rudifungcache,1,fitpara, fitdatverplgr,useKAfstVQ,'DBgf01Q-'+globset)
#stQa
# -


