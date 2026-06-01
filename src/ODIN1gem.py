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
#een set doorsnedes van ODiN gegevens
#voor exploratie
#traag door eerst pc6 plotjes
#draai minimaal 1 maal rasteruts in docker container om geopandas te laden
# -

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

import RUDIbas

myname='ODIN1gem'
suprtests= myname in RUDIbas.suprtests 
suprdata= myname in RUDIbas.suprdata
#suprtests=True
print ('Suprtests',suprtests)

import ODiN2readpkl

#set Houten als target
targgem =321
targgemcode = 'GM0321'
targpc4=range(3990,4000)


def getgemyrs():
    gems1= [ ODiN2readpkl.getgwb(year)[0].assign(jaar=year) for year in range(2020,2026)]
    rv=pd.concat(gems1)
    rv=geopandas.GeoDataFrame(rv, geometry=rv['geometry'])
#    rv= rv.mask(rv==-99999999.0, np.nan)
    return rv
allgem=getgemyrs()


def selgemyrs(iv,gemcode):
    mv=iv[iv['GM_CODE'] == gemcode]
    sv=iv[(iv['H2O']=='NEE' ) | (iv['GM_CODE'] != gemcode) ].groupby(['jaar']).agg('sum').reset_index()
    for c in ['GM_CODE','GM_NAAM']:
        sv[c]="rest_NL"
    rv=mv.append(sv)
    rv=rv.copy().reset_index()
    return rv
allgem_sum=selgemyrs(allgem,targgemcode)
allgem_sum

prov0=cx.providers.nlmaps.grijs.copy()
#print( odf.crs)
plot_crs=3857
#data_crs="epsg:28992"
if 1==1:
#    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:28992/{z}/{x}/{y}.png'
    prov0['url']='https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/{variant}/EPSG:3857/{z}/{x}/{y}.png'    
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
#    prov0['bounds']=  [[48.040502, -1.657292 ],[56.110590 ,12.431727 ]]  
    prov0['min_zoom']= 0
    prov0['max_zoom'] =12
    print (prov0)

plot_crs=3857
plot_crs="epsg:28992"
allgem_sum.set_crs(crs="epsg:28992")
pland=allgem_sum.boundary.plot(color='green',alpha=0.1)
cx.add_basemap(pland, source= prov0,crs=plot_crs)


# +
def mkgroei(dfin,idxjr):
    idxs=['GM_CODE','GM_NAAM']
    #select only summable fields
    dfsum=dfin.groupby(idxs+['jaar']).agg('sum')
    dfidx=dfsum[[]].reset_index()
    dff=dfsum.reset_index()
    refjr=(dff[dff['jaar']==idxjr] ).drop(columns=['jaar'])
    #print(refjr)
    rv= dfidx.merge(refjr,how='left').set_index(idxs+['jaar'])
    rv = dfsum / rv
    return rv.reset_index()
    
selrgroei=mkgroei(allgem_sum,2022)
# -

sns.lineplot(data=selrgroei.reset_index(),x='jaar',y='AANT_INW',style='GM_CODE')

ODiN2readpkl.allodinyr.dtypes


def chklevstat(df,grpby,dbk,vnamcol,myNiveau):
    chkcols = dbk [ (dbk.Niveau == myNiveau) & ~ ( dbk[vnamcol].isin( excols) )]
    for chkcol in chkcols[vnamcol]:
        nonadf= df[~ ( df[chkcol].isna() |  df[grpby].isna() ) ]
#        print (chkcol)
#        print (nonadf['RitID'])
#        sdvals= nonadf. groupby([grpby]) [chkcol].std_zero()
        sdvals= nonadf. groupby([grpby]) [chkcol].agg(['min','max']).reset_index()
#        print(sdvals)
        sdrng = (sdvals.iloc[:,2] != sdvals.iloc[:,1]).replace({True: 1, False: 0})
#        print(sdrng)
        maxsd=sdrng.max()
        #als alle standaard deviaties 0 zijm, zijn de waarden homogeen in de groep
        if maxsd !=0:
            print (chkcol,maxsd)


dbk_2022 = ODiN2readpkl.dbk_allyr
dbk_2022_cols = dbk_2022 [~ dbk_2022.Variabele_naam_ODiN_2022.isna()]
dbk_2022_cols [ dbk_2022_cols.Niveau.isna()]

# +
largranval =ODiN2readpkl.largranval 

specvaltab = ODiN2readpkl.mkspecvaltab(dbk_2022)
#specvaltab
# -

allodinyr=ODiN2readpkl.allodinyr
len(allodinyr.index)

# +
isnhexpl = {7:"rondje huis",6: "naar huis" , 5:"van huis",4:"ronde elders" }

allodinyr['isnaarhuis'] =  np.where(allodinyr ['Doel'] ==1, 
                    np.where(allodinyr ['VertLoc']<=2 , 7,6),
                    np.where(allodinyr ['VertLoc']<=2 , 5,4 ) ) 
allodinyr['isnaarhuis_expl'] =  allodinyr['isnaarhuis'].astype(str) + " "+ allodinyr['isnaarhuis'].map(isnhexpl)
allodinyr['FactorKm']= allodinyr['FactorV'] * allodinyr['AfstV'] *0.1
def setamodes(pstatsa):
    amodes= [5,6]
    pstatsa['FactorKmActive']= np.where(np.isin(pstatsa['KHvm'], amodes ),pstatsa['FactorKm']  ,0)
    pstatsa['FactorVActive']= np.where(np.isin(pstatsa['KHvm'], amodes ),pstatsa['FactorV']  ,0)
setamodes(allodinyr)


# -

def addgrpexpl (pstatsn,myspecvals,pltgrp,ext=""):
    grplrv = len(myspecvals [ (myspecvals ['Code'] ==largranval) & 
                            (myspecvals ['Variabele_naam'] ==pltgrp) ] ) !=0    
    if not grplrv:
        explhere = myspecvals [myspecvals['Variabele_naam'] == pltgrp].copy()
        if (len(explhere) >1 ):
            explhere['Code'] = pd.to_numeric(explhere['Code'],errors='coerce')
            explheres= explhere.set_index('Code').to_dict()['Code_label'] 
    #   print(explhere)            
            pstatsn[pltgrp+ext] = pstatsn[pltgrp].astype(str)  + " : " + \
                 (pstatsn[pltgrp].map(explheres) )
    return pstatsn


keepexplclasses=['KHvm','MotiefV','KAfstV' ]
def addexp(df, lst):
    for c in lst:
        addgrpexpl (df,specvaltab, c,ext="_expl" )
    return [c+"_expl" for c in lst]
keepclasses=['Jaar','AankUur','VertUur','isnaarhuis','isnaarhuis_expl']
kflgsflds=['FactorV',"FactorKm","FactorKmActive","FactorVActive"]
keepexplcs=addexp(allodinyr,keepexplclasses)

allodinyr.columns

# +
gemeentefields= ['WoGem' , 'VertGem', 'AankGem' ]
def maskgems(df,lst,keepval, onbval):
    for c in lst:
        df[c].mask(df[c]!=keepval, onbval,inplace=True)

odindatamask=ODiN2readpkl.allodinyr.copy(deep=True)
maskgems(odindatamask,gemeentefields,targgem,9999)
odindatamask.groupby(gemeentefields)['FactorV'].agg('sum')
# -

gfields=gemeentefields+keepclasses+keepexplclasses+keepeplcs
summ1gemdata=odindatamask.groupby(gfields)[kflgsflds].agg('sum').reset_index()


# +
#summ1gemdata

# +
def addverplricht(df,keepval, onbval):
    sep=10000
    ridict= { keepval*sep+keepval : 'binnen',keepval*sep+onbval : 'uit',
              onbval*sep+keepval : 'in',onbval*sep+onbval : 'buiten'  }
    df['verplricht'] = (df['VertGem']*sep+df['AankGem']).map(ridict)
    
addverplricht(summ1gemdata,targgem,9999)    
# -

rscalet={'in':1,'uit':1, 'binnen': 0.5, 'buiten' : 20000000 / 18000000000}
def pltjr4gr(dat,txt,rscale):
    inuittot=summ1gemdata.groupby(['verplricht','Jaar'])['FactorV'].agg('sum').reset_index()
    inuittot['FactorVs'] = inuittot['FactorV'] * (inuittot['verplricht'].map(rscale))
    #display(inuittot)
    p=sns.lineplot(data=inuittot,x='Jaar',y='FactorVs',hue='verplricht')
    p.set_title(txt)
pltjr4gr(summ1gemdata,'totaal aantal verplaatsingen',rscalet)

rscalew={'in':1/365,'uit':1/365, 'binnen': 0.5/365, 'buiten' : 20000000 / 18000000000/365}
#auto bestuurders
pltjr4gr(summ1gemdata[summ1gemdata['KHvm']==1],
         'aantal verplaatsingen als auto bestuurder per dag',         rscalew)

rscalea={'in':1/365,'uit':1/365, 'binnen': 0.05/365, 'buiten' : 5000000 / 18000000000/365}
def pltjr4gra(dat,txt,rscale):
    inuittot=dat.groupby(['verplricht','Jaar'])['FactorVActive'].agg('sum').reset_index()
    inuittot['FactorVs'] = inuittot['FactorVActive'] * (inuittot['verplricht'].map(rscale))
    #display(inuittot)
    p=sns.lineplot(data=inuittot,x='Jaar',y='FactorVs',hue='verplricht')
    p.set_title(txt)
pltjr4gra(summ1gemdata,'totaal aantal verplaatsingen actieve modes',rscalea)    

# +
gemtxt={targgem:'eigen',9999:'rest_nl',19999:'bezoeker'}
def __pltjr3gms(dat,field,txt,ri):    
        dat['gc']= dat[gemeentefields].sum(axis=1)== len(gemeentefields) *19999 
        #print(dat['gc'])
        dat['gc'] = np.where(dat['gc'] ,9998,dat[ri])
        inuittot=dat.groupby(['gc','Jaar'])[['FactorV',field]].agg('sum').reset_index()
        inuittot['FactorVr'] = inuittot[field] /inuittot['FactorV'] 
        inuittot['gemexpl'] = ri+ " = "+(inuittot['gc'].map(gemtxt) )
        return inuittot

def pltjr3gms(dat,field,txt,ris):
    inuittot= pd.concat([ __pltjr3gms(dat,field,txt,ri) for ri in ris])
    p=sns.lineplot(data=inuittot,x='Jaar',y='FactorVr',hue='gemexpl')
    p.set_title(txt)
pltjr3gms(summ1gemdata,'FactorVActive','deel van ritten Actieve modes',['VertGem','AankGem','WoGem'])  
# -

pltjr3gms(summ1gemdata,'FactorKm','gemiddelde afstanden',['VertGem','AankGem','WoGem'])  

# +
#print(allodinyr2)
#allodinyr = allodinyr2
# -


