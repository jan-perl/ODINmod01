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
targgemcode = 'GM%04.0f'%targgem
targpc4=range(3990,4000)
targgemcode


def getgemyrs():
    gems1= [ ODiN2readpkl.getgwb(year)[0].assign(jaar=year) for year in range(2020,2026)]
    print ([len (d) for d in gems1])
    rv=pd.concat(gems1)
    rv=geopandas.GeoDataFrame(rv, geometry=rv['geometry'])
#    rv= rv.mask(rv==-99999999.0, np.nan)
    return rv
allgem=getgemyrs()

#tabel regio
regtab=allgem[(allgem['GM_CODE']>"GM0305") & 
              (allgem['GM_CODE']<"GM0357") & (allgem['jaar']==2020)]
regtab[["GM_CODE","GM_NAAM","jaar","H2O"]].reset_index()


def selgemyrs(iv,gemcode):
    mv=iv[iv['GM_CODE'] == gemcode]
    sv=iv[(iv['H2O']=='NEE' ) | (iv['GM_CODE'] != gemcode) ].groupby(['jaar']).agg('sum').reset_index()
    for c in ['GM_CODE','GM_NAAM']:
        sv[c]="rest_NL"
    rv=mv.append(sv)
    rv=rv.copy().reset_index()
    rv.to_pickle("../intermediate/gem1sum_"+gemcode+".pkl")    
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


keepexplclasses=['KHvm','MotiefV','KAfstV' ,'Weekdag']
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

gfields=gemeentefields+keepclasses+keepexplclasses+keepexplcs
summ1gemdata=odindatamask.groupby(gfields)[kflgsflds].agg('sum').reset_index()


# +
def addverplricht(df,keepval, onbval):
    sep=10000
    ridict= { keepval*sep+keepval : 'binnen',keepval*sep+onbval : 'uit',
              onbval*sep+keepval : 'in',onbval*sep+onbval : 'buiten'  }
    df['verplricht'] = (df['VertGem']*sep+df['AankGem']).map(ridict)
    
#addverplricht(summ1gemdata,targgem,9999)    


# -

def mkodgemsum(indf,selgem):
    gemcode = 'GM%04.0f'%selgem
    odindatamask=indf.copy(deep=True)
    maskgems(odindatamask,gemeentefields,selgem,9999)
    odindatamask.groupby(gemeentefields)['FactorV'].agg('sum')
    gfields=gemeentefields+keepclasses+keepexplclasses+keepexplcs
    rv=odindatamask.groupby(gfields)[kflgsflds].agg('sum').reset_index()
    addverplricht(rv,selgem,9999)    
    rv.to_pickle("../intermediate/gem1odin_"+gemcode+".pkl")
    return rv
summ1gemdata=  mkodgemsum(ODiN2readpkl.allodinyr,targgem)

  

rscalea={'in':1/365,'uit':1/365, 'binnen': 1/365, 'buiten' : 20000000 / 18000000000/365}
def pltjr4gra(dat,xfield,field,txt,rscale,normfactorV):
    fieldexpl= {"FactorVActive":{False:"aantal loop+fiets (buiten rel)",True:"deel loop+fiets"},
                "FactorV":{False:"aantal ritten (buiten rel)",True:"een"},
                "FactorKm":{False:"totale reisafstand (km) (buiten rel)",True:"gemiddelde afstand (km)"}
               }
    #['VertGem','AankGem','WoGem']
    gemfield=min(dat['VertGem'])
    addfv=[] if (field=='FactorV') or not normfactorV else ['FactorV']
    inuittot=dat.groupby(['verplricht',xfield])[[field]+addfv].agg('sum').reset_index()
    
    if normfactorV:
        inuittot['FactorVs'] = inuittot[field] / inuittot['FactorV'] 
    else:
        inuittot['FactorVs'] = inuittot[field] * (inuittot['verplricht'].map(rscale)) 
    #display(inuittot)
    inuittot['richting'] = inuittot['verplricht'] + " " + ("%.0f" %gemfield)
    p=sns.lineplot(data=inuittot,x=xfield,y='FactorVs',hue='richting')
    if normfactorV & (field =="FactorVActive"):
        p.set_ylim(bottom=0,top=1)
    else:
        p.set_ylim(bottom=0)
    ylab=fieldexpl[field][normfactorV]
    p.set_ylabel(ylab)     
    p.set_title(txt)
pltjr4gra(summ1gemdata,"Jaar",'FactorVActive','totaal aantal verplaatsingen actieve modes',
          rscalea,False)    

# +
#summ1gemdata
# -

rscalet={'in':1,'uit':1, 'binnen': 1, 'buiten' : 20000000 / 18000000000}
def pltjr4gr(dat,txt,rscale):
    pltjr4gra(dat,'FactorV',txt,rscale,False)
pltjr4gra(summ1gemdata,'Jaar','FactorV','verplaatsingen per jaar ODIN',rscalet,False)

rscalew={'in':1/365,'uit':1/365, 'binnen': 1/365, 'buiten' : 20000000 / 18000000000/365}
#auto bestuurders
pltjr4gra(summ1gemdata[summ1gemdata['KHvm']==1],'Jaar','FactorV',
         'auto bestuurders per dag Houten',         rscalea,False)

pltjr4gra(summ1gemdata,'Jaar','FactorVActive','actieve modes Houten',rscalea,True)    

pltjr4gra(summ1gemdata,'VertUur','FactorVActive','actieve modes Houten',rscalea,True)    

pltjr4gra(summ1gemdata[summ1gemdata['KHvm']==1],'Jaar','FactorKm',
         'Veplaatsingafstand als auto bestuurder per dag',  rscalea,False)

pltjr4gra(summ1gemdata[summ1gemdata['KHvm']==1],'Jaar','FactorKm',
         'Gemiddelde afstand als auto bestuurder per dag',  rscalea,True)

summ1gemdata.groupby(['VertGem','AankGem','WoGem'] )[['FactorV']].agg('sum')


# +

def __pltjr3gms(dat,field,txt,ri):    
        gemtxt={9998:'rest_nl',9999:'bezoeker'}
        dat['gc']= dat[gemeentefields].sum(axis=1)== len(gemeentefields) *9999 
        #print(dat['gc'])
        dat['gc'] = np.where(dat['gc'] ,9998,dat[ri])
        inuittot=dat.groupby(['gc','Jaar'])[['FactorV',field]].agg('sum').reset_index()
        inuittot['FactorVr'] = inuittot[field] /inuittot['FactorV'] 
        inuittot['gemexpl'] = ri+ " = "+(inuittot['gc'].map(lambda x: gemtxt.get(x,'eigen')) )
        inuittot['opdel'] = ri
        return inuittot

def pltjr3gms(dat,field,txt,ris):
    inuittot= pd.concat([ __pltjr3gms(dat,field,txt,ri) for ri in ris])
    p=sns.lineplot(data=inuittot,x='Jaar',y='FactorVr',hue='gemexpl',alpha=0.5,style='opdel')
    p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    p.set_title(txt)
pltjr3gms(summ1gemdata,'FactorVActive','deel van ritten Actieve modes',['VertGem','AankGem','WoGem'])  
# -

pltjr3gms(summ1gemdata,'FactorKm','gemiddelde afstanden',['VertGem','AankGem','WoGem'])  

# +
#print(allodinyr2)
#allodinyr = allodinyr2
# -
allgem_sum352=selgemyrs(allgem,'GM0352')
allgem_sum352


summ1gemdata352=  mkodgemsum(ODiN2readpkl.allodinyr,352)

selrgroei352=mkgroei(allgem_sum352,2022)
sns.lineplot(data=selrgroei352.reset_index(),x='jaar',y='AANT_INW',style='GM_CODE')

pltjr4gra(summ1gemdata352,'Jaar','FactorV','totaal aantal verplaatsingen',rscalet,False)

print("klaar")


