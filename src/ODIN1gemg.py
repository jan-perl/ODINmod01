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

import math
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
allgemoog=getgemyrs()

if False:
    allgemog=ODiN2readpkl.getgemyrs(range(2020,2027),True) 
    #print(len(allgemo))
    allgemog

aant1819=pd.read_excel("../data/bevaant_201819.xlsx",skiprows=3)
aant1819

aant1819['H2O']="NEE"
aant1819=aant1819.rename(columns={'Jaar':'jaar'})
cods2020= allgemoog[(allgemoog['jaar']==2020) & (allgemoog['H2O']=='NEE')][["GM_CODE","GM_NAAM"]]
addl1819= aant1819.merge(cods2020,how='left')
addl1819=addl1819 [ False == ( addl1819['GM_CODE'].isna() | addl1819['AANT_INW'].isna() ) ]
addl1819['GM_CODE'].fillna("GM7000",inplace=True) 
addl1819

rv=pd.concat([allgemoog,addl1819])
allgemoo=geopandas.GeoDataFrame(rv, geometry=rv['geometry'])

inw_piv=allgemoo[allgemoo['H2O']=='NEE'].pivot(columns = 'jaar',index="GM_NAAM", values='AANT_INW')
inw_piv
#inw_reccnt=allgemoo.groupby('H2O','')

# +
#toevoegen wernemers banen
# -

dat_85481 = pd.read_csv('../data/CBS/85481NED/Observations.csv',sep=';')
dat_85481_mc = pd.read_csv('../data/CBS/85481NED/MeasureCodes.csv',sep=';')
dat_85481['Value'] = pd.to_numeric(dat_85481['Value'].str.replace(",","."))
#dat_85481['Value'] = dat_85481['Value'] *1000

dat_85481_mc

dat_8548nl=dat_85481[(dat_85481['WoonregioS']=="NL00" ) & 
                      ((dat_85481['WerkregioS'].str[0:2])=="GM" )  ].merge(dat_85481_mc, 
            left_on='Measure',right_on='Identifier')
dat_8548nl['jaar']=dat_8548nl['Perioden'].str[0:4].astype('int')
dat_8548nltb=dat_8548nl.pivot_table(columns='Title',
                              index=['WerkregioS','jaar'],values='Value').reset_index()

dat_8548nltb

allgem=allgemoo.merge(dat_8548nltb,left_on=['GM_CODE','jaar'],right_on=['WerkregioS','jaar'],
                     how='left').drop(columns=['WerkregioS'])

# +
#for debugging allgem=allgemoo
# -

#tabel regio
regtab=allgem[(allgem['GM_CODE']>"GM0305") & 
              (allgem['GM_CODE']<"GM0357") & (allgem['jaar']==2020)]
regtab[["GM_CODE","GM_NAAM","jaar","H2O","STED","AANT_INW"]].reset_index()

addl=["GM1904","GM0632","GM1581","GM0216","GM1961","GM1960","GM0214","GM0736"]
rs=regtab[(regtab['AANT_INW'] >20000 ) & (regtab['AANT_INW'] <100000 )]
rbm=(list(rs['GM_CODE'].unique()) ) +addl
print ( len(regtab[regtab['GM_CODE'].isin(rbm)])  )
rbm

rbmlrg=list (regtab[(regtab['AANT_INW'] >100000 ) & (regtab['AANT_INW'] <10000000 )]['GM_CODE'].unique() )
rbmlrg

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


# +
#en nu netjes, met schaal in km
def plaxkm(x, pos=None):
      return '%.0f'%(x/1000.)

def addbasemkmsch(ax,mapsrc):
    cx.add_basemap(ax,source= mapsrc,crs="epsg:28992")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plaxkm))


# -

rbmgeo=allgem[(allgem['GM_CODE'].isin(rbm+rbmlrg) ) & (allgem['jaar']==2024 )
             & (allgem['H2O']=='NEE' )].copy(deep=False)
rbmgeo['center']= rbmgeo.representative_point()
fig, axs = plt.subplots(1, 1,figsize=(14,10))    
plot_crs=3857
plot_crs="epsg:28992"
rbmgeo.set_crs(crs="epsg:28992")
pland=rbmgeo.plot(ax=axs,alpha=1,column='AANT_INW',legend=True, cmap='OrRd') 
#             legend_kwds={"label": "Aantal_inwoners"})
#cbar=pland.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,alpha=0.3)
plandb=rbmgeo.boundary.plot(ax=axs,color='green',alpha=0.3)
for index, row in rbmgeo.iterrows(): 
    axs.text(row['center'].x,row['center'].y,row['GM_NAAM'], ha='center', va='center',
             alpha=0.5,size =10)
#was cx.add_basemap(pland, source= prov0,crs=plot_crs)    
addbasemkmsch (axs, prov0)
figname = "../output/reg_gempop_"+'utr'+"_"+'m1.svg';
fig.savefig(figname, dpi=300, bbox_inches="tight")



#tabel regio
regtab2=allgem[(allgem['GM_NAAM'].str.contains("Vene" ) ) & 
                (allgem['jaar']==2020)]
regtab2[["GM_CODE","GM_NAAM","jaar","H2O","AANT_INW"]].reset_index()


def selgemyrs(iv,gemcode):
    mv=iv[iv['GM_CODE'] == gemcode]
    sv=iv[(iv['H2O']=='NEE' ) | (iv['GM_CODE'] != gemcode) ].groupby(['jaar']).agg('sum').reset_index()
    for c in ['GM_CODE','GM_NAAM']:
        sv[c]="rest_NL"
    rv=mv.append(sv)
    rv=rv.copy().reset_index()
    rv.to_pickle("../intermediate/gemdata/gem1sum_"+gemcode+".pkl")    
    return rv
allgem_sum=selgemyrs(allgem,targgemcode)
allgem_sum

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
    rv=rv.where(rv!=0,np.NaN)
    return rv.reset_index()
    
selrgroei=mkgroei(allgem_sum,2022)
# -

selrgroei

#echte functie in ODIN1gemvis
sns.lineplot(data=selrgroei.reset_index(),x='jaar',y='AANT_INW',style='GM_CODE',
             color='blue',marker='o',label='inwoners')
sns.lineplot(data=selrgroei.reset_index(),x='jaar',y='Banen van werknemers',
             style='GM_CODE',color='green',marker='x',label='banen werknemers')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



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
            pstatsn[pltgrp+ext] = (pstatsn[pltgrp].apply(lambda x: "%4.0f : "%x) )  + \
                 (pstatsn[pltgrp].map(explheres) ) 
    return pstatsn


keepexplclasses=['KHvm','MotiefV','KAfstV' ,'Weekdag']
def addexp(df, lst):
    for c in lst:
        addgrpexpl (df,specvaltab, c,ext="_expl" )
    return [c+"_expl" for c in lst]
keepclasses=['Jaar','AankUur','VertUur','isnaarhuis','isnaarhuis_expl']
kflgsflds=['Nwaarn','FactorV',"FactorKm","FactorKmActive","FactorVActive"]
keepexplcs=addexp(allodinyr,keepexplclasses)

# +
#allodinyr['KAfstV_expl']
# -

allodinyr['Nwaarn']=1
allodinyr.columns

allodinyr[['OP','FactorV']]

# +
ODINgemeentefields= ['WoGem' , 'VertGem', 'AankGem' ]
def maskgems(df,lst,keepval, onbval):
    for c in lst:
        df[c].mask(df[c]!=keepval, onbval,inplace=True)

odindatamask=ODiN2readpkl.allodinyr.copy(deep=True)
maskgems(odindatamask,ODINgemeentefields,targgem,9999)
odindatamask.groupby(ODINgemeentefields)['FactorV'].agg('sum')
# -

gfields=ODINgemeentefields+keepclasses+keepexplclasses+keepexplcs
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
    maskgems(odindatamask,ODINgemeentefields,selgem,9999)
    odindatamask.groupby(ODINgemeentefields)['FactorV'].agg('sum')
    gfields=ODINgemeentefields+keepclasses+keepexplclasses+keepexplcs
    rv=odindatamask.groupby(gfields)[kflgsflds].agg('sum').reset_index()
    addverplricht(rv,selgem,9999)    
    rv.to_pickle("../intermediate/gemdata/gem1odin_"+gemcode+".pkl")
    return rv
summ1gemdata=  mkodgemsum(ODiN2readpkl.allodinyr,targgem)


#echte functie in ODIN1gemvis
#bekijk statistiek per jaar
def mksampletab(dat, fieldsplit):
    dat2=dat.copy().rename(columns={fieldsplit:'GM_CODE'})
#    print(dat2)
    repflds=['Nwaarn','FactorV','FactorKm']
    rt=dat2.groupby(['GM_CODE','Jaar'])[repflds].agg('sum').reset_index()
    rt['gemwgtdag'] = rt['FactorV'] / rt['Nwaarn'] /365
    rt['gemafst'] = rt['FactorKm'] / rt['FactorV']
    return rt.assign(opdeling=fieldsplit)
def mksamplegopd(dat):
    t2 = [ mksampletab(dat, gf) for gf  in ODINgemeentefields ] 
    rv = pd.concat(t2).reset_index()
    return rv
sampletab= mksamplegopd(summ1gemdata)
sampletab

sns.lineplot(data=sampletab,x='Jaar',y='gemwgtdag',hue='opdeling',style='GM_CODE', marker= 'o')

sns.lineplot(data=sampletab,x='Jaar',y='gemafst',hue='opdeling',style='GM_CODE', marker= 'o')

summ1gemdata.groupby('KHvm_expl')['FactorV'].agg('sum')


# +
#wat betekent dit voor modale totalen ?
# -

def modplotopd(dat, fieldsplit,selgem,valfield,grpfield):
    dsel= dat[dat [fieldsplit] < 9000] 
    dagg = dsel.groupby (['Jaar' ,grpfield] )[[valfield]].agg('sum')
    dagg = dagg*7/365
    dagg= dagg.reset_index().sort_values(grpfield)
    dagg[valfield]=dagg.groupby(['Jaar'])[valfield].cumsum()
    hues=dagg[grpfield].unique()
    sns.barplot(data=dagg,x='Jaar', y= valfield , hue=grpfield,dodge=0,hue_order=hues[::-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Opdeling is '+fieldsplit)
    #return daggcum
modplotopd(summ1gemdata,'VertGem',321,'FactorV','KHvm_expl')

modplotopd(summ1gemdata,'AankGem',321,'FactorV','KHvm_expl')

#Meeste kms auto en OV
modplotopd(summ1gemdata,'VertGem',321,'FactorKm','KHvm_expl')

# +
#pendel deel in ODIN1gemVis programma
# -





# +
#check opdeling: in / uit /binnen/buiten ; alle dagen van de week
# -

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

# +
#nu wat relatieve grafieken; niet zo nuttig want deze groepen vertekenen ze
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
        dat['gc']= dat[ODINgemeentefields].sum(axis=1)== len(ODINgemeentefields) *9999 
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
#nu wat algemenere kentallen over inter-gemeentelijk verkeer
# -

def prepigem(dfin):
    df=dfin[dfin['VertGem']!=dfin['AankGem'] ].copy()
    df['WaarWoon']='elders'
    df['WaarWoon']=df['WaarWoon'].where(df['VertGem']!=df['WoGem'],'Vert');
    df['WaarWoon']=df['WaarWoon'].where(df['AankGem']!=df['WoGem'],'Aank');
    df['Werkdag']=df['Weekdag'].isin([2,3,4,5,6]) 
    return df
igin=prepigem(ODiN2readpkl.allodinyr)
#igin[['VertGem','AankGem','WoGem','WaarWoon']]

iginuurt1=igin.groupby(['WaarWoon','VertUur','Werkdag'])[['FactorV']].agg('sum').reset_index()
g = sns.FacetGrid(iginuurt1, col="Werkdag",hue='WaarWoon')
g.map(sns.lineplot ,'VertUur','FactorV',marker='o')



def igipergem(dfin,gdbin,jrs):
    dfs= dfin[(dfin['Jaar'].isin(jrs)) & (dfin['KHvm']==1)]
    dfp = [ dfs.rename(columns={gg:'Gem'}).groupby(['Gem','Jaar'])[['FactorV']].\
                        agg('sum').reset_index().assign(GrpGem=gg) \
           for gg in ['VertGem', 'AankGem' ] ]
    df=pd.concat(dfp)
    df ['GM_CODE'] = df ['Gem'] .apply(lambda x: 'GM%04.0f'%(x))
    gdb=gdbin[ (gdbin['jaar'].isin(jrs)) & (gdbin['H2O']=='NEE') ].rename(columns={'jaar':'Jaar'})
#    print(gdb)
    df = df.merge(gdb ,how='left')
    df['Widx'] = 2*df['FactorV']*(7/365)/df['AANT_INW']
    return df
igingt=igipergem(igin   ,allgem ,[2023,2024])
igingt[igingt['GM_CODE']=='GM0321']

igingtr=igingt[(igingt['AANT_INW']<100000 ) & (igingt['AANT_INW']>20000 )]
sns.lineplot(data=igingtr,x='AANT_INW',y='Widx',hue='GrpGem')

igingtr=igingt[(igingt['AANT_INW']<54000 ) & (igingt['AANT_INW']>46000 ) 
               & (igingt['STED']==2)]
sns.lineplot(data=igingtr,x='AANT_INW',y='Widx',hue='GrpGem')
klasgem=igingtr['GM_CODE'].unique()

igingtr



def mnstdgrps(df,gr,fld):
    rv=igingtr.groupby(gr)[fld].agg(['mean','std']).reset_index()
    rv.columns = ['_'.join(col) if isinstance (col,tuple) else col for col in rv.columns]
    print (rv.columns)
    return rv
igingtrs = mnstdgrps(igingtr,['GM_CODE'],['Widx','AANT_INW'])
sns.scatterplot(data=igingtrs,x='Widx_mean' , y='Widx_std')

#de hoogste variatie neemt af met aantallen inwoners, maar blijft rond de 1
sns.scatterplot(data=igingtrs,x='AANT_INW_mean' , y='Widx_std')


# +
#nu database bouwen voor alle gemeenten
#pas later selecteren op grootte
# -

def odinyrallgem(indf):
    map1={True:'binnen',False:'in'}
    dd1= indf.copy(deep=False)
    dd1['Gem']=dd1['AankGem']
    dd1['verplricht'] = (dd1['VertGem']==dd1['Gem']) .map(map1)
    map2={True:'binnen',False:'uit'}
    dd2= indf.copy(deep=False)
    dd2['Gem']=dd2['VertGem']
    dd2['verplricht'] = (dd2['AankGem']==dd2['Gem']) .map(map2)
    rv= pd.concat([dd1,dd2])
    mapw={True:'inwoner',False:'niet-inwoner'}
    rv['WoGemCat']= (rv['WoGem']==rv['Gem']) .map(mapw)
    rv['GM_CODE'] = rv ['Gem'] .apply(lambda x: 'GM%04.0f'%(x))
    return rv
summ1gemdbl=  odinyrallgem(ODiN2readpkl.allodinyr)
summ1gemdbl

ochturen=(5,6,7,8,9)
miduren=(15,16,17,18,19)

# +
#onderstaande functie 1-op-1 over
lokaallabel='lokaal'
def pendelcODIN(gemdat):
    df= gemdat[(gemdat['verplricht'] !='buiten')].copy(deep=False);
    rscalea={'in':1/365,'uit':1/365, 'binnen': 1/365, 'buiten' : 20000000 / 18000000000/365}
    #gemfield=min(df['VertGem'])
    df['Uur'] = df['AankUur']
    df['Uur'] .where(False== (df['verplricht']=="uit"), df['VertUur'],inplace=True )
    pccodes= {0:'buitensp',1: 'bezoekersp',2: 'bewonersp',3:'weekendp',4:lokaallabel,5:'niet-gem' }
    df['pendelcati'] = 0
    pci= df['pendelcati']
    pci.where(False== ((df['AankUur'].isin(ochturen) ) & (df['verplricht']=="in")),1,inplace=True )
    pci.where(False==( (df['VertUur'].isin(ochturen) ) & (df['verplricht']=="uit")),2,inplace=True )
    pci.where(False== ((df['VertUur'].isin(miduren) ) & (df['verplricht']=="uit")),1,inplace=True )
    pci .where(False==( (df['AankUur'].isin(miduren) ) & (df['verplricht']=="in")),2,inplace=True )
    pci .where(df['Weekdag'].isin([2,3,4,5,6]) ,3,inplace=True )
    pci.where(False==( df['verplricht'].isin( ["binnen"]  )),4,inplace=True )
    pci.where(False==( df['verplricht'].isin( ["buiten"]  )),5,inplace=True )
    df['pendelcat']=pci.map(pccodes)
    return df

allvgrp=pendelcODIN(summ1gemdbl) 
allvtot=allvgrp.groupby(['pendelcat','Jaar'])['FactorV'].agg('sum').reset_index()
#allvtot
# -

sns.lineplot(data=allvtot,x='Jaar',y='FactorV',hue='pendelcat')



pc2htn=allvgrp[allvgrp['Gem']==321]

pc2htn


# +
def datplotcumcatwo(dat, fieldsplit,valfield,mult):
    dagg = dat.groupby (['pendelcat',fieldsplit] )[[valfield]].agg('sum')
    dagg = dagg*mult
    dagg= dagg.reset_index().sort_values(fieldsplit)
    dagg[valfield]=dagg.groupby(['pendelcat'])[valfield].cumsum()
    dagg[valfield] = dagg[valfield].where(dagg['pendelcat']  !=lokaallabel,0.5 * dagg[valfield])
    dagg['opdeling'] =fieldsplit
    return dagg
    
def modplotopdcatwo(fig,ax,dat, fieldsplit,title,valfield,jaarnorm):
    mult=7/365;
    if jaarnorm:
        jaren=dat['Jaar'].unique()
        mult /= len(jaren)
    dagg=datplotcumcatwo(dat, fieldsplit,valfield,mult ) 
    dagg=dagg.sort_values([fieldsplit,'pendelcat'],ascending=[True,True])
    
#    dagg['Jaar'] += dagg['opdeling'] .map(fieldsplit)

    hues=dagg[fieldsplit].unique()
#    print(hues[::-1])
    sns.barplot(ax=ax,data=dagg,x='pendelcat', y= valfield , 
                hue=fieldsplit,dodge=0,hue_order=hues[::-1])
    leglabels= {'WoGemCat':'Woongemeente','MotiefV_expl':'Reismotief',
                'GM_CODE':'Gemeente',
                'KHvm_expl' : 'Hoofdvervoermiddel'}
    ax.legend(title=leglabels[fieldsplit], bbox_to_anchor=(1.01, 0.95), loc=2, borderaxespad=0.,framealpha=0)
    ax.set_title(title )
    #return daggcum
fig, axs = plt.subplots(1, 1)    
modplotopdcatwo(fig,axs,pc2htn,'WoGemCat', 'Alle verplaatsingen per week','FactorV',True)
# -
fig, axs = plt.subplots(1, 1)    
pc2htni= pc2htn[pc2htn['pendelcat'] !=lokaallabel]
modplotopdcatwo(fig,axs,pc2htni,'MotiefV_expl', 'Alle verplaatsingen per dag','FactorV',True)

modplotopd(pc2htn,'VertGem',321,'FactorV','KHvm_expl')

modplotopd(pc2htn,'VertGem',321,'FactorV','pendelcat')


def mkautointergem(indb):
    return indb [(indb ['KHvm'] ==1 ) & (indb ['pendelcat'] !=lokaallabel ) ] .copy(deep=False)
autointergemrecs=mkautointergem(allvgrp)


def mkautoall(indb):
    return indb [(indb ['KHvm'] ==1 )  ] .copy(deep=False)
autoallrecs=mkautoall(allvgrp)

autointergemrecs

allgem


def modplotopdfacet(aig,bm,usegem, savnam,facets,valfield,grpfield):
    dsel=aig [(aig['GM_CODE'].isin(bm)) & (aig['Jaar']>=2018)]
    d2 = dsel.groupby (['Jaar' ,grpfield,'GM_CODE'] )[[valfield]].agg('sum').reset_index()
    allgemn= usegem[usegem['H2O']=='NEE']
    dagg=d2.merge(allgemn.rename(columns={'jaar':'Jaar'}),how='left' )
    dagg[valfield] = dagg[valfield]*7/365
    dagg= dagg.reset_index().sort_values(grpfield)
    dagg[valfield]=dagg.groupby(['Jaar',facets])[valfield].cumsum()
    dagg[valfield]/=dagg['AANT_INW']
    print(dagg[['Jaar',valfield,'GM_CODE','GM_NAAM','AANT_INW']])
    #print(dagg[['Jaar' ,grpfield,'GM_CODE','GM_NAAM',valfield,'AANT_INW']])
    hues=dagg[grpfield].unique()
    gemslen=len(bm)
    gemslensqrt = max(3,int(math.sqrt(gemslen)+1))
    #print ((gemslensqrt,gemslen))
    #print(hues)
    g=sns.FacetGrid(dagg, col=facets,col_wrap=gemslensqrt, hue=grpfield,hue_order=hues[::-1] )
    #g.map_dataframe(sns.lineplot, x='Jaar', y=valfield )
    #g.map_dataframe(sns.lineplot, x='Jaar', y=valfield )
    g.map(sns.barplot, 'Jaar', valfield )
    #,dodge=0,               hue_order=hues[::-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figname = "../output/reg_jrcmp_"+savnam+"_"+'m1.svg';
    g.savefig(figname, dpi=300, bbox_inches="tight")
    #plt.title('Opdeling is '+fieldsplit)
    #return daggcum
modplotopdfacet(allvgrp,rbmlrg,allgem,'rbmlrg-V','GM_NAAM','FactorV','KHvm_expl')

modplotopdfacet(allvgrp,rbm,allgem,'rbm-V','GM_NAAM','FactorV','KHvm_expl')

modplotopdfacet(autointergemrecs,rbm,allgem,'rbm-pcat1-V','GM_NAAM','FactorV','pendelcat')

autointergemrecsrbmtot= autointergemrecs[autointergemrecs['GM_CODE'].isin(rbm)] .copy()
rmb_rep_code="GM_RAND_20_100"
autointergemrecsrbmtot['GM_CODE']=rmb_rep_code
allgemrbmtot=allgem[allgem['GM_CODE'].isin(rbm)].groupby(['jaar','H2O'])[[
     'AANT_INW','Banen van werknemers']].agg('sum').reset_index()
allgemrbmtot['GM_CODE']=rmb_rep_code
allgemrbmtot['GM_NAAM']="Gemeenten 20-100k"
print(allgemrbmtot)
modplotopdfacet(autointergemrecsrbmtot,[rmb_rep_code],allgemrbmtot,'VertGem','GM_NAAM','FactorV','pendelcat')

modplotopdfacet(pd.concat([autointergemrecs,autointergemrecsrbmtot]),
        rbmlrg+[rmb_rep_code],pd.concat([allgem,allgemrbmtot]),
                'VertGem','GM_NAAM','FactorV','pendelcat')

rt2024=pd.concat([allgem[allgem['GM_CODE'].isin(rbmlrg)],allgemrbmtot])
rt2024[rt2024['jaar']==2024]

modplotopd(autointergemrecs [autointergemrecs['Gem'] ==321],'Gem',321,'FactorV','pendelcat')

modplotopd(autointergemrecs [autointergemrecs['Gem'] ==321],'Gem',321,'FactorV','MotiefV_expl')

fig, axs = plt.subplots(1, 1)    
modplotopdcatwo(fig,axs,autointergemrecs [autointergemrecs['Gem'] ==321]
                ,'MotiefV_expl', 'Alle verplaatsingen per dag','FactorV',True)

fig, axs = plt.subplots(1, 1)    
modplotopdcatwo(fig,axs,autointergemrecs [autointergemrecs['Gem'] ==321]
                ,'MotiefV_expl', 'Alle verplaatsingen per dag','FactorV',True)


def refjaarvarwidx(aig,bm,usegem,lbl,tit):
    fig, axs = plt.subplots(1, 1)    
    a2=aig [aig['GM_CODE'].isin(bm)]
    d2=a2.groupby(['Jaar','GM_CODE'])[['FactorV','Nwaarn']].agg('sum').reset_index()
    d2=d2.merge(usegem.rename(columns={'jaar':'Jaar'}),how='left' )
    d2['Widx'] = d2['FactorV']/d2['AANT_INW']*(7/365)
    d2['Nwid'] = d2['Widx']/np.sqrt(d2['Nwaarn'])
    sns.lineplot(ax=axs,data=d2,x='Jaar',y='Widx',hue='GM_NAAM',marker='o')
#    sns.scatterplot(ax=axs,data=d2,x='Jaar',y='Widx',hue='GM_NAAM',marker='o')
    axs.errorbar(x=d2['Jaar'],y=d2['Widx'],yerr=d2['Nwid'],fmt='none',color='grey')
    axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axs.set_title('jaarvariatie ODIN autoritten '+tit)
    axs.set_ylabel('autobewegingen van/naar gemeente / inw/week')
refjaarvarwidx(autointergemrecs,klasgem,allgem,'kg1','sted 2, 46-54 k inw')    

refjaarvarwidx(autointergemrecs,rbmlrg,allgem,'rbm01','Omgeving Utrecht 20k+ inw')

refjaarvarwidx(autointergemrecs,rbmlrg,allgem,'rbmlrg01','Gemeenten Utrecht 100k+ inw')


# +
def errorbar_plot_refjaarvarwidxfacet(x,y,yerr, **kwargs): 
    plt.errorbar( x=x,y=y,yerr=yerr,**kwargs ) 

def refjaarvarwidxfacet(aig,bm,usegem,opd2,lbl,tit):
#    fig, axs = plt.subplots(1, 1)    
    a2=aig [aig['GM_CODE'].isin(bm)]
    d2=a2.groupby(['Jaar','GM_CODE']+opd2)[['FactorV','Nwaarn']].agg('sum').reset_index()
    d2['H2O']='NEE'
    d2=d2.merge(usegem.rename(columns={'jaar':'Jaar'}),how='left' )
    d2['Widx'] = d2['FactorV']/d2['AANT_INW']*(7/365)
    d2['Nwid'] = d2['Widx']/np.sqrt(d2['Nwaarn']/2)
#    print(d2[['Jaar','Widx','Nwid','GM_CODE','GM_NAAM','AANT_INW']])
    gemslen=len(bm)
    gemslensqrt = max(3,int(math.sqrt(gemslen)+1))    

    if (len (opd2)==0):
        g=sns.FacetGrid(d2,col='GM_NAAM',col_wrap=gemslensqrt)
    else:
        g=sns.FacetGrid(d2,col='GM_NAAM',hue=opd2[0],col_wrap=gemslensqrt)
    g.map(errorbar_plot_refjaarvarwidxfacet,'Jaar','Widx','Nwid',marker="o", fmt='o-', capsize=4)
    if (len (opd2)!=0):
        g.add_legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout() 

refjaarvarwidxfacet(autointergemrecs,klasgem,allgem,['pendelcat'],'kg1','sted 2, 46-54 k inw',)    
# -

refjaarvarwidxfacet(autointergemrecs,rbm,allgem,[],'kg1','sted 2, 46-54 k inw')    

refjaarvarwidxfacet(pd.concat([autointergemrecs,autointergemrecsrbmtot]),
        rbmlrg+[rmb_rep_code], pd.concat([allgem,allgemrbmtot]), [],'rbm-v','rbm clustered')                    


# +
def refrichtvarwidx(aig,bm,usegem,gn,grpfield,valfield,lbl,tit):
    fig, axs = plt.subplots(1, 1)    
    a2=a2=aig [aig['GM_CODE'].isin(bm)]
    d2=a2.groupby(['Jaar',grpfield,'GM_CODE'])[[valfield]].agg('sum').reset_index()
    d2=d2.merge(usegem.rename(columns={'jaar':'Jaar'}),how='left' )
    d3=d2.groupby([grpfield,'GM_CODE','GM_NAAM'])[[valfield,'AANT_INW']].agg('sum').reset_index()
    valfieldn='Widx'
    d3['Widx'] = d3[valfield]/d3['AANT_INW']*(7/365)    
#    dagg = d3.groupby ([gn ,grpfield] )[[valfieldn]].agg('sum')
    dagg=d3
    dagg= dagg.reset_index().sort_values(grpfield)
    dagg[valfieldn]=dagg.groupby([gn])[valfieldn].cumsum()
    hues=dagg[grpfield].unique()
    sns.barplot(data=dagg,y=gn, x= valfieldn , hue=grpfield,dodge=0,
                hue_order=hues[::-1])    
    axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axs.set_title('groepering ODIN autoritten 2018-2024 '+tit)
    xlmap={'FactorV': 'autobewegingen van/naar gemeente / inw/week',
           'FactorKm': 'autokilometers van/naar gemeente / inw/week'}
    axs.set_xlabel(xlmap[valfield])
    axs.set_ylabel('Gemeente')

refrichtvarwidx(autointergemrecs,klasgem,allgem,'GM_NAAM','pendelcat','FactorV','kg1','sted 2, 46-54 k inw')    
# -

refrichtvarwidx(autointergemrecs,rbm,allgem,'GM_NAAM','pendelcat','FactorV','rmb01','Omgeving Utrecht 20-100 k inw')

refrichtvarwidx(autointergemrecs,rbm,allgem,'GM_NAAM','pendelcat','FactorKm','rbm01','Omgeving Utrecht 20-100 k inw')

refrichtvarwidx(autointergemrecs,rbm,allgem,'GM_NAAM','MotiefV_expl','FactorV','rbm01','Omgeving Utrecht 20-100 k inw')

refrichtvarwidx(autointergemrecs,rbm+rbmlrg,allgem,'GM_NAAM','WoGemCat','FactorV','rbm01','Omgeving Utrecht 20-100 k inw')

autoallrecs['inwofbinnen'] = autoallrecs['WoGemCat']  
autoallrecs['inwofbinnen'].where(autoallrecs['pendelcat']!=lokaallabel,'rit lokaal',inplace=True)
refrichtvarwidx(autoallrecs,rbm+rbmlrg,allgem,'GM_NAAM','inwofbinnen','FactorV','rbm01','Omgeving Utrecht 20-100 k inw')

refrichtvarwidx(autoallrecs,rbm+rbmlrg,allgem,'GM_NAAM','inwofbinnen','FactorKm','rbm01','Omgeving Utrecht 20k+ inw')

maphhg={1:"0-30%",2:"0-30%",3:"0-30%",4:"30-60%",5:"30-60%",6:"30-60%",
         7:"60-80%",8:"60-80%",9:"80-100%",10:"80-100%",11:"onbek"}
autointergemrecs['HHGestInkGR']= autointergemrecs['HHGestInkG'].map(maphhg)
refrichtvarwidx(autointergemrecs,rbm,allgem,'GM_NAAM','HHGestInkGR','FactorV','rmb01','Omgeving Utrecht 20-100 k inw')

print("klaar")


