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
import os as os
from sklearn.linear_model import LinearRegression

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

# +
#import RUDIbas
# -

myname='ODIN1gemvis'
suprtests=False# myname in RUDIbas.suprtests 
suprdata= False # myname in RUDIbas.suprdata
#suprtests=True
print ('Suprtestsvis',suprtests)

ODINgemeentefields= ['WoGem' , 'VertGem', 'AankGem' ]

odir="../output/gemvis01"
os.makedirs(odir, exist_ok=True)

#set Houten als target
targgem =321
targgemcode = 'GM%04.0f'%targgem
targpc4=range(3990,4000)
targgemcode


def selgemyrs(targgem):
    targgemcode = 'GM%04.0f'%targgem
    allgem_CBSsum=pd.read_pickle ("../intermediate/gemdata/gem1sum_"+targgemcode+".pkl")
    ODindta =pd.read_pickle ("../intermediate/gemdata/gem1odin_"+targgemcode+".pkl")
    rv =(allgem_CBSsum,ODindta)
    return rv
(allgem_sum,summ1gemdata)=selgemyrs(targgem)
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
    rv=rv.where(rv!=0,np.NaN)
    return rv.reset_index()
    
selrgroei=mkgroei(allgem_sum,2022)


# +
#selrgroei

# +
def pltecongroei(grtab,tit,pltpref):
    sns.lineplot(data=grtab.reset_index(),x='jaar',y='AANT_INW',style='GM_CODE',
                 color='blue',marker='o',label='inwoners')
    sns.lineplot(data=grtab.reset_index(),x='jaar',y='Banen van werknemers',
                 style='GM_CODE',color='green',marker='x',label='banen werknemers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Economische groei "+tit)
    figname = "../output/"+pltpref+"_econgr.svg";
    plt.savefig(figname,dpi=300, bbox_inches='tight')

pltecongroei(selrgroei, 'Houten' ,'htn')  


# -

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
#sampletab

sns.lineplot(data=sampletab,x='Jaar',y='gemwgtdag',hue='opdeling',style='GM_CODE', marker= 'o')

sns.lineplot(data=sampletab,x='Jaar',y='gemafst',hue='opdeling',style='GM_CODE', marker= 'o')

targgem


# +
def datplotcum(dat, fieldsplit,selgem,valfield):
    dsel= dat[dat [fieldsplit] == selgem] 
    dagg = dsel.groupby (['Jaar' ,'KHvm'] )[[valfield]].agg('sum')
    dagg = dagg*1/365
    dagg= dagg.reset_index().sort_values('KHvm')
    dagg[valfield]=dagg.groupby(['Jaar'])[valfield].cumsum()
    dagg['opdeling'] =fieldsplit
    return dagg
    
def modplotopd(dat, fieldsplit,selgem,valfield):
    d2=[datplotcum(dat, f2,selgem,valfield ) for f2 in fieldsplit.keys() ]    
    dagg=pd.concat(d2)
    dagg['Jaar'] += dagg['opdeling'] .map(fieldsplit)

    hues=dagg['KHvm'].unique()
    print(hues[::-1])
    sns.barplot(data=dagg,x='Jaar', y= valfield , hue='KHvm',dodge=0,hue_order=hues[::-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Opdeling is links Vertgem en rechts Aankgem' )
    #return daggcum
modplotopd(summ1gemdata,{'VertGem':0, 'AankGem':0.25 } ,targgem,'FactorV')
# -

modplotopd(summ1gemdata,{'VertGem':0, 'AankGem':0.25 } ,targgem,'FactorKm')

# +
#eens wat vergelijkings plaatjes maken; deze blijken grotendeels geen zin te hebben  
# -

fieldexpl= {"FactorVActive":{False:"aantal loop+fiets (buiten rel)",True:"deel loop+fiets"},
                "FactorV":{False:"aantal ritten (buiten rel)",True:"een"},
                "FactorKm":{False:"totale reisafstand (km) (buiten rel)",True:"gemiddelde afstand (km)"}
               }
rscalea={'in':1/365,'uit':1/365, 'binnen': 1/365, 'met' : 0.5/365,'buiten' : 20000000 / 18000000000/365}
def pltjr4gra(dat,xfield,field,txt,rscale,normfactorV):
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
#print(allodinyr2)
#allodinyr = allodinyr2
# +
#verder vereenvoudigen, naar reizen van/naar gemeente , rest , niet genormaliseerd
#idee: invloed van zowel inwoners als bezoekers, beiden te beinvloeden
#reizigers van/naar gemeente maken per definitie langere afstanden, dus niet te vergelijken op modal share
#maar: gemiddelde afstanden zouden wel beinvloebaar kunnen zijn met beleid
#let op: landelijk optellen over alle gemeenten zou tot 2* FactorV leiden, FactorV dus corrigeren

#inwoners , alle gemeenten vs inwoners deze gemeente, per inwoner (leeftijdsopbouw niet mee nemen)
#naddel: invloed bedrijvigheid in gemeente telt alleen voor eigen inwoners
#voordeel: dit zijn de kiezers
# -

def gemvncmpdf(df):
    rv=df.copy()
    rv['verplrichto'] = rv['verplricht']
    for t in fieldexpl.keys():
        rv[t+ "o"] = rv[t]
        rv[t] = rv[t].where(rv['verplrichto']!='binnen' ,2*rv[t] )        
    rv['verplricht'] = rv['verplricht'].where(rv['verplricht']=='buiten' ,'met' )        
    return rv
summ1gemvn= gemvncmpdf(summ1gemdata)
pltjr4gra(summ1gemvn[summ1gemvn['KHvm']==1],'Jaar','FactorV',
         'auto bestuurders per dag Houten',         rscalea,False)


# +
def distmodescmp1Dim(datin,fld1D,txt):
    dat=datin.copy()
    dat['allereizen']="alle"
    gemfield=min(dat['VertGem'])
#    inuittot['richting'] = inuittot['verplricht'] + " " + ("%.0f" %gemfield)

    dfr=dat.groupby(fld1D)['FactorVo',"FactorKmo"].agg('sum').rename(
           columns={"FactorVo":"FactorV","FactorKmo":"FactorKm"})
    dfr=dfr/dfr.sum()
    dfr=dfr.reset_index()
    dfr['set']="landelijk gemiddelde"
    #print(dfr)
    dg=dat[dat["verplricht"]=="met"]
    dfg=dg.groupby(fld1D)['FactorV',"FactorKm"].agg('sum')
    dfg=dfg/dfg.sum()
    dfg=dfg.reset_index()
    dfg['set']="met "+str(gemfield)

    dfgb=dg.groupby(fld1D)['FactorVo',"FactorKmo"].agg('sum').rename(
           columns={"FactorVo":"FactorV","FactorKmo":"FactorKm"})    
    dfgb=dfgb/dfgb.sum()
    dfgb=dfgb.reset_index()
    dfgb['set']="metFOUT "+str(gemfield)

    dw=dat[dat["WoGem"]==gemfield]
    dfw=dw.groupby(fld1D)['FactorVo',"FactorKmo"].agg('sum').rename(
           columns={"FactorVo":"FactorV","FactorKmo":"FactorKm"})
    dfw=dfw/dfw.sum()
    dfw=dfw.reset_index()
    dfw['set']="inwoner "+str(gemfield)    
    dfc=pd.concat([dfr,dfg,dfw])
#    fig, ax = plt.subplots(figsize=(6, 4))
    cat=sns.catplot(data=dfc,y=fld1D[0],x='FactorV',hue='set',kind='bar',legend_out=True,col=fld1D[1],col_wrap=3)
    figname="%s/dmcmp_%s_%s_%.0f.svg"%(odir,fld1D[0],fld1D[1],gemfield)
    cat.fig.subplots_adjust(top=0.92)
    cat.fig.suptitle(txt)    
    cat.set_xlabels('Fractie verplaatsingen')
    cat.fig.savefig(figname, bbox_inches="tight")
#    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    p.set_title(txt)    
    
distmodescmp1Dim(summ1gemvn,["KHvm_expl","MotiefV_expl"] ,"Beneden gemiddeld lopen en fietsen in Houten")    
# -

distmodescmp1Dim(summ1gemvn,["KHvm_expl","allereizen"],"Beneden gemiddeld lopen en fietsen in Houten")    

distmodescmp1Dim(summ1gemvn,["KAfstV_expl","allereizen"],"per afstand")    

distmodescmp1Dim(summ1gemvn,["KAfstV_expl","MotiefV_expl"],"per afstand")  


def pltverplfracs (pltdfvn,gtit):
    distmodescmp1Dim(pltdfvn,["KHvm_expl","MotiefV_expl"],"Modi per motief "+gtit)        
    distmodescmp1Dim(pltdfvn,["KHvm_expl","allereizen"],"Per modus "+gtit)        
    distmodescmp1Dim(pltdfvn,["KAfstV_expl","MotiefV_expl"],"Afstanden per motief "+gtit)        
    distmodescmp1Dim(pltdfvn,["KAfstV_expl","allereizen"],"Afstanden "+gtit)        
pltverplfracs (summ1gemvn,"Houten")



#wijk bij duurstede
targgem=352
(allgem_sumexp,summ1gemdataexp)=selgemyrs(targgem)


pland=allgem_sumexp.boundary.plot(color='green',alpha=0.1)
cx.add_basemap(pland, source= prov0,crs=plot_crs)

selrgroei=mkgroei(allgem_sumexp,2022)
pltecongroei(selrgroei, 'Wijk bij Duurstede' ,'wijk') 

# +
#ODIN sampling
# -

sampletab= mksamplegopd(summ1gemdataexp)
sns.lineplot(data=sampletab,x='Jaar',y='gemwgtdag',hue='opdeling',style='GM_CODE', marker= 'o')

modplotopd(summ1gemdataexp,{'VertGem':0, 'AankGem':0.25 } ,targgem,'FactorV')

modplotopd(summ1gemdataexp,{'VertGem':0, 'AankGem':0.25 } ,targgem,'FactorKm')

pltjr4gra(summ1gemdataexp,'Jaar','FactorV','totaal aantal verplaatsingen',rscalet,False)

summ1gemvnexp= gemvncmpdf(summ1gemdataexp)

distmodescmp1Dim(summ1gemvnexp,["KAfstV_expl","MotiefV_expl"],"per afstand")  

pltverplfracs (summ1gemvnexp,"Wijk bij Duurstede")





#Utrechtse Heuvelrug
targgem=1581
(allgem_sumexp,summ1gemdataexp)=selgemyrs(targgem)

pland=allgem_sumexp.boundary.plot(color='green',alpha=0.1)
#cx.add_basemap(pland, sourexpe= prov0,crs=plot_crs)

selrgroei=mkgroei(allgem_sumexp,2022)
pltecongroei(selrgroei, 'Utrechtse Heuvelrug' ,'hrg') 

pltjr4gra(summ1gemdataexp,'Jaar','FactorV','totaal aantal verplaatsingen',rscalet,False)

#IJsselstein
targgem=353
(allgem_sumexp,summ1gemdataexp)=selgemyrs(targgem)

pland=allgem_sumexp.boundary.plot(color='green',alpha=0.1)
cx.add_basemap(pland, source= prov0,crs=plot_crs)

selrgroei=mkgroei(allgem_sumexp,2022)
pltecongroei(selrgroei, 'Ijsselstein' ,'ijst') 

pltjr4gra(summ1gemdataexp,'Jaar','FactorV','totaal aantal verplaatsingen',rscalet,False)

#Culemborg
targgem=216
(allgem_sumexp,summ1gemdataexp)=selgemyrs(targgem)

pland=allgem_sumexp.boundary.plot(color='green',alpha=0.1)
cx.add_basemap(pland, source= prov0,crs=plot_crs)

selrgroei=mkgroei(allgem_sumexp,2022)
pltecongroei(selrgroei, 'Culemborg' ,'cl') 

# +
#ODIN sampling
# -

sampletab= mksamplegopd(summ1gemdataexp)
sns.lineplot(data=sampletab,x='Jaar',y='gemwgtdag',hue='opdeling',style='GM_CODE', marker= 'o')

modplotopd(summ1gemdataexp,{'VertGem':0, 'AankGem':0.25 } ,targgem,'FactorV')

modplotopd(summ1gemdataexp,{'VertGem':0, 'AankGem':0.25 } ,targgem,'FactorKm')

pltjr4gra(summ1gemdataexp,'Jaar','FactorV','totaal aantal verplaatsingen',rscalet,False)

summ1gemvnexp= gemvncmpdf(summ1gemdataexp)

distmodescmp1Dim(summ1gemvnexp,["KAfstV_expl","MotiefV_expl"],"per afstand")  

pltverplfracs (summ1gemvnexp,"Culemborg")

print("klaar")


