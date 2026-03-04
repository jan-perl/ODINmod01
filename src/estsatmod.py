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
#Modellering van ODIN gegevens obv ruimtelijk dichtheden
# -

# todo
# actieve modes per afstandsklasse en type
# gemiddelde afstand per afstandsklasse en type
# active mode kms en passive mode kms per PC naar & van (let op: dubbeltelling)
# referentie reiskms actieve modes
# vergelijking co2 uitstoot active modes met kentallen


# +
#todo
#fit tov fractie (l komt uit data FactorVL)
#waarde:  p=1/(1/l + 1/f) -> f= 1/ (1/p - 1/l) -> divergeert dus alleen als w>.1, anders 0
#gewicht: w= l/ (l-p) -> aparte kolom

# +
#visualiseer relaties getallen, bijv. het 50 % punt(beide componenten gelijk groot) in km
#ook: bijdrage per ring / oppervlak -> multipliers per ring visualiseren
#omdat additief model _. som van componenten is ook terug te rekenen
#uitgaande van 1 PC : som model daar =1
#ga uit ven centrum punt postcode
#-maak grid met grootste ring
#-trek bijdrages vorige ring er helemaal van af (die komen later)
#bepaal coefficienten voor  'OW','OO','OM','OA' als som van alle groepen in ring
#tel op zodat een relatie plaatje op buurt/wijk niveau ontstaat
#valdeer dat de sommen voor die PC kloppen met model uitkomst

# +
#splits code op
#- analyse top PC4 per Motief voor 2 ritten op dag en niet woon kant
#- t.o.v. gebieds oppervlak
#- bij top PC4 maak annotatie naar 1 of meerdere PC6 mogelijk
# maak ook afstandsklassen tabel (met index per indeling naam) en schrijf naar pkl

#
#- maak grid met die toewijzing op 1 pixel & visualiseer met 5 km smooth
#- maak gridder routines in apart werkboek en save de output pkl
#- selecteer alleen grootste naarhuis - motief combinaties , rest: houdt van/naar, motief rest
#      *van/ naar samen selecteren !
#- 

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

#het inlezen van odinverplgr loopt in deze versie via ODINcatVNuse
#ODINcatVNuse zorgt ook voor defaults
import ODINcatVNuse

# +
#gebruik wat globale waarden, zoals in ODIN1lKAfmo.py
# -

useKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
xlatKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
#was<20
useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <180].copy()

fitgrps=['MotiefV','isnaarhuis']
expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0}

indatverplmxigr=pd.read_pickle("../intermediate/indatverplmxigr_ini.pkl") 
#MLlen(indatverplmxigr)

# indatverplmxigr is een excel baar overzicht met afstands overzichten per motief, gesommeerd over het land
# en niet gefilterd
# daarmee is dit formaat de basis voor gedrags schatters
# maak apart klein werkboek dat deze optmaliseert en parameters extract
# en run het daarna apart op gefilterde data
# indatverplmxigr


#M_LW_AL, M_LO_AL zou uiteindelijk voor landelijke schatting alleen afhankelijk moeten zijn van excluded PCs
#Voor vergelijking met de data moet je aannemen dat als een PC mist in de oorspronkelijke data
#het aantal reizen klein is, en dat dit hoort te passen bij de schatting
def showtotmot(df,metm):
    dfl = df[df['KAfstCluCode']==15].copy(deep=False)
    dfl['nrecdfl']=1
    shgr=['FactorV', 'nrecdfl']
    if metm:
        shgr = shgr + ['M_LW_AL','M_LO_AL']
    dflg= dfl.groupby(['MotiefV','GeoInd'])[shgr].agg('sum')
    return dflg.T
showtotmot(indatverplmxigr,True)    

indatverplmxigr [(indatverplmxigr ['FactorV']>0 ) ==False]


# +
#no data showtotmot(indatverplpc4gr,True)  

# +
#alleen eerste kolom vergelijken. de M_ velden zitten niet in deze database
#no data showtotmot(odinverplgr,False)

# +
#daarom verder met kolommen met een F_ (filtered)

#oude versie: ieder record fit naar ofwel ALsafe of naar osafe, of nergens heen

def choose_cutoffold(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    outframe=indat[[grpind,'GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    recisAL=indat['MaxAfst']==0
    wval1= indat[recisAL] [['FactorV',grpind,'GeoInd'] +pltgrps].copy(deep=False)
    wval1= wval1.rename(columns={'FactorV':'EstVPAL'})
    outframe=outframe.merge(wval1,how='left')    
    outframe['FactorVFAL'] = indat['FactorV']  /outframe['EstVPAL'] 
    outframe['FactorVFoCor'] =indat['FactorV'] 
    outframe['FactorVFo'] = indat['FactorV']  /outframe['EstVPAL']

    if hasfitted:
        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 500.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.3 * prevrres['FactorEstAL'] ) | (indat['FactorV'] < .3 *  prevrres['FactorEstAL'] )
        outframe['osafe']  = outframe['osafe']  & (outframe['MaxAfst']!=0)
        outframe['FactorVFo'] = indat['FactorV']  /prevrres['FactorEstAL']
#        outframe['FactorVFoCor'] =indat['FactorV']  * prevrres['FactorEstAL'] /outframe['EstVPAL'] 
#        outframe['ALsafe'] = indat['FactorV'] > .9 * outframe['EstVPAL'] 
#        outframe['osafe']  = (outframe['FactorVFoCor'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
    elif True:        
        outframe['ALsafe'] = indat['FactorV'] > 1.9 * outframe['EstVPAL'] 
        outframe['osafe'] =  indat['FactorV'] < .4 * outframe['EstVPAL'] 
        outframe['ALsafe'] = outframe['ALsafe'] | (outframe['MaxAfst']==0)
        outframe['osafe'] =  outframe['osafe'] & (outframe['MaxAfst']!=0) & (indat['FactorV'] > 0)
    else:
        outframe['ALsafe'] =True
        outframe['osafe'] = True
        outframe['FactorVFoCor'] =indat['FactorV'] 
        for lkey in ('LW','LO'):
            colnamAL="M_"+ lkey +"_AL"
            for okey in ('OW','OO'):
                colnam="M_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe['ALsafe'] = outframe['ALsafe'] & (indat[colnam] > 0.99  * indat[colnamAL])
                outframe['osafe']  = outframe['osafe']  & (indat[colnam] < 0.2 * indat[colnamAL])
        outframe['ALsafe'] = outframe['ALsafe'] | (outframe['MaxAfst']==0)
        outframe['osafe'] = outframe['osafe'] & (outframe['MaxAfst']!=0)
    if 1==1:
        outframe['ALsafe'] = outframe['ALsafe'].astype(int)
        outframe['osafe'] = outframe['osafe'].astype(int)
        overlap = outframe['osafe'] * outframe['ALsafe']
        outframe['ALsafe'] = outframe['ALsafe'] - overlap
        outframe['osafe'] = outframe['osafe'] - overlap
        if np.sum(outframe['osafe'] * outframe['ALsafe']) !=0:
            raise ("Error: overlapping fits")

        outframe['FactorVP'] =indat['FactorV']
        outframe['FactorVF'] = ( indat['FactorV'] * outframe['ALsafe'] +
                                  outframe['FactorVFoCor' ]* outframe['osafe']  )
        for lkey in ('LW','LO'):
            colnamAL ="M_"+ lkey +"_AL"
            colnamALo="F_"+ lkey +"_AL"
            outframe[colnamALo] = indat[colnamAL] * (outframe['ALsafe'])
            for okey in ('OW','OO','OM','OA'):
                colnam ="M_"+ lkey +"_" + okey
                colnamo="F_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe[colnamo] = indat[colnam] * (outframe['osafe'])
#    outframe['ALmult'] = ( (outframe['ALsafe']==False).astype(int))
    return outframe

#cut2=  choose_cutoffold(indatverplgr,fitgrps,False,0,'PC4',expdefs)   
cut2=  choose_cutoffold(indatverplmxigr,fitgrps,False,0,'mxigrp',expdefs)   
#cut2

# +
#todo
#fit tov fractie (l komt uit data FactorVL)
#waarde:  p=1/(1/l + 1/f) -> f= 1/ (1/p - 1/l) -> divergeert dus alleen als w>.1, anders 0
#gewicht: w= (p * (1/p - 1/l))** (+ pow+1)   -> aparte kolom
# -

def pointspertype(cutdf):
    cutcnt= cutdf.copy(deep=False)[['osafe','ALsafe','FactorVP','GrpExpl','FactorVFAL']]
    cutcnt['FactorVok'] =cutcnt['FactorVP'] >0
    cutcnt['allrecs'] = cutcnt['FactorVok']*0+1
    cutcnt['osafrat'] = cutcnt['FactorVFAL'] * cutcnt['osafe']
    rv= cutcnt.groupby('GrpExpl').agg('sum')
    rv['osafrat'] = rv['osafrat'] / rv['osafe']
    mlim=0
    return rv[rv['allrecs'] > mlim]
pointspertype(cut2)


# +
#originele code had copy. Kost veel geheugen en tijd
#daarom verder met kolommen met een F_ (filtered)

def choose_cutoffnw(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    curvpwr = pu['CP']
    outframe=indat[['PC4','GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    minwgt=.5
    recisAL=indat['MaxAfst']==0
    if hasfitted:
#        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 5.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
#        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
        outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
        outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
        outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
    else: 
        wval1= indat[recisAL] [['FactorV','PC4','GeoInd'] +pltgrps].copy(deep=False)
        wval1= wval1.rename(columns={'FactorV':'EstVPAL'})
        wval1['EstVPAL'] =np.power(wval1['EstVPAL'],curvpwr)
        outframe=outframe.merge(wval1,how='left')
        outframe['EstVP']  =np.power(indat['FactorV'],curvpwr)
        outframe['EstVPo'] =np.where (recisAL,1e20,1e-9 )    
    if 1==1:
        outframe['FactorVP'] =np.power(indat['FactorV'],curvpwr)
        denomo=  (1/outframe['EstVP']- 1/outframe['EstVPAL'])
        outframe['osafe'] = np.where( outframe['EstVP'] * denomo <=0,0, np.power(
                             outframe['EstVP']*denomo,  curvpwr+1) )  
        outframe['osafe'] =np.where (outframe['osafe'] <minwgt,0,outframe['osafe'] )       
        outframe['FactorVFo'] = np.where(outframe['EstVP']* denomo <=0,0, 
                            np.power(denomo ,-1/curvpwr))
        outframe['FactorVFo'] = indat['FactorV'] *outframe['FactorVFo'] /outframe['EstVP'] 
    if hasfitted:
        denomAL= (1/outframe['EstVP']- 1/outframe['EstVPo'])
    else:
        denomAL= (1/outframe['EstVP']- np.power(outframe['FactorVFo'],- curvpwr ) ) 
    if 1==1:
        outframe['FactorVFAL'] = np.where(outframe['EstVP']* denomAL <=0,0, 
                            np.power(denomAL ,-1/curvpwr))
        outframe['FactorVFAL'] = indat['FactorV'] *outframe['FactorVFAL'] /outframe['EstVP'] 
        outframe['FactorVFAL'] = np.where (recisAL,indat['FactorV'],
                            outframe['FactorVFAL'] )               
    if hasfitted:
        outframe['ALsafe'] = np.where( outframe['EstVP'] * denomAL <=0,0, np.power(
                             outframe['EstVP']*denomAL,  curvpwr+1) )  
        outframe['ALsafe'] = np.where (outframe['ALsafe'] <minwgt,0,outframe['ALsafe'] )  
        outframe['ALsafe'] = np.where (recisAL,1.0,outframe['ALsafe'] )                             

        chooseAL=np.where( (outframe['ALsafe'] > outframe['osafe'] )| recisAL ,1,0)
        outframe['ALsafe'] = outframe['ALsafe'] *chooseAL
        outframe['osafe'] = outframe['osafe'] *(1-chooseAL)
    else:                
        outframe['ALsafe'] = np.where (recisAL,1,0)
    if 1==1:
        if np.sum(outframe['osafe'] * outframe['ALsafe']) !=0:
            raise ("Error: overlapping fits")
        outframe['FactorVF'] = (outframe['FactorVFo']*outframe['osafe'] +
                                outframe['FactorVFAL'] * outframe['ALsafe'] )
        for lkey in ('LW','LO'):
            colnamAL ="M_"+ lkey +"_AL"
            colnamALo="F_"+ lkey +"_AL"
            outframe[colnamALo] = indat[colnamAL] * outframe['ALsafe']
            for okey in ('OW','OO','OM','OA'):
                colnam ="M_"+ lkey +"_" + okey
                colnamo="F_"+ lkey +"_" + okey
#                print(colnam,(lvals[np.isnan(lvals)]))
                outframe[colnamo] = indat[colnam] * outframe['osafe']
#        outframe['osafe2'] =outframe['osafe']
#        outframe['ALsafe2'] =outframe['ALsafe']
        print( ( np.sum(outframe['ALsafe']),np.sum(outframe['osafe']),
                 np.max(outframe['ALsafe']),np.max(outframe['osafe'])) )
    return outframe

def choose_cutoff(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr):
    if False:
        return choose_cutoffnw(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)
    else:
        return choose_cutoffold(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)


#cut2=  choose_cutoff(indatverplgr,fitgrps,False,0,'PC4',expdefs)   
cut2=  choose_cutoff(indatverplmxigr,fitgrps,False,0,'mxigrp',expdefs)   
#cut2
# -

def fitinddiag(fitdf,motiefc,naarhuisc,geoindex,grpind,pu):
    curvpwr = pu['CP']
    seldf = fitdf [ (fitdf ['MotiefV'] ==motiefc) &
                  (fitdf ['isnaarhuis'] ==naarhuisc) &
                  (fitdf ['GeoInd'] ==geoindex) & 
                  (fitdf ['FactorVP'] >0)] .copy()
    if False:
        print(seldf[['EstVPAL','MotiefV']].groupby(['EstVPAL']).agg('count').\
         groupby(['MotiefV',' isnaarhuis']).agg('count'))
    pldf=    seldf[['osafe','ALsafe']].copy()
    pldf['FactorVPrel'] =    seldf['FactorVP'] /  seldf['EstVPAL']  
    sumchk = np.power (np.power(seldf['FactorVFo'],-curvpwr) + 
                       np.power(seldf['FactorVFAL'] ,-curvpwr) , -curvpwr )
    pldf['FactorVSrel'] = sumchk /  seldf['FactorVP']
    pldf['FactorVFoCor'] =   np.where(seldf['osafe']==0,-.1,
                                      seldf['FactorVFo'] /  seldf['FactorVP']  )
    pldf['FactorVFALCor'] =   np.where(seldf['ALsafe']==0,-.1,
                                seldf['FactorVFAL'] /  seldf['FactorVP'] )
    if True:
        print(pldf[pldf['FactorVFALCor'] >5])
        print(seldf[pldf['FactorVFALCor'] >5])
        print(pldf[( abs(pldf['FactorVSrel']-1) >1e-3) & (( abs(pldf['FactorVSrel']-0) >1e-3)) ])
    plmelt =  pd.melt(pldf, 'FactorVPrel', var_name='cols',  value_name='vals')
    
    if False:
        print(seldf.sort_values(by=[grpind,'KAfstCluCode'])[['FactorVFo','FactorVP']] )
    fig, ax = plt.subplots()    
    seaborn.scatterplot(data=plmelt,x="FactorVPrel",y="vals", hue='cols', ax=ax)
    ax.set_xscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fitinddiag(cut2,10,5,'VertPC','mxigrp',expdefs)    


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


#@jit(parallel=True)
def _fitsub(indf,fitgrp,_regressgrp,  colvacols2, colpacols2):
    rf= indf.groupby(fitgrp ).apply( _regressgrp, 'FactorVF', colvacols2, colpacols2)
    return rf
    
    
def fit_cat_parameters(indf,topreddf,pltgrp,pu):
    debug=False
    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'F_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    Fitperscale=False
    if Fitperscale:
        fitgrp=pltgrp +['KAfstCluCode','GeoInd'  ]
    else:
        fitgrp=pltgrp + ['GeoInd' ]
    rf= _fitsub(indf,fitgrp,_regressgrp,  colvacols2, colpacols2).reset_index()
    return rf

def predict_values(indf,topreddf,pltgrp,rf,pu,stobijdr):
    curvpwr = pu['CP']
#    indf = indf[(indf['MaxAfst']!=95.0) & (indf[pltgrp]<3) ]
    debug=False
    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'F_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    
    outdf = topreddf.merge(rf,how='left')
    #let op: voorspel uit M kolommen
    colvacols2 = np.array( list ( (re.sub(r'P_','M_',s) for s in list(colpacols2) ) ) )
    
    colpacols2alch = np.array( list ( (re.sub(r'_AL','_xx',s) for s in list(colvacols2) ) ) )
    colpacols2alchisAL = (colpacols2alch !=colvacols2)
    if (debug):
        print(colpacols2alchisAL)
    blk1=outdf[colvacols2 ] * ((colpacols2alchisAL ==False ).astype(int))
    blk2=outdf[colpacols2 ] * ((colpacols2alchisAL ==False ).astype(int))
#    print(blk1)
    s2= np.sum(np.array(blk1)*np.array(blk2),axis=1).astype(float)

    blk1al=outdf[colvacols2 ] * (colpacols2alchisAL.astype(int))
    blk2al=outdf[colpacols2 ] * (colpacols2alchisAL.astype(int))
#    print(blk1)
    s2al= np.sum(np.array(blk1al)*np.array(blk2al),axis=1).astype(float)    
    outdf['FactorEstAL']  =s2al
#todo: als s2al 0 is, opzoeken waar MaxAfst ==0 en de s2al daarvan invullen als default
#dat levert dan min of meer consistente kantelpunten op
    outdf['FactorEstNAL'] =s2
    if (debug):
        print ((s2al, s2))
    #s2ch= np.min( (np.where((s2==0),s2al,s2 ), np.where((s2al==0),s2,s2al ) ) ,axis=0)
    s2ch= np.where((s2<=0),np.where(outdf['MaxAfst']==0, s2al,0), 
                           np.where((s2al==0),s2,
                           np.power (np.power(s2,-curvpwr) + np.power(s2al,-curvpwr), -curvpwr )) )
    if (debug):
        print (s2ch)
    outdf['FactorEst'] = s2ch
    outdf['DiffEst'] = np.where(outdf['FactorV']>0, outdf['FactorV']-s2ch,np.nan)
    if stobijdr:
        outdf[colvacols2 ] = np.array(blk1al)*np.array(blk2al)
    return(outdf)

def _dofitdatverplgr(indf,topreddf,pltgrp,pu):
    rf = fit_cat_parameters(indf,topreddf,pltgrp,pu)
    return predict_values(indf,topreddf,pltgrp,rf,pu,False)

fitpara= fit_cat_parameters(cut2,indatverplmxigr,fitgrps,expdefs)
fitdatverplgr = predict_values(cut2,indatverplmxigr,fitgrps,fitpara,expdefs,False)
#fitdatverplgr = dofitdatverplgr(cut2,indatverplgr,fitgrps,expdefs)
#fitdatverplgr = dofitdatverplgr(cut2,indatverplmxigr,fitgrps,expdefs)
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="FactorEst",y="DiffEst",hue="GeoInd")
# -

cut3=  choose_cutoff(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs)  
#cut3=  choose_cutoff(indatverplgr,fitgrps,True,fitdatverplgr,expdefs)  
#cut3

# +
#fitinddiag(cut3,10,5,'VertPC',p_CP)    
# -

#voor de time being, overschrijf de vorige selectie gegevens
for r in range(2):
    cut3=  choose_cutoff(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs) 
    fitpara= fit_cat_parameters(cut3,indatverplmxigr,fitgrps,expdefs)
    fitdatverplgr = predict_values(cut3,indatverplmxigr,fitgrps,fitpara,expdefs,False)
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="FactorEst",y="DiffEst",hue="GeoInd")

fitdatverplgr["x_LM_AL"] = fitdatverplgr["M_LW_AL"] * fitdatverplgr["M_LO_AL"]
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="x_LM_AL",y="DiffEst",hue="GeoInd")

seaborn.scatterplot(data=fitdatverplgrx,x="M_LO_AL",y="DiffEst",hue="GeoInd")

pointspertype(cut3)

gr5km=fitdatverplgr[(fitdatverplgr['MaxAfst']==5) & (fitdatverplgr['MotiefV']==1)].copy()
gr5km['linpmax']=gr5km['FactorEstNAL']/ gr5km['FactorEstAL']
gr5km['linpch']= gr5km['FactorEst']/ gr5km['FactorEstAL']
gr5km['drat']= gr5km['FactorV']/ gr5km['FactorEstAL']
fig, ax = plt.subplots()
seaborn.scatterplot(data=gr5km,x="linpmax",y="drat",size=.02,hue="GeoInd",ax=ax)
ax.set_xscale('log')
ax.set_yscale('log')

# +
#gr5km[gr5km['linpch'] >1000][['FactorEst','FactorEstAL','FactorEstNAL'] ]
# -

ds=fitdatverplgr[(fitdatverplgr['MotiefV']==1) & (fitdatverplgr['MaxAfst']==0)].sort_values(by='DiffEst').copy()
ds['dchk' ] = ds['DiffEst'] / ds ['FactorVSpec']
ds


# +
#noot normeren naar 1e6 rittenjaar
#noot2: er zal een afstand schaal zijn waarbij M_LW_OM en M_LO_OM een betere fit info geven
# het is niet zo vreemd dat het totale aantal woon-werk ritten ook een 
# nabijheids component heeft

def mxitotdiagpl(ds,xax):
    fig, ax = plt.subplots()
    #seaborn.scatterplot(data=ds,x='mxigrp',y='DiffEst',hue='GrpExpl')
    ds['huecol'] = ds['GrpExpl'] + ' ' + ds['GeoInd']
    seaborn.scatterplot(data=ds,x=xax,y='FactorV',hue='huecol')
    seaborn.lineplot(data=ds,x=xax,y='FactorEst',hue='huecol')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('1 groep landelijke sommen per mxigrp : data versus fit')
    print(ds.groupby([ 'GrpExpl','GeoInd'])[['FactorV','FactorEst','DiffEst']].agg('sum') )
#mxitotdiagpl(ds,'M_LO_AL') 
mxitotdiagpl(ds,'mxigrp')
# -

fitdatverplgr[fitdatverplgr['MaxAfst']==0]

paratab=fitdatverplgr[fitdatverplgr['MaxAfst']==0].groupby (['MotiefV','isnaarhuis','GeoInd']).agg('mean')
paratab.to_excel("../output/fitparatab1.xlsx")
paratab

fitdatverplgr.groupby (['MotiefV','isnaarhuis','GeoInd','MaxAfst']).agg('mean')


# +
def getmaxafstadmax( dd, landcod,myKAfstV):
    rf = dd[dd['KAfstCluCode'] == landcod ] 
    binst= myKAfstV.iloc[-2,1]
    print(binst)
    binstm=1
    rf['MaxShow'] = binstm * rf['FactorKm_c'] / rf['FactorV_c'] - (binstm-1)*binst
    rf['MaxShStat'] =  rf['FactorV_c'] 
    rf = rf[ODINcatVNuse.fitgrpse +['MaxShow','MaxShStat']]
    return rf
#maxvals = 
getmaxafstadmax(ODINcatVNuse. odindiffflginfo, ODINcatVNuse.landcod,useKAfstV)

# hier komen waarden uit ONDER binstm. Dat is niet goed.

# +
#mogelijk: maxafst joinen uit diifdata


def pltmotdistgrp (mydati,horax,vertax,vnsep):
    mydat=pd.DataFrame(mydati)    
    opdel=['MaxAfst','GeoInd','MotiefV','GrpExpl']
#    mydat['FactorEst2'] = np.where(mydat['FactorEstNAL']==0,0, 
#                                 1/ (1/mydat['FactorEstAL'] + 1  /mydat['FactorEstNAL'] ) )
    fsel=['FactorEst','FactorV', 'FactorEstNAL','FactorEstAL']

    rv2= mydat.groupby (opdel)[fsel].agg(['sum']).reset_index()
#    print ( rv2[ (rv2['MotiefV']==1) & (rv2['MaxAfst']==0) ] ) 
    rv2.columns=opdel+fsel
    if(vertax=='FactorEst'):
        limcat=2e9
    else:
        limcat=1e9
    bigmotd= rv2[(rv2['MaxAfst']==0) & (rv2['FactorV']>limcat)  ].groupby('GrpExpl').agg(['count']).reset_index()
    bigmotl = list( bigmotd['GrpExpl'])
#    print(bigmotl)
    stelafst =100.0
    rv2['MaxAfst']=np.where(rv2['MaxAfst']==0 ,stelafst ,rv2['MaxAfst'])
#    rv2['MaxAfst']=rv2['MaxAfst'] * np.where(rv2['GeoInd']=='AankPC',1,1.02)
    rv2['Qafst']=1/(1/(rv2['MaxAfst']  *0+1e10) +1/ (np.power(rv2['MaxAfst'] ,1.8) *2e8 ))
    rv2['linpmax'] = rv2['FactorEstNAL']/ rv2['FactorEstAL']
    rv2['linpch']= rv2['FactorEst']/ rv2['FactorEstAL']
    rv2['drat']= rv2['FactorV']/ rv2['FactorEstAL']

    rvs = rv2[np.isin(rv2['GrpExpl'],bigmotl)]
#    rv2['MotiefV']=rv2['MotiefV'].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(12, 6))
    
#    print ( rvs[ (rvs['MotiefV']==1) & (rvs['MaxAfst']== stelafst) ] ) 
    
    rvs['huecol'] = rvs['GrpExpl']
    if vnsep:
        rvs['huecol'] = rvs['huecol'] + ' ' + rvs['GeoInd']
    if(vertax=='FactorEst'):
        seaborn.scatterplot(data=rvs,x=horax,y='FactorV',hue='huecol',ax=ax)
        seaborn.lineplot(data=rvs,x=horax,   y='FactorEst',hue='huecol',ax=ax)
#Qafst nooit zomaar plotten: ggeft grote verwarring !
#        seaborn.lineplot(data=rvs,x=horax,   y='Qafst',ax=ax)
    elif(vertax=='linpch'):
        seaborn.scatterplot(data=rvs,x=horax,y='drat',hue='huecol',ax=ax)
        ax.axhline(0.5)
        seaborn.lineplot(data=rvs,x=horax,   y='linpch',hue='huecol',ax=ax)
    ax.set_xscale('log')
#    ax.set_yscale('log')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figname = "../output/gplo_fmdg_"+"horax"+"_"+vertax+"_"+'G1.svg';
    fig.savefig(figname, bbox_inches="tight") 
    return (rv2)
ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','FactorEst',False)
# -

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==1],'MaxAfst','FactorEst',True)

ov=pltmotdistgrp(fitdatverplgr,'linpmax','linpch',False)

ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','linpch',False)


def calcchidgrp (mydati):
    mydat=pd.DataFrame(mydati)
    mydat['chisq'] = mydat['DiffEst'] ** 2
    mydat['insq'] = mydat['FactorV'] ** 2
    csel=['chisq','insq']
    opdel=['MaxAfst','GeoInd']
    rv= mydat.groupby (opdel)[csel].agg(['sum']).reset_index()
    rv.columns=opdel+csel
    rv['ChiRat'] = rv['chisq']/ rv['insq']
    mydat['HeeftEst']=(np.isnan(mydat['FactorEst'])==False).astype(int)
    mydat['HeeftFV']=(np.isnan(mydat['FactorV'])==False).astype(int)
    fsel=['FactorEst','FactorV','HeeftEst','HeeftFV']
    rv2= mydat.groupby (opdel)[fsel].agg(['sum']).reset_index()
    rv2.columns=opdel+fsel
#    print(rv2)
    rv2['EstRat'] = rv2['FactorEst']/ rv2['FactorV']
    rv=rv.merge(rv2,how='left')
    return rv
calcchidgrp(fitdatverplgr)




