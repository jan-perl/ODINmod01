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

# +
#het inlezen van odinverplgr loopt in deze versie via ODINcatVNuse
#ODINcatVNuse zorgt ook voor defaults
#import ODINcatVNuse
# -

#global version for consistency
#version 1: cutoff old
#version 3: cutoff new
esmalgversion=8

# +
#gebruik wat globale waarden, zoals in ODIN1lKAfmo.py
# -

if False:
    useKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
    xlatKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
    #was<20
    useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <180].copy()

fitgrps=['MotiefV','isnaarhuis']
#SP tussen 0.3 en 1 per motief
expdefs = {'LW':1.2, 'LO':1.0, 'OA':1.0,'CP' :1.0,'SP' :1.0, 'XAL':2.5}

indatverplmxigr=pd.read_pickle("../intermediate/indatverplmxigr_ini.pkl") 
#MLlen(indatverplmxigr)
indatverplmxigr['PC4'] =0
indatverplmxigr.dtypes

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
esmalgversionscoold=[1,2]

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

def pointspertype(cutdf,grps=fitgrps,selmask='All'):
    cutcnt= cutdf.copy(deep=False)[['osafe','ALsafe','FactorVP','GrpExpl','FactorVFAL']+grps]
    cutcnt['FactorVok'] =cutcnt['FactorVP'] >0
    cutcnt['allrecs'] = cutcnt['FactorVok']*0+1
    cutcnt['osafrat'] = cutcnt['FactorVFAL'] * cutcnt['osafe']
    rv= cutcnt.groupby(['GrpExpl']+grps).agg('sum').reset_index()
    rv['osafrat'] = rv['osafrat'] / rv['osafe']
    if selmask=='All':
        mlim=0
        rmask=rv['allrecs'] > mlim
    elif selmask=='rondje huis':        
        rmask=rv['GrpExpl'].str.contains(selmask)
    elif selmask=='Err':        
        rmask=np.isnan(rv['FactorVFAL']  ) | (rv['osafe'] ==0) | (rv['osafe'] ==0)
    else:
        die ("pointspertype: wrong value for " +selmask)
    return rv[rmask]
#pointspertype(cut2,fitgrps,'rondje huis').sort_values('FactorVP')
#pointspertype(cut2,fitgrps,'Err')
#pointspertype(cut2,fitgrps,'All')
#pointspertype(cut2,'Err')
#pointspertype(cut2)


# +
def choose_cutoffv5(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    curvpwr = pu['CP']
    outframe=indat[[grpind,'GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    minwgt=.5
    recisAL=indat['MaxAfst']==0
    if hasfitted:
#        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 5.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
#        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
        outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
        outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
        outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
    else: 
        wval1= indat[recisAL] [['FactorV',grpind,'GeoInd'] +pltgrps].copy(deep=False)
        wval1= wval1.rename(columns={'FactorV':'EstVPAL'})
        wval1['EstVPAL'] =np.power(wval1['EstVPAL'],curvpwr)
        outframe=outframe.merge(wval1,how='left')
        outframe['EstVP']  =np.power(indat['FactorV'],curvpwr)
        outframe['EstVPo'] =1.0
        outframe['EstVPo'] =np.where (recisAL,1e20*outframe['EstVPo'],1e-9 *outframe['EstVPo'])        
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


#cut2=  choose_cutoff(indatverplgr,fitgrps,False,0,'PC4',expdefs)   
cut2=  choose_cutoffv5(indatverplmxigr,fitgrps,False,0,'mxigrp',expdefs)   
#cut2

# +
#originele code had copy. Kost veel geheugen en tijd
#daarom verder met kolommen met een F_ (filtered)

def choose_cutoffv6(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    curvpwr = pu['CP']
    outframe=indat[[grpind,'GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps].copy(deep=False)
    minwgt=.5
    recisAL=indat['MaxAfst']==0
    estAL2=(esmalgversion==7)    
    if hasfitted:
#        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 5.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
#        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
        outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
        outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
        outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
    else: 
        wval1= indat[recisAL] [['FactorV',grpind,'GeoInd'] +pltgrps].copy(deep=False)
        wval1= wval1.rename(columns={'FactorV':'EstVPAL'})
        wval1['EstVPAL'] =np.power(wval1['EstVPAL'],curvpwr)
        print (outframe.shape)
        outframe=outframe.merge(wval1,how='left')
        print (outframe.shape)
        outframe['EstVP']  =np.power(indat['FactorV'],curvpwr)
#        outframe['EstVPo'] =np.where (recisAL,1e20,1e-9 )    
        outframe['EstVPo'] =1.0
        outframe['EstVPo'] =np.where (recisAL,1e20*outframe['EstVPo'],1e-9 *outframe['EstVPo']) 
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
    if hasfitted & estAL2:
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
cut2=  choose_cutoffv6(indatverplmxigr,fitgrps,False,0,'mxigrp',expdefs)   


# +
def satfunc(v_AL,v_in,pu):
        satpwr = pu['SP']
        onezero=(v_in<=0) | (v_AL <=0)
#        print(v_in[onezero])
        val=  np.where (onezero,0,
              np.power( (np.power(v_in +onezero,-satpwr ) +  np.power(v_AL+onezero ,-satpwr) ), -1/satpwr )  )
        return val

def satinvfunc(v_AL,val,pu):
        satpwr = pu['SP']
        badval=np.where(np.isnan(v_AL*val),1,  (val > v_AL) | (v_AL<=0))
        wgt= np.where (badval,0,np.power((v_AL-val)/v_AL,satpwr+1) )
        v_in=np.where (badval,0,
            np.power( (np.power(val ,-satpwr ) -  np.power(v_AL ,-satpwr) ), -1/satpwr )  )
        return(wgt,v_in)

#while this is still symmetric    
def satinvfuncAL(v_in,val,pu):
        return satinvfunc(v_in,val,pu)    


# +
#fit AL alleen op indat['MaxAfst']==0. Deze waarde blijft gelijk voor asymptoot
#gebruik asymptoot in ronde 2 voor correcties

def choose_cutoffv8(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    curvpwr = pu['CP']    
    colscpy=[grpind,'GrpExpl','MaxAfst','KAfstCluCode','GeoInd' ] +pltgrps
    outframe=indat[colscpy].copy(deep=False)
    minwgto=np.power(0.25,pu['SP'])
    minwgtal=.5 

    recisAL=indat['MaxAfst']==0
    if 1==1:
        outframe['FactorVP'] =np.power(indat['FactorV'],curvpwr)
        wval1= indat[recisAL] [['FactorV',grpind,'GeoInd'] +pltgrps].copy(deep=False)
        wval1= wval1.rename(columns={'FactorV':'LimVPAL'})
        wval1['LimVPAL'] =np.power(wval1['LimVPAL'],curvpwr)
        outframe=outframe.merge(wval1,how='left')
    if hasfitted:
#        outframe['ALsafe'] = (prevrres['FactorEstNAL'] > 5.0 * prevrres['FactorEstAL'] ) | (outframe['MaxAfst']==0) 
#        outframe['osafe']  = (prevrres['FactorEstNAL'] < 0.2 * prevrres['FactorEstAL'] ) & (outframe['MaxAfst']!=0)
        outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
        outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
        outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
        (wgt_est,v_in_est) = satinvfunc (outframe['EstVPAL'] , outframe['FactorVP'] ,pu)
        (wgt_lim,v_in_lim) = satinvfunc (outframe['LimVPAL'] , outframe['FactorVP'] ,pu)
        EstVPoh  =np.where(outframe['EstVPo'] ==0, 1e4*outframe['FactorVP'],outframe['EstVPo'] )
        (wgt_al,v_al_est) =  satinvfunc (EstVPoh , outframe['FactorVP'] ,pu)
        hlpos=pu['XAL']
        extral= (outframe['LimVPAL'] >= hlpos * outframe['EstVPAL']) | (outframe['LimVPAL'] * hlpos <= outframe['EstVPAL']) 
        if 1==0:
            print (outframe[extral & recisAL][colscpy+['LimVPAL','EstVPAL','FactorVP']])
       
#voor diagnostiek      
#        outframe['NormS2'] = v_in_est
    else: 
#bootstrap by selecting smaller FactorVs within the grpind        
        outframe['EstVP']  =np.power(indat['FactorV'],curvpwr)
        outframe['EstVPAL']  =outframe['LimVPAL'] 
        outframe['EstVPo'] =1.0
        outframe['EstVPo'] =np.where (recisAL,1e20*outframe['EstVPo'],1e-9 *outframe['EstVPo'])   
        extral=False

    if hasfitted:
        wgt_est = np.sqrt(wgt_est*wgt_lim)
        outframe['osafe'] = np.where(wgt_est>minwgto, wgt_est,0)
        outframe['FactorVFo'] = np.where(outframe['osafe'] >0, v_in_est,0)        
        outframe['FactorVFAL'] = v_al_est
        #outframe['ALsafe'] = np.where(recisAL,wgt_al ,0)
        outframe['ALsafe'] = np.where(wgt_al>minwgtal,wgt_al ,0)
    else:
        (wgt_est,v_in_est) = satinvfunc (outframe['EstVPAL'] , outframe['FactorVP'] ,pu)
        outframe['osafe'] = np.where(wgt_est>minwgto, wgt_est,0)
        outframe['FactorVFo'] = np.where(outframe['osafe'] >0, v_in_est,0)        
        outframe['FactorVFAL'] = outframe['FactorVP'] 
        outframe['ALsafe'] = np.where(recisAL,np.where( (outframe['FactorVFAL'] !=0),1,1e-3),0)
    if 1==0:
        outframe['osafe'] = np.where((outframe['FactorVFo'] < (minwgt * outframe['EstVPAL'] )) & \
                                 (outframe['FactorVFo'] !=0) &  outframe['osafe'] ,1,0) *(1-recisAL)
    if 1==1:
        outframe['osafe'] = np.where((outframe['osafe'] !=0) | recisAL | ( outframe['FactorVFo'] ==0),
                                 outframe['osafe'] ,1e-10)        
        bothmask=((outframe['osafe'] * outframe['ALsafe']==0 ) & (extral ==False) )
        outframe['osafe'] *=bothmask
        outframe['ALsafe'] *=bothmask
        if np.sum(outframe['osafe'] * outframe['ALsafe']) !=0:
            raise ("Error: overlapping fits")
        outframe['FactorVF'] = (outframe['FactorVFo']*outframe['osafe'] +
                                outframe['FactorVFAL'] * outframe['ALsafe'] )
    if 0==1:
        ofwithna= outframe[np.isnan(outframe['FactorVF'])]
        print (ofwithna)
    if 1==1:
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
    if esmalgversion in esmalgversionscoold:
        return choose_cutoffold(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)
    elif esmalgversion in [5]:
        return choose_cutoffv5(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)
    elif esmalgversion in [6,7]:
        return choose_cutoffv6(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)
    else:
        return choose_cutoffv8(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr)


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
                       np.power(seldf['FactorVFAL'] ,-curvpwr) , -1/curvpwr )
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

pointspertype(cut2)


# +
def _regressgrp(indf, yvar, xvars,pcols):  
#        reg_nnls = LinearRegression(fit_intercept=False )
        y_train=indf[yvar]
        X_train=indf[xvars]
#        print(('o',len(indf),X_train.sum(),y_train.sum()) )
        if 1==1:
            #smask = ~ (np.isnan(y_train) | np.isnan(np.sum(X_train)) )
            #smask = np.where(False== np.isnan(y_train) ,
            #                  (y_train >0) & (np.sum(X_train,axis=1) >0) ,0)
            smask = (y_train >0) & (np.sum(X_train,axis=1) >0)
            indf= indf[smask]
            y_train=indf[yvar]
            X_train=indf[xvars]
#            print(('f', len(indf)))
        else:
            y_train[np.isnan(y_train)]=0.1
        if(len(indf)==0) :
            rv=np.zeros(len(xvars))
        else:
#            print(('lr',len(indf),X_train.sum(),y_train.sum()) )
            fit1 = nnls(X_train, y_train)    
            rv=pd.DataFrame(fit1[0],index=pcols).T
        return(rv)


#@jit(parallel=True)
def _fitsub(indf,fitgrp,_regressgrp,  colvacols2, colpacols2):
    rf= indf.groupby(fitgrp ).apply( _regressgrp, 'FactorVF', colvacols2, colpacols2)
    return rf

landcrossmults= {'OA':1,'OW':1,'OO':1,'OM':1+1e-10}
lwlopair=['LW','LO']

def addtotscgrps(rf,pu):
     for lkey in lwlopair:
        colin="P_"+ lkey +"_" + "AL"
        satlim= rf[colin]
        screst=0
        for okey in landcrossmults.keys():
            colnam="P_"+ lkey +"_" + okey
            screst +=  rf[colnam]*  landcrossmults[okey]
#        sc2=satfunc(satlim,screst,pu) 
        colout="S_"+ lkey +"_" + "AL"
        rf[colout] =screst
    
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
    addtotscgrps(rf,pu)
    return rf

#previously, these were est to zero; now they are used as a check
def setlandcols(outdf,colvacols2,pu ):
    landdf=outdf.copy()
    for lkey in lwlopair:
        colin="M_"+ lkey +"_" + "AL"
        indat= landdf[colin]
        for okey in landcrossmults.keys():
            colnam="M_"+ lkey +"_" + okey
            landdf[colnam]= indat * landcrossmults[okey]
    return landdf    

#neem de parameters , en voorspel
#neem ook landelijke normalisatie mee
def predict_values(indf,topreddf,pltgrp,rf,pu,stobijdr):    
#    indf = indf[(indf['MaxAfst']!=95.0) & (indf[pltgrp]<3) ]
    debug=False
    outdf = topreddf.merge(rf,how='left')

    colvacols = indf.columns
    colpacols = np.array( list ( (re.sub(r'F_','P_',s) for s in list(colvacols) ) ) )
    colvacols2 = colvacols[colvacols != colpacols]
    colpacols2 = colpacols[colvacols != colpacols]
    
    #let op: voorspel uit M kolommen
    colvacols2 = np.array( list ( (re.sub(r'P_','M_',s) for s in list(colpacols2) ) ) )
    oldALmode=True

    if oldALmode:
        colpacols2alch = np.array( list ( (re.sub(r'_AL','_xx',s) for s in list(colvacols2) ) ) )
        colpacols2alchisAL = (colpacols2alch !=colvacols2)
        if (debug):
            print(colpacols2alchisAL)
        blk1=outdf[colvacols2 ] * ((colpacols2alchisAL ==False ).astype(int))
        blk2=outdf[colpacols2 ] * ((colpacols2alchisAL ==False ).astype(int))
    else:
        blk1=outdf[colvacols2 ] 
        blk2=outdf[colpacols2 ] 
#    print(blk1)
    s2= np.sum(np.array(blk1)*np.array(blk2),axis=1).astype(float)
    outdf['FactorEstNAL'] =s2
    
    if 0&oldALmode:
        blk1al=outdf[colvacols2 ] * (colpacols2alchisAL.astype(int))
        blk2al=outdf[colpacols2 ] * (colpacols2alchisAL.astype(int))
    #    print(blk1)
        s2al= np.sum(np.array(blk1al)*np.array(blk2al),axis=1).astype(float)    
    elif 0==1:
        outdfland=setlandcols(outdf,colvacols2,pu )
        blk1al=outdfland[colvacols2 ] 
        blk2al=outdfland[colpacols2 ] 
    #    print(blk1)
        s2al= np.sum(np.array(blk1al)*np.array(blk2al),axis=1).astype(float)
    else:
        s2al=0
        s2nl=0
        for lkey in lwlopair:
            colext= lkey +"_" + "AL"
            s2al+= outdf["M_"+colext] *outdf["P_"+colext]
            s2nl+= outdf["M_"+colext] *outdf["S_"+colext]

    outdf['FactorEstAL']  =s2al
    outdf['FactorEstNL']  =satfunc(s2al,s2nl,pu) 
        
#todo: als s2al 0 is, opzoeken waar MaxAfst ==0 en de s2al daarvan invullen als default
#dat levert dan min of meer consistente kantelpunten op
    if (debug ):
        print (outdf[['FactorEstAL','FactorEstNAL']])
    #s2ch= np.min( (np.where((s2==0),s2al,s2 ), np.where((s2al==0),s2,s2al ) ) ,axis=0)
    #was s2ch= np.where(outdf['MaxAfst']==0, s2al,  np.where((s2<=0),0,  satfunc(s2al,s2,pu) ) )
    s2ch=  satfunc(s2al,s2,pu) 
    outdf['FactorEst'] = s2ch
    outdf['DiffEst'] = np.where(outdf['FactorV']>0, outdf['FactorV']-s2ch,np.nan)
    (wgt_rec,v_in_rec) = satinvfunc (s2al, outdf['FactorV'] ,pu)
    outdf['DiffS2'] = np.where(wgt_rec<=0, np.nan,s2-v_in_rec  )
    if (debug):
        print (outdf[['FactorEstAL','FactorEstNAL','FactorEst','DiffS2']])
    if stobijdr:
        outdf[colvacols2 ] = np.array(blk1al)*np.array(blk2al)
    return(outdf)


def _dofitdatverplgr(indf,topreddf,pltgrp,pu):
    rf = fit_cat_parameters(indf,topreddf,pltgrp,pu)    
    return predict_values(indf,topreddf,pltgrp,rf,pu,False)

fitpara= fit_cat_parameters(cut2,indatverplmxigr,fitgrps,expdefs)
if 1==1:
    fitdatverplgr = predict_values(cut2,indatverplmxigr,fitgrps,fitpara,expdefs,False)
    #fitdatverplgr = dofitdatverplgr(cut2,indatverplgr,fitgrps,expdefs)
    #fitdatverplgr = dofitdatverplgr(cut2,indatverplmxigr,fitgrps,expdefs)
    fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
    seaborn.scatterplot(data=fitdatverplgrx,x="FactorEst",y="DiffEst",hue="GeoInd")


# -

def exp_fitpara(fitdat,clufrom,outfn):
    if clufrom >20:
        sel=fitdatverplgr['MaxAfst']==0
    else:
        sel=fitdatverplgr['KAfstCluCode']>=clufrom
    paratab=fitdatverplgr[sel].groupby (['MotiefV','isnaarhuis','GeoInd','GrpExpl']).agg('mean').reset_index()
    paratab.to_excel("../output/"+outfn,index=False)
    return paratab
exp_fitpara(fitdatverplgr,99,"fitparatab0.xlsx")


# +
def diagdtaAL(indat,pltgrps,hasfitted,prevrres,grpind,pu):
    recisAL=(indat['MaxAfst']==0 ) & ((indat['FactorV']!=0 ) )
    outframe=indat.copy(deep=False)
    curvpwr=pu['CP']
    outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
    outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
    outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
    xaxval='EstVPAL'
    outframe['FactVrat']  = outframe['FactorV']  / outframe[xaxval]  
    ALframe= outframe[recisAL].copy(deep=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    seaborn.scatterplot(data=ALframe,x=xaxval,y="FactVrat",hue="GeoInd",ax=ax)  
    hlpos=pu['XAL']
    ax.axhline(hlpos,alpha=0.3)
    ax.axhline(1/hlpos,alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('total volume per area estimates (using AL): FactorV/ '+xaxval)
#    ax.set_yscale('log')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    retcols=fitgrps+[grpind,'GrpExpl','GeoInd',xaxval,'FactorV','FactVrat']
    extr= (ALframe['FactVrat'] > hlpos) | (ALframe['FactVrat']<1/ hlpos) 
    return ALframe[extr][retcols]

diagdtaAL(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs)     


# +
def diagdtaSF(indat,pltgrps,hasfitted,prevrres,grpind,curvpwr):
    recisAL=(indat['MaxAfst']==0 ) & ((indat['FactorV']!=0 ) )
    outframe=indat.copy(deep=False)
    curvpwr=1 #unused
    outframe['EstVPAL']  =np.power(prevrres['FactorEstAL'],curvpwr)
    outframe['EstVPo']   =np.power(prevrres['FactorEstNAL'],curvpwr)
    outframe['EstVP']    =np.power(prevrres['FactorEst'],curvpwr)
    xaxval='EstVPAL'
    outframe['FactVrat']  = outframe['FactorV']  / outframe[xaxval]  
    ALframe= outframe[recisAL].copy(deep=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    seaborn.scatterplot(data=ALframe,x=xaxval,y="FactVrat",hue="GeoInd",ax=ax)
    hlpos=2.5
    ax.axhline(hlpos,alpha=0.3)
    ax.axhline(1/hlpos,alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('total volume per area estimates (using AL): FactorV/ '+xaxval)
#    ax.set_yscale('log')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    retcols=fitgrps+[grpind,'GrpExpl','GeoInd',xaxval,'FactorV','FactVrat']
    extr= (ALframe['FactVrat'] > hlpos) | (ALframe['FactVrat']<1/ hlpos) 
    return ALframe[extr][retcols]

diagdtaSF(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs) 


# +
#fitdatverplgr.dtypes

# +
#mogelijk: maxafst joinen uit diffdata

def pltmotdistgrp (mydati,horax,vertax,vnsep):
    stelafst =195.0 # afstand waar maxbin wordt geplaatst
    mydat=pd.DataFrame(mydati)    
    opdel=['MaxAfst','GeoInd','MotiefV','GrpExpl']
#    mydat['FactorEst2'] = np.where(mydat['FactorEstNAL']==0,0, 
#                                 1/ (1/mydat['FactorEstAL'] + 1  /mydat['FactorEstNAL'] ) )
    fsel=['FactorEst','FactorV', 'FactorEstNAL','FactorEstAL','FactorEstNL','DiffS2']

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
    
    rv2['MaxAfst']=np.where(rv2['MaxAfst']==0 ,stelafst ,rv2['MaxAfst'])
#    rv2['MaxAfst']=rv2['MaxAfst'] * np.where(rv2['GeoInd']=='AankPC',1,1.02)
    rv2['Qafst']=1/(1/(rv2['MaxAfst']  *0+1e10) +1/ (np.power(rv2['MaxAfst'] ,1.8) *2e8 ))
    rv2['linpmax'] = rv2['FactorEstNAL']/ rv2['FactorEstNL']
    rv2['linpch']= rv2['FactorEst']/ rv2['FactorEstNL']
    rv2['drat']= rv2['FactorV']/ rv2['FactorEstNL']
    rv2['DiffS2']= np.where(rv2['DiffS2']< rv2['FactorEstNAL']/100 ,np.nan,  rv2['DiffS2'])

    rvs = rv2[np.isin(rv2['GrpExpl'],bigmotl)].copy()
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
        ax.axhline(0.5,alpha=0.3)
        seaborn.lineplot(data=rvs,x=horax,   y='linpch',hue='huecol',ax=ax)
    elif(vertax=='DiffS2'):
#        ax.axhline(0.0)
        seaborn.scatterplot(data=rvs,x=horax,   y=vertax,hue='huecol',ax=ax)
        seaborn.lineplot(data=rvs,x=horax,   y=horax,hue='huecol',ax=ax,alpha=0.3)
        ax.set_yscale('log')
    else:
        seaborn.scatterplot(data=rvs,x=horax,   y=vertax,hue='huecol',ax=ax)
        
    ax.set_xscale('log')
#    ax.set_yscale('log')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figname = "../output/gplo_fmdg_"+"horax"+"_"+vertax+"_"+'G1.svg';
    fig.savefig(figname, bbox_inches="tight") 
    return (rv2)
ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','FactorEst',False)
# -

ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','linpch',False)

cut3=  choose_cutoff(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs)  
#cut3=  choose_cutoff(indatverplgr,fitgrps,True,fitdatverplgr,expdefs)  
cut3

if 0==1:
    cutstrs=['GeoInd','GrpExpl']
    c3diff = cut3.drop(columns=cutstrs)-cut2.drop(columns=cutstrs)
    c3diff.to_excel("../output/chk-c3diff.xlsx")


def selrecs(c3in):    
    rv= c3in[fitgrps+['MaxAfst','GeoInd']+['FactorVP','EstVPAL','NormS2','FactorVF']].copy(deep=False)
    rv['r1']=rv['FactorVP']/rv['EstVPAL']
    rv['r2']=rv['FactorVP']/rv['NormS2']
    return rv
#selrecs(cut3) 


cut3[cut3[ 'FactorVFo']>1][[ 'FactorVFo']]

# +
#fitinddiag(cut3,10,5,'VertPC',p_CP)    
# -

pointspertype(cut3)

# +
#voor de time being, overschrijf de vorige selectie gegevens
nsatiterdef=4
for r in range(nsatiterdef):
    cut3=  choose_cutoff(indatverplmxigr,fitgrps,True,fitdatverplgr,'mxigrp',expdefs) 
    print(pointspertype(cut3))
    fitpara= fit_cat_parameters(cut3,indatverplmxigr,fitgrps,expdefs)
    fitdatverplgr = predict_values(cut3,indatverplmxigr,fitgrps,fitpara,expdefs,False)
    exp_fitpara(fitdatverplgr,99,"fitparatab"+(str(r+1))+".xlsx")
    
fitdatverplgrx = fitdatverplgr[abs(fitdatverplgr["DiffEst"])> 2e6] 
seaborn.scatterplot(data=fitdatverplgrx,x="FactorEst",y="DiffEst",hue="GeoInd")
# -

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

exp_fitpara(fitdatverplgr,99,"fitparatabd.xlsx")

exp_fitpara(fitdatverplgr,13,"fitparatab1hi.xlsx")

fitdatverplgr.groupby (['MotiefV','isnaarhuis','GeoInd','MaxAfst']).agg('mean')


# +
#deze routine wordt niet meer gebruikt
def getmaxafstadmax_old( dd, landcod,myKAfstV):
    rf = dd[dd['KAfstCluCode'] == landcod ] 
    binst= myKAfstV.iloc[-2,1]
    print(binst)
    binstm=1
    rf['MaxShow'] = binstm * rf['FactorKm_c'] / rf['FactorV_c'] - (binstm-1)*binst
    rf['MaxShStat'] =  rf['FactorV_c'] 
    rf = rf[ODINcatVNuse.fitgrpse +['MaxShow','MaxShStat']]
    return rf
#maxvals = getmaxafstadmax(ODINcatVNuse. odindiffflginfo, ODINcatVNuse.landcod,useKAfstV)

# hier komen waarden uit ONDER binstm. Dat is niet goed.


# -

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==2],'MaxAfst','FactorEst',True)

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']!=99],'linpmax','FactorEst',True)

ov=pltmotdistgrp(fitdatverplgr,'linpmax','linpch',False)

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==7],'linpmax','linpch',True)

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==8],'linpmax','linpch',True)

ov=pltmotdistgrp(fitdatverplgr[fitdatverplgr['MotiefV']==8],'FactorEstNAL','DiffS2',True)

ov=pltmotdistgrp(fitdatverplgr,'MaxAfst','linpch',False)

ov=pltmotdistgrp(fitdatverplgr,'linpmax','DiffS2',False)


def calcchidgrp (mydati,opdel):
    mydat=pd.DataFrame(mydati)
    mydat['chisq'] = mydat['DiffEst'] ** 2
    mydat['insq'] = mydat['FactorV'] ** 2
    csel=['chisq','insq']
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
calcchidgrp(fitdatverplgr,['GeoInd']).sort_values(['ChiRat'])


def tryexpp(indat,pu,niter,diag):
    lcut2=  choose_cutoff(indat,fitgrps,False,0,'mxigrp',pu)
    lfitpara= fit_cat_parameters(lcut2,indat,fitgrps,pu)
    lfitdatverplgr = predict_values(lcut2,indat,fitgrps,lfitpara,pu,False)
    if diag:
        chirat2=calcchidgrp(lfitdatverplgr,['GeoInd'])['ChiRat'].mean()
        print((pu,0,chirat2))
    for r in range(niter):
        lcut3=  choose_cutoff(indat,fitgrps,True,lfitdatverplgr,'mxigrp',pu) 
        lfitpara= fit_cat_parameters(lcut3,indat,fitgrps,pu)
        lfitdatverplgr = predict_values(lcut3,indat,fitgrps,lfitpara,pu,False)
        chirat2=calcchidgrp(lfitdatverplgr,['GeoInd'])['ChiRat'].mean()
        if diag:
            print((pu,r+1,chirat2))
tryexpp(indatverplmxigr,expdefs,nsatiterdef,True)        


def varpu(indat,pu,niter,diag):
    lpu=pu
    for sp in [.49,.5,.55,0.9,1.0]:
        lpu['SP'] =sp
        tryexpp(indat,lpu,niter,diag)
varpu(indatverplmxigr,expdefs,4,True) 

calcchidgrp(fitdatverplgr,['MaxAfst','GeoInd'])

calcchidgrp(fitdatverplgr,['MotiefV','GeoInd']).sort_values(['ChiRat'])

calcchidgrp(fitdatverplgr,['GeoInd']).sort_values(['ChiRat'])




