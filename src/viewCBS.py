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
#wat visualisaties van CBS gegevens
#gebruik dit boek om daardoorheen te brwosen
# -

import pandas as pd
import numpy as np
import os as os
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

import sys
print(sys.path)
sys.path.append('/home/jovyan/work/pyshp')
import shapefile

import geopandas
import contextily as cx
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

#import ODiN2pd
import ODiN2readpkl

os.system("pip install geopandas")

os.system("pip install contextily")

import geopandas
import contextily as cx
plt.rcParams['figure.figsize'] = [10, 6]

# +
#nog te leren van https://github.com/VNG-Realisatie/grenswijzigen

# +
#TODO CBS lezen arbeidsplaatsen en inwoners
#TODO CBS relateren arbeidsplaatsen en inwoners aan BAG getallen

# +
#niet meer gebruikt
#wijkendata = shapefile.Reader("../data/CBS/wijken_2023_v1.dbf")

# +
#wijkendata.fields
# -

#eerste read is alleen nodig om kolom namen op te pakken
data_pc4_2020 = ODiN2readpkl.getpc4stats(2020)
data_pc4_2020

data_pc4_2022 = ODiN2readpkl.getpc4stats(2022)
data_pc4_2022

data_pc4_2022.dtypes

dat_83504 = pd.read_csv('../data/CBS/83504NED/Observations.csv',sep=';')
dat_83504_mc = pd.read_csv('../data/CBS/83504NED/MeasureCodes.csv',sep=';')

dat_83504.dtypes

dat_83504_mc.dtypes

dat_83504_mc

dat_85560 = pd.read_csv('../data/CBS/85560NED/Observations.csv',sep=';')
dat_85560_mc = pd.read_csv('../data/CBS/85560NED/MeasureCodes.csv',sep=';')
dat_85560['Value'] = pd.to_numeric(dat_85560['Value'].str.replace(",","."))

dat_85560_mc

dat_85560_mc[dat_85560_mc['Title'].str.contains(r'innen')]

dat_85560

gemeentendata ,  wijkgrensdata ,    buurtendata = ODiN2readpkl.getgwb(2020)    

# +
#gemeentendata_c ,  wijkgrensdata_c ,    buurtendata_c = getgwb(2021)    

# +
#print(wijkgrensdata.crs.to_string() )
#gemeentendata.to_crs(epsg=3857)
# -

buurtendata.dtypes

print(gemeentendata.dtypes)

print(gemeentendata.columns)

#Okay: postcode is alleen soort van indicator
recspb =buurtendata[['BU_CODE','BU_NAAM']].groupby(['BU_CODE']).count().reset_index()
recspb[recspb['BU_NAAM']>1]

#examples from https://geopandas.org/en/stable/docs/user_guide/mapping.html
buurtendata.plot()

buurtendata.plot(column="BEV_DICHTH",legend=True, cmap='OrRd')

gemset0 = ['Houten']
gemset1 = ['Houten','Bunnik','Nieuwegein']
gemset2 = ['Houten','Bunnik','Nieuwegein', 
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','Utrechtse Heuvelrug',
          'De Bilt']
gemset3 = ['Houten','Bunnik','Nieuwegein', 'Vijfheerenlanden',
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','De Bilt','Utrechtse Heuvelrug',
           'Stichtse Vecht' ]
gemset4 = ['Houten','Bunnik','Nieuwegein', 'Vijfheerenlanden',
          'Utrecht','Culemborg','Wijk bij Duurstede','Zeist','De Bilt','Utrechtse Heuvelrug',
           'Stichtse Vecht' ,'IJsselstein','Baarn','Soest', 'Amersfoort',
           'Leusden','Doorn', 'Bunschoten','Eemnes','Lopik', 'Montfoort',
          'Oudewater', 'Renswoude','Rhenen','De Ronde Venen', 'Stichtse Vecht',
          'Veenendaal', 'Woerden','Woudenberg',
          'Gooise Meren','Hilversum','Huizen','Laren','Wijdemeren','Blaricum',
          'Culemborg','Buren','Neder-Betuwe','Scherpenzeel','Tiel','West Betuwe']
gemset5=['De Ronde Venen']
gemset = gemset4
buurtdata_sel=buurtendata[buurtendata['GM_NAAM'].isin (gemset)]
gemdata_sel=gemeentendata[gemeentendata['GM_NAAM'].isin(gemset)]
wijkgrdata_sel=wijkgrensdata[wijkgrensdata['GM_NAAM'].isin(gemset)]

base=wijkgrdata_sel.boundary.plot(color='green');
#cx.add_basemap(base, crs=wijkgrdata_sel.crs.to_string(), source=cx.providers.Stamen.TonerLite)

#base=gemdata_sel.boundary.plot(color='green');
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_sel.plot(ax=base ,column="BEV_DICHTH",legend=True, cmap='OrRd',
                  legend_kwds={"label": "Bevolkingsdichtheid"})

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
#buurtdata_sels=buurtdata_sel[buurtdata_sel['STED'] !=5]
#bij 1000 doen Schaltwijk en 't Goy ook nog mee
#bij 2000 valen Bunnik en 't Goy af
#bij 3000 valt ook de Molen af
buurtdata_sels=buurtdata_sel[buurtdata_sel['BEV_DICHTH'] >3500]
buurtdata_sels.plot(ax=base ,column="STED",legend=True, cmap='RdYlBu',
                  legend_kwds={"label": "Stedelijkheid"})

buurtdata_selfield='BU_CODE'
adcol01= ['D000025','A045790_3','A045791_3','A045792_5']
def adddta(df1,df2,addmeas):
    df1s=df1[['geometry','BU_CODE','BU_NAAM','GM_NAAM','AANT_INW','BEV_DICHTH']]
    df2flds = df2[df2['Measure'].isin(addmeas)]    
    df2tab= df2flds.pivot_table(index='WijkenEnBuurten', columns='Measure',values='Value')
    seljoin = df1s.merge(df2tab,left_on=buurtdata_selfield, right_on='WijkenEnBuurten' ,how="left")    
    return(seljoin)
buurtdata_withfield01= adddta(buurtdata_sels,dat_85560,adcol01)

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield01.plot(ax=base,column=adcol01[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield01['SUPER_LOOP'] = buurtdata_withfield01[adcol01[1]] !=0
buurtdata_withfield01.plot(ax=base,column='SUPER_LOOP',legend=True, cmap='RdYlBu',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

super_loop_selgem = buurtdata_withfield01.groupby(['GM_NAAM','SUPER_LOOP']).agg({'AANT_INW':'sum'}).reset_index()
super_loop_selgem.to_excel('../output/super_loop_selgem01.xlsx')
super_loop_selgem

base=gemdata_sel.boundary.plot(color='green');
base.set_axis_off();
nietloop = buurtdata_withfield01[buurtdata_withfield01 [adcol01[1]] ==0 ]
nietloop.plot(ax=base,column=adcol01[1],legend=True, cmap='RdYlBu',
             legend_kwds={"label": "Aantal grote supermarkt < 1 km"}
           )

nietloop.sort_values(by='BEV_DICHTH').reset_index()['BEV_DICHTH'].plot()

nietloop.sort_values(by='AANT_INW').reset_index()

adcol02= ['D000038', 'A045790_4', 'A045791_4','A045792_6' ]
buurtdata_withfield02= adddta(buurtdata_sel,dat_85560,adcol02)
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield02.plot(ax=base,column=adcol02[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal dagelijkse levensmiddelen < 1 km"}
           )          

adcol03= ['D000020',
'A045790_5',
'A045791_5',
'A045792_8',
'D000021',
'A045790_6',
'A045791_6',
'A045792_9',
'D000043',
'A045790_7',
'A045791_7',
'A045792_10' ]
buurtdata_withfield03= adddta(buurtdata_sel,dat_85560,adcol03)
base=wijkgrdata_sel.boundary.plot(color='green');
base.set_axis_off();
buurtdata_withfield03.plot(ax=base,column=adcol03[1],legend=True, cmap='OrRd',
             legend_kwds={"label": "Aantal cafes < 1 km"}
           )          

pc6hnryr =ODiN2readpkl.getpc6hnryr(2020) 
pc6hnryr.dtypes

pc6hnryr[pc6hnryr['PC4']==3992]

#this frame does not contain geo info
if 0==1:
    pc6hnryr.plot(ax=base,column='Huisnummer',legend=True, cmap='OrRd',
             legend_kwds={"label": "Huisnummer"}
           )          

#2023 werkt nog niet
for year in range(2018,2023):
    pc6hnryr =ODiN2readpkl.getpc6hnryr(year) 
    print(pc6hnryr.dtypes)

import rasteruts1
import rasterio
calcgdir="../intermediate/calcgrids"


# +
def setaxhtn(ax):
    ax.set_xlim(left=137000, right=143000)
    ax.set_ylim(bottom=444000, top=452000)
    
def setaxutr(ax):
    ax.set_xlim(left=113000, right=180000)
    ax.set_ylim(bottom=480000, top=430000)


# -

buurtendata=buurtendata.reset_index()

lasttifname=calcgdir+'/cbsbuurtin-NL.tif'
buurtinwgrid = rasteruts1.createNLgrid(100,lasttifname,8,'')

dfrefs= rasteruts1.makegridcorr (buurtendata,buurtinwgrid)

buurtendata['area_geo'] = buurtendata.area

#veel niet gevonden uit landelijk !
missptdf= rasteruts1.findmiss(buurtendata,dfrefs)

import seaborn
seaborn.scatterplot(data=buurtendata,x='area_geo',y='area_pix')

buurtexcols= ['AANT_INW','AANT_MAN', 'AANT_VROUW']
imagelst=rasteruts1.mkimgpixavgs(buurtinwgrid,dfrefs,False,False,
                                 buurtendata[buurtexcols])  

buurtinwgrid.close()

buurtinwgrid = rasterio.open(lasttifname)

rogirdtifname=calcgdir+'/oriTN2-NL.tif'
rofinwgrid = rasterio.open(rogirdtifname)

fig, ax = plt.subplots()
base=buurtendata.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((buurtinwgrid,2), cmap='OrRd',ax=ax)
rasteruts1.setaxutr(ax)

fig, ax = plt.subplots()
base=buurtendata.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((rofinwgrid,2), cmap='OrRd',ax=ax)
rasteruts1.setaxutr(ax)

sambuurt1= np.abs( buurtinwgrid.read(2) - rofinwgrid.read(2))<1.5+rofinwgrid.read(2)/10
plt.imshow(sambuurt1,origin='lower')
plt.colorbar()

difbuurtsize=  buurtinwgrid.read(2) + 0*rofinwgrid.read(2)
plt.imshow(difbuurtsize,origin='lower')
plt.colorbar()

fig, ax = plt.subplots()
base=buurtendata.boundary.plot(color='green',ax=ax,alpha=.3);
rasterio.plot.show((buurtinwgrid,5),cmap='Reds',ax=ax,alpha=0.5)
rasterio.plot.show((buurtinwgrid,4),cmap='Blues',ax=ax,alpha=0.5)
setaxutr(ax)

buurtimsums = list(np.sum(buurtinwgrid.read(i+2)[False == np.isnan(buurtinwgrid.read(i+2))]) for i in range(4)) 
buurtimsums

print(np.sum(buurtendata[buurtexcols]))

from sklearn import linear_model
buurtcorrM = (False == np.isnan(buurtinwgrid.read(1+2))) & \
               sambuurt1 & (rofinwgrid.read(3) >0) & ( buurtinwgrid.read(1+2) >0)
columns=['Inwoners','WoonOppervlak']
buurtcorr= pd.DataFrame( np.array ( (buurtinwgrid.read(1+2)[buurtcorrM ],
                                      rofinwgrid.read(3)[buurtcorrM ] ) ), index=columns).T 
buurtcorr.dtypes

lm = linear_model.LinearRegression(fit_intercept=True)
model = lm.fit(np.log(buurtcorr['WoonOppervlak'].values.reshape(-1, 1)),
               np.log(buurtcorr['Inwoners'].values.reshape(-1, 1))) 
#lm.fit(buurtcorr['WoonOppervlak'],buurtcorr['Inwoners']) 
buurtcorr['Predict']=np.exp(model.predict(np.log(buurtcorr['WoonOppervlak'].values.reshape(-1, 1))))
buurtcorr['Predictu']=buurtcorr['Predict']*np.exp(1)
buurtcorr['Predictl']=buurtcorr['Predict']*np.exp(-1)
print( model.coef_, model.intercept_)
fig, ax = plt.subplots(figsize=(6, 4))
seaborn.lineplot(data=buurtcorr,x='WoonOppervlak',y='Predict', color='g',ax=ax)
seaborn.lineplot(data=buurtcorr,x='WoonOppervlak',y='Predictu', color='r',ax=ax)
seaborn.lineplot(data=buurtcorr,x='WoonOppervlak',y='Predictl', color='r',ax=ax)
seaborn.scatterplot(data=buurtcorr,x='WoonOppervlak',y='Inwoners',ax=ax)
ax.set_xscale('log')
ax.set_yscale('log')

buurtratioa =  np.where(buurtcorrM , np.log(buurtinwgrid.read(1+2)) - \
                        (np.log(rofinwgrid.read(3)) * model.coef_ + model.intercept_ ),\
                        np.nan)
buurtratio =  np.where(abs(buurtratioa)< 1,buurtratioa,np.nan)
plt.imshow(buurtratio,origin='lower', cmap='plasma')
plt.colorbar()

buurtratiox =  np.where(abs(buurtratioa)> 1,buurtratioa,np.nan)
plt.imshow(buurtratiox, origin='lower', cmap='plasma')
plt.colorbar()

plt.hist(buurtratio[~ np.isnan(buurtratio)])

print("Finished")

# +
#alle transformaties zijn voorbereid per wijk. Nu naar voorzieningen kijken

# +
#eerst voorzienigen reeken, ook als validatie methode van toewijzen aan een wijk
# -

measlist =dat_85560_mc
#print(measlist)
cidx=np.array(measlist.Index)
#print (cidx)
bcolnm = np.array(buurtendata.columns)[[cidx]]
#print(bcolnm)
print(np.array([cidx,bcolnm, measlist.Identifier] ).T)

bcolnm = pd.DataFrame( (buurtendata.columns),columns=["ColnmBu"] )
bcolnm['Index'] = bcolnm.index
bcolnm.to_excel("../output/chkblox.xlsx")
#print(np.array([cidx,bcolnm, measlist.Identifier] ).T)

np.isnan(buurtendata).agg('sum')

bcol2 = pd.DataFrame( (buurtendata.dtypes),columns=["ColTyp"] )
bcol2['ColnmBu'] = buurtendata.columns
oktyps=[np.float64,np.int64]
#print(bcol2)
bcol2s= bcol2[True== (bcol2['ColTyp'].isin(oktyps) )]
bcol2n= buurtendata.loc[:,bcol2s['ColnmBu'] ]
#print(bcol2n)
bcol2nan=  pd.DataFrame( (( bcol2n).isna()==False).agg('sum'),columns=["NumNan"] )
bcol2nan['ColnmBu'] = bcol2nan.index
print(bcol2nan)
bcol3 =bcol2.merge(bcol2nan)
bcol3
bcol3.to_excel("../output/chkblox.xlsx")

# +
#dit zijn velden met de gemiddelde atstanden tot het DICHTSTBIJZIJNDE object
#dat houdt in dat als het NIET in de wijk is, de kortste afstanden zullen beginnen
#met halve wijk afstand minus deze gemiddelde afstand
#dat houdt in dat als het WEL in de wijk is, de aanname kan zijn dat ca de helft van de
#inwoners er op die afstand kan zijn.
#nog steeds kan er voorkeur zijn voor grotere / ander merk
#aanname kan ook zijn dat bezoek aan DAGLM op houdt SUPERM begint

# +
#plan2: schrijf NIET genormeerd veld met gemiddelde afstanden
# smooth dit (samen met vetd index !=0) met een breed gaussisch grid (5 km breed, gaauss =200m)
# aan de hand van deze waarden zijn dan gebruiksbanden te bepalen
# bijv: winkel start 500 mtr voor korste tot 2 * gem afstand,
# en dan lineair oplopend met afstandsbin in die band (=1 > afstand)
# -

smsc1 = np.array(buurtendata.columns[buurtendata.columns.str.contains('AV3')].str.replace('AV3',''))
print(smsc1)
smscs= smsc1[1:3]
smscs

lmsc1 = np.array(buurtendata.columns[buurtendata.columns.str.contains('AV20')].str.replace('AV20',''))
lmsc2 = np.array(buurtendata.columns[buurtendata.columns.str.contains('AV20')].str.replace('AV20','_'))
print(lmsc1)

amsc1 = np.array(buurtendata.columns[buurtendata.columns.str.contains('AF_')].str.replace('AF_','_'))
mmsc1=np.concatenate((smsc1,lmsc1,lmsc2))
rmsc1 =amsc1[False==np.isin(amsc1,mmsc1)]
print(rmsc1)

feattstgrd=calcgdir+'/cbsfeattst.tif'
smftg1 = rasteruts1.createNLgrid(100,lasttifname,len(smscs)*8,'')

buurtendata['center']= buurtendata.representative_point()


# +
#now some kernel and transformations
#note: for gaussian: ony can do 2 1d convolutions
def roundfilt(gridstep,dist):
    maxkernrng= int(dist/gridstep+2)*gridstep
    x = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    y = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    X, Y = np.meshgrid(x, y)
    Z = np.int8(np.sqrt(X*X+Y*Y) <dist)
    return Z

print(roundfilt(100,660) )

#example fietskern1= kernelfietspara()
#example dcalc=(fietskern1['Z2D']-fietskern1['Z'])
#example print(np.max(np.abs(dcalc)) )

# +
#read back points, using rasterio directly

import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

#@jit
def addobjvals(img,coords,values):
    for obj in range(coords.shape[0]): 
        img[coords[obj,1],coords[obj,0]] += values[obj]
        
#@jit
def getobjvals(img,coords,values):
    for obj in range(coords.shape[0]): 
        values[obj]= img[coords[obj,1],coords[obj,0]]         

def gridoncenters (grid,r1):
    corrgrid = np.zeros([grid.width, grid.height],dtype=np.int32)    
    addobjvals(corrgrid,r1,range(len(r1)))
    valtst = np.zeros (len(r1),dtype=np.int32)
    getobjvals(corrgrid,r1,valtst)
    print(valtst[abs(valtst - range(len(r1)) ) > 1e-6])
    return corrgrid

def centergridcoords (grid,ctrser):
    r1= np.array(rasterio.transform.rowcol(grid.transform,xs=ctrser.x,ys=ctrser.y)).T
    return r1
    
ctrxform = centergridcoords (smftg1,buurtendata[ "center"]) 
gridoncenters (smftg1,ctrxform)
# -



# +
def mktagrds(grid,bdf,selbase,r1):
    fig, ax = plt.subplots(figsize=(6, 4))
    bdf['geoavgdist'] = np.sqrt(bdf['area_geo']/2000000 )
    filter1km=roundfilt(100,660) 
    for tag in selbase:
        f1km="AV1"+tag
        f3km="AV3"+tag
        f5km="AV5"+tag
        af  ="AF"+tag
        kpf1=bdf[bdf[f1km]>0]        
        #seaborn.scatterplot(data=kpf1,x='geoavgdist',y=af,ax=ax)
        hasbuurt= np.float32(gridoncenters (grid,r1) !=0)
        aantbuurfd= rasteruts1.convfiets2d(hasbuurt,filter1km)
        nbuurt1km = np.zeros (len(r1),dtype=np.int32)
        getobjvals(aantbuurfd,r1,nbuurt1km)
        bdf['nbuurt1km']=nbuurt1km
#        print(bdf[bdf['nbuurt1km']>1])
#'nbuurt1km',
        stats1km=bdf.groupby([f3km])[af].agg('count')
        print(stats1km)
        

mktagrds(smftg1,buurtendata,smscs,ctrxform)
# -


