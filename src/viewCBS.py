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



print("Finished")


