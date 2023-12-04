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

import pandas as pd
import numpy as np
import os as os

# +
#TODO hernoem wat kolommen

# +
#TODO parse ook data labels

# +
#nu postcode match hulptabel
# -

allodinyr =pd.read_pickle("../intermediate/allodinyr.pkl")

dbk_allyr_cols= pd.read_pickle("../intermediate/dbk_allyr_cols.pkl")

fietswijk1pc4=pd.read_pickle("../intermediate/fietswijk1pc4.pkl")

#note: to clean: copied in ODiN2pd
excols= ['Wogem', 'AutoHhl', 'MRDH', 'Utr', 'FqLopen', 'FqMotor', 'WrkVervw', 'WrkVerg', 'VergVast', 
         'VergKm', 'VergBrSt', 'VergOV', 'VergAans', 'VergVoer', 'VergBudg', 'VergPark', 'VergStal', 'VergAnd', 
         'BerWrk', 'RdWrkA', 'RdWrkB', 'BerOnd', 'RdOndA', 'RdOndB', 'BerSup', 'RdSupA', 'RdSupB',
         'BerZiek', 'RdZiekA', 'RdZiekB', 'BerArts', 'RdArtsA', 'RdArtsB', 'BerStat', 'RdStatA', 'RdStatB', 
         'BerHalte', 'RdHalteA', 'RdHalteB', 'BerFam', 'RdFamA', 'RdFamB', 'BerSport', 'RdSportA', 'RdSportB',
          'VertMRDH', 'VertUtr', 'AankMRDH', 'AankUtr' ]


