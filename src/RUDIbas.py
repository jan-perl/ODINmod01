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
#model basics
# load every time for generic constants
#loads only fast, standard libraries
#also defines testing properties for using with workbooks
# -

import numpy as np
import pandas as pd

suprtests=['RUDIbas']
suprdata=['RUDIbas']
myname='RUDIbas'

# +
#nu ODIN ranges opzetten
#we veranderen NIETS aan odin data
#wel presenteren we het steeds als cumulatieve sommen tot een bepaalde bin
# -

if ()
    useKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01uKA.pkl")
    xlatKAfstVa=pd.read_pickle("../intermediate/ODINcatVN01xKA.pkl")
    useKAfstV  = useKAfstVa [useKAfstVa ["MaxAfst"] <180].copy()
    maxcuse= np.max(useKAfstV[useKAfstV ["MaxAfst"] !=0] ['KAfstCluCode'])
    xlatKAfstV  = xlatKAfstVa [(xlatKAfstVa['KAfstCluCode']<=maxcuse ) |
                               (xlatKAfstVa['KAfstCluCode']==np.max(useKAfstV[ 'KAfstCluCode']) )].copy()
    #print(xlatKAfstV)   
    print(useKAfstV)   

#dit was alleen voor ODIN1KAFmo om met kleine sets te werken.
#deze variabele niet gebruiken voor verwerken hele sets, wel voor regressie tests op Q set
useKAfstVQ  = useKAfstV [useKAfstV ["MaxAfst"] <4]
