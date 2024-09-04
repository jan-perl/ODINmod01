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

import ODIN1lKAfmo

import glob
import re

globset="e0904a"
flst = glob.glob ("../intermediate/addgrds/"+globset+"*.tif")
elst = list(re.sub(".tif$",'',re.sub('^.*/','',f) ) for f in flst) 
elst

stQ = ODIN1lKAfmo.grosres (elst,ODIN1lKAfmo.rudifungcache,1,ODIN1lKAfmo.fitpara,ODIN1lKAfmo.fitdatverplgr,
                                ODIN1lKAfmo.useKAfstVQ,'Set03Q-'+globset ) 

stN = ODIN1lKAfmo.grosres (elst,ODIN1lKAfmo.rudifungcache,1,ODIN1lKAfmo.fitpara,ODIN1lKAfmo.fitdatverplgr,
                                ODIN1lKAfmo.useKAfstV,'Set03N-'+globset )  

print ("Finished")


