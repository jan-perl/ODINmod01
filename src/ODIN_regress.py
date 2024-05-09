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

import os as os
import re as re
import pandas as pd

#maak lijst van nog niet regressie gereste R bestanden
cmd1="grep -e 'import ' *.py | sed -e 's+^.*[^#]import *++' -e 's+ as .*$++' -e 's+$+.py+' | sort -u | grep -v -e 'import' -e ' ' > ../output/called_R.txt"
#message(cmd1)    
#cmd1=re.sub('xxxxx','"',cmd1)
#message(cmd1)     
os.system(cmd1)
# !echo Filename > ../output/notcalled_R.txt
#r2=system(cmd2)
# !ls *.py | grep -v -f ../output/called_R.txt >> ../output/notcalled_R.txt
#r3=system(cmd3)

#a list of used sources
# #!cat ../output/called_R.txt
# !ls -l $(cat ../output/called_R.txt)

# +
# initialise data of lists.
notcallinit = {'Filename':['ODIN_regress.py','s3dataregen.R'], 'reason':["main to be called" ,        "main to be called"   ]}
OKnotvcalled = pd.DataFrame(notcallinit)

notcalled=pd.read_csv("../output/notcalled_R.txt")
notcalled_expl = notcalled .merge( OKnotvcalled,how='left')
print(notcalled_expl)
if (len(notcalled_expl [ notcalled_expl  ['reason'] .isna() ]) >0 ):
    print("Untested files found. Add calling to it to EMS_regress.R ")

#tabel should not have NA entries
# -

#old main file, should not be used anymore
import charging_main
import chargingProbability

# +
#to import ODIN data, run
#!./download_ODIN.sh
# -

#after downloading ODIN data, convertp to pkl
import ODiN2pd

#to download files from the CBS site, use
import getCBS

#after reading CBS data, convert to pkl
import CBS2pkl

#import ODiN2readpkl, ook gebruikt als subroutines
#
import ODiN2readpkl

#import conversion routines buurt to PC4/6, ook gebruikt als subrouteines
import viewCBS

#convert RUDIFUN to usable PC4 data
import cnvRUDIFUN

import prepareData

import preproc_spss
