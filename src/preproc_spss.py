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

# +
#system(pip install --upgrade pip setuptools wheel)

# +
#system(pip install spss-converter)

# +
#to be run as tooy inside container: docker exec -u 0 -it jupyter03 bash
#system(apt-get install libz-dev)

# +
#system(pip uninstall pyreadstat)

# +
#system(pip install pyreadstat==0.3.4)
# -

#system(pip install spss-converter)
print (pd.__version__) 

df_2018 = pd.read_csv("../data/ODiN2018_Databestand_v2.0.csv", encoding = "ISO-8859-1", sep=";")  
df_2019 = pd.read_csv("../data/ODiN2019_Databestand_v2.0.csv", encoding = "ISO-8859-1", sep=";")  
df_2020 = pd.read_csv("../data/ODiN2020_Databestand_v2.0.csv", encoding = "ISO-8859-1", sep=";")  
df_2021 = pd.read_csv("../data/ODiN2021_Databestand.csv", encoding = "ISO-8859-1", sep=";")  
df_2022 = pd.read_csv("../data/ODiN2021_Databestand.csv", encoding = "ISO-8859-1", sep=";")  

df_2019


