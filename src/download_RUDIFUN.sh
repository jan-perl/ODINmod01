#!/bin/bash
# +
#get the 2 RUDIFUN dataasets from PBL
#description of data at https://www.pbl.nl/publicaties/rudifun-2024
cd ../data
for od in RUDIFUN2 RUDIFUN_2024
do
l1=https://dataportaal.pbl.nl/$od
mkdir $od
cd $od
wget -q -O index.html $l1
grep -e href index.html | sed -e 's+^.*href=.http+http+' -e 's+".*$++' | grep -e ^http | while read a
do 
  echo $a
  wget -q -N $a
done

for t in *.zip
do
   zd=$(echo $t | sed -e 's/.zip$//')
   mkdir $zd
   unzip  -o $t -d $zd
done

cd ..
done

