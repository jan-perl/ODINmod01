#!/bin/bash
#file  ../DANS-API-Token.txt should contain API key
grep -v -e "^#" ../inputs/DANSlnks01.txt|head -100 | while read dset link
do
l2=$(echo $link | sed -e 's+dataset.xhtml+api/access/dataset/:persistentId/+')
curl -L -O -J -H "X-Dataverse-key:$(cat ../../DANS-API-Token.txt)"   $l2
mv dataverse_files.zip ../data/dnload_$dset.zip
mkdir ../data/$dset ../data/$dset/migration-info
unzip -o ../data/dnload_$dset.zip -d ../data/$dset
unzip -o ../data/$dset/easy-migration.zip -d ../data/$dset/migration-infp
grep -B1 \.sav ../data/$dset/migration-infp/easy-migration/files.xml | grep dct:identifier > ../data/$dset/savids.txt
 cat ../data/$dset/savids.txt
l3=$(echo $link | sed -e 's+dataset.xhtml+api/access/datafile/:persistentId+')
echo "Getting $l3"
#curl -L -O -J -H "X-Dataverse-key:$(cat ../../DANS-API-Token.txt)"   $l3
#werkte niet, csvs met hand gedownload
done
cd ../data
curl -L -O -J https://www.cbs.nl/-/media/_excel/2020/39/2020-cbs-pc6huisnr20200801-buurt.zip
unzip -o 2020-cbs-pc6huisnr20200801-buurt.zip

