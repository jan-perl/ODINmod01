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
#leest geconverteerde pickle data tabellen
#goede basis voor verdere berekeningen
# -

import pandas as pd
import numpy as np
import os as os
import io as io

# +
#TODO hernoem wat kolommen

# +
#TODO parse ook data labels

# +
#nu postcode match hulptabel
# -

allodinyr =pd.read_pickle("../intermediate/allodinyr.pkl")

dbk_allyr= pd.read_pickle("../intermediate/dbk_allyr.pkl")

fietswijk1pc4=pd.read_pickle("../intermediate/fietswijk1pc4.pkl")

#note: to clean: copied in ODiN2pd
excols= ['Wogem', 'AutoHhl', 'MRDH', 'Utr', 'FqLopen', 'FqMotor', 'WrkVervw', 'WrkVerg', 'VergVast', 
         'VergKm', 'VergBrSt', 'VergOV', 'VergAans', 'VergVoer', 'VergBudg', 'VergPark', 'VergStal', 'VergAnd', 
         'BerWrk', 'RdWrkA', 'RdWrkB', 'BerOnd', 'RdOndA', 'RdOndB', 'BerSup', 'RdSupA', 'RdSupB',
         'BerZiek', 'RdZiekA', 'RdZiekB', 'BerArts', 'RdArtsA', 'RdArtsB', 'BerStat', 'RdStatA', 'RdStatB', 
         'BerHalte', 'RdHalteA', 'RdHalteB', 'BerFam', 'RdFamA', 'RdFamB', 'BerSport', 'RdSportA', 'RdSportB',
          'VertMRDH', 'VertUtr', 'AankMRDH', 'AankUtr' ]

# +
largranval = -9999999999
ODINmissint = -99997 
def mkspecvaltab(indbk): 
    lastlbl=''
    nrec=int(len(indbk))
    c0=[]
    c1=[]
    c2=[]
    for irec in range(0,nrec):
        nxtlbl=indbk.iloc[[irec]]
#        print (nxtlbl)
        if nxtlbl['Variabele_naam_ODiN_2022'].isna().item():
            if ".." in str(nxtlbl['Code_ODiN_2022'].item()):
                vrng = str(nxtlbl['Code_ODiN_2022'].item()).split("..")
                vrng= [int(vrng[0]),int(vrng[1])+1]
                if (vrng[1] - vrng[0]) >15:
                    print ('Setting (unchecked) large range',vrng,lastlbl,
                           nxtlbl['Code_label_ODiN_2022'].item())
                    c0.append(lastlbl)
                    c1.append(largranval)
                    c2.append(nxtlbl['Code_label_ODiN_2022'].item() )
                else:
                    for num in range(vrng[0],vrng[1]):
                        c0.append(lastlbl)
                        c1.append(num)
                        c2.append(num)
            else:
                c0.append(lastlbl)
                c1.append(nxtlbl['Code_ODiN_2022'].item())
                c2.append(nxtlbl['Code_label_ODiN_2022'].item() )
        else:
            lastlbl=nxtlbl['Variabele_naam_ODiN_2022'].item()
    outcol_names =  ['Variabele_naam', 'Code', 'Code_label'] 
    outdf=pd.DataFrame(list(zip(c0,c1,c2)),columns=outcol_names)
    return(outdf)

specvaltab = mkspecvaltab(dbk_allyr)
#specvaltab
# -
some_string="""dbfname,fieldname
BU_CODE,buurtcode
BU_NAAM,buurtnaam
WK_CODE,wijkcode
WK_NAAM,wijknaam
GM_CODE,gemeentecode
GM_NAAM,gemeentenaam
IND_WBI,indelingswijziging_wijken_en_buurten
H2O,water
POSTCODE,meest_voorkomende_postcode
DEK_PERC,dekkingspercentage
OAD,omgevingsadressendichtheid
STED,stedelijkheid_adressen_per_km2
BEV_DICHTH,bevolkingsdichtheid_inwoners_per_km2
AANT_INW,aantal_inwoners
AANT_MAN,mannen
AANT_VROUW,vrouwen
P_00_14_JR,percentage_personen_0_tot_15_jaar
P_15_24_JR,percentage_personen_15_tot_25_jaar
P_25_44_JR,percentage_personen_25_tot_45_jaar
P_45_64_JR,percentage_personen_45_tot_65_jaar
P_65_EO_JR,percentage_personen_65_jaar_en_ouder
P_ONGEHUWD,percentage_ongehuwd
P_GEHUWD,percentage_gehuwd
P_GESCHEID,percentage_gescheid
P_VERWEDUW,percentage_verweduwd
AANTAL_HH,aantal_huishoudens
P_EENP_HH,percentage_eenpersoonshuishoudens
P_HH_Z_K,percentage_huishoudens_zonder_kinderen
P_HH_M_K,percentage_huishoudens_met_kinderen
GEM_HH_GR,gemiddelde_huishoudsgrootte
P_NL_ALL,percentage_met_herkomstland_nederland
P_EUR_ALL,percentage_met_herkomstland_uit_europa_excl_nl
P_NEU_ALL,percentage_met_herkomstland_buiten_europa
P_GEBNL_NL,percentage_geb_in_nl_met_herkomstland_nederland
P_GEBNL_EU,perc_geb_in_nl_met_herkomstland_in_europa_ex_nl
P_GEBNL_NE,perc_geb_in_nl_met_herkomstland_buiten_europa
P_GEBBL_EU,perc_geb_buiten_nl_met_herkomstlnd_in_europa_ex_nl
P_GEBBL_NE,perc_geb_buiten_nl_met_herkomstlnd_buiten_europa
OPP_TOT,oppervlakte_totaal_in_ha
OPP_LAND,oppervlakte_land_in_ha
OPP_WATER,oppervlakte_water_in_ha
JRSTATCODE,jrstatcode
JAAR,jaar"""
#read CSV string into pandas DataFrame
wgrename= pd.read_csv(io.StringIO(some_string), sep=",")
#display(wgrename)

some_string="""fieldname
geboorte_totaal
geboortes_per_1000_inwoners
sterfte_totaal
sterfte_relatief
aantal_bedrijven_landbouw_bosbouw_visserij
aantal_bedrijven_nijverheid_energie
aantal_bedrijven_handel_en_horeca
aantal_bedrijven_vervoer_informatie_communicatie
aantal_bedrijven_financieel_onroerend_goed
aantal_bedrijven_zakelijke_dienstverlening
aantal_bedrijven_overheid_onderwijs_en_zorg
aantal_bedrijven_cultuur_recreatie_overige
aantal_bedrijfsvestigingen
woningvoorraad
gemiddelde_woningwaarde
percentage_eengezinswoning
percentage_meergezinswoning
percentage_bewoond
percentage_onbewoond
percentage_koopwoningen
percentage_huurwoningen
perc_huurwoningen_in_bezit_woningcorporaties
perc_huurwoningen_in_bezit_overige_verhuurders
percentage_woningen_met_eigendom_onbekend
percentage_bouwjaarklasse_tot_2000
percentage_bouwjaarklasse_vanaf_2000
gemiddeld_aardgasverbruik
gemiddeld_gasverbruik_appartement
gemiddeld_gasverbruik_tussenwoning
gemiddeld_gasverbruik_hoekwoning
gemiddeld_gasverbruik_2_onder_1_kap_woning
gemiddeld_gasverbruik_vrijstaande_woning
gemiddeld_gasverbruik_huurwoning
gemiddeld_gasverbruikkoopwoning
gemiddelde_elektriciteitslevering
gemiddeld_elektriciteitsverbruik_appartement
gemiddeld_elektriciteitsverbruik_tussenwoning
gemiddeld_elektriciteitsverbruik_hoekwoning
gem_elektriciteitsverbruik_2_onder_1_kap_woning
gem_elektriciteitsverbruik_vrijstaande_woning
gemiddeld_elektriciteitsverbruik_huurwoning
gemiddeld_elektriciteitsverbruikkoopwoning
percentage_woningen_met_stadsverwarming
aantal_leerlingen_primair_onderwijs
aantal_leerlingen_voortgezet_onderwijs
aantal_studenten_mbo
aantal_studenten_hbo
aantal_studenten_wo
aantal_personen_met_bvm_als_hoogst_beh_ond_niv
aantal_personen_met_hvm_als_hoogst_beh_ond_niv
aantal_personen_met_hw_als_hoogst_beh_ond_niv
netto_arbeidsparticipatie
percentage_werknemers
percentage_zelfstandigen
aantal_pers_werkzame_beroepsbevolking
percentage_werknemers_met_vaste_arbeidsrelatie
percentage_werknemers_met_flexibele_arbeidsrelatie
aantal_inkomensontvangers
gemiddeld_inkomen_per_inkomensontvanger
gemiddeld_inkomen_per_inwoner
percentage_personen_met_laag_inkomen
percentage_personen_met_hoog_inkomen
percentage_huishoudens_met_laag_inkomen
percentage_huishoudens_met_hoog_inkomen
percentage_huishoudens_onder_of_rond_sociaal_minimum
percentage_huishoudens_met_lage_koopkracht
aantal_personen_met_een_ao_uitkering_totaal
aantal_personen_met_een_ww_uitkering_totaal
aantal_personen_met_een_alg_bijstandsuitkering_tot
aantal_personen_met_een_aow_uitkering_totaal
gemiddeld_gestandaardiseerd_inkomen_van_huishoudens
huishoudens_tot_110_percent_van_sociaal_minimum
huishoudens_tot_120_percent_van_sociaal_minimum
mediaan_vermogen_van_particuliere_huish
aantal_jongeren_met_jeugdzorg_in_natura
percentage_jongeren_met_jeugdzorg_in_natura
aantal_wmo_clienten
aantal_wmo_clienten_per_1000_inwoners
personenautos_totaal
personenautos_per_huishouden
personenautos_per_km2
motortweewielers_totaal
aantal_personenautos_met_brandstof_benzine
aantal_personenautos_met_overige_brandstof
huisartsenpraktijk_gemiddelde_afstand_in_km
huisartsenpraktijk_gemiddeld_aantal_binnen_1_km
huisartsenpraktijk_gemiddeld_aantal_binnen_3_km
huisartsenpraktijk_gemiddeld_aantal_binnen_5_km
huisartsenpost_gemiddelde_afstand_in_km
apotheek_gemiddelde_afstand_in_km
ziekenhuis_incl_buitenpolikliniek_gem_afst_in_km
ziekenhuis_incl_buitenpoli_gem_aantal_binnen_5_km
ziekenhuis_incl_buitenpoli_gem_aantal_binnen_10_km
ziekenhuis_incl_buitenpoli_gem_aantal_binnen_20_km
ziekenhuis_excl_buitenpolikliniek_gem_afst_in_km
ziekenhuis_excl_buitenpoli_gem_aantal_binnen_5_km
ziekenhuis_excl_buitenpoli_gem_aantal_binnen_10_km
ziekenhuis_excl_buitenpoli_gem_aantal_binnen_20_km
grote_supermarkt_gemiddelde_afstand_in_km
grote_supermarkt_gemiddeld_aantal_binnen_1_km
grote_supermarkt_gemiddeld_aantal_binnen_3_km
grote_supermarkt_gemiddeld_aantal_binnen_5_km
winkels_ov_dagelijkse_levensm_gem_afst_in_km
winkels_ov_dagel_levensm_gem_aantal_binnen_1_km
winkels_ov_dagel_levensm_gem_aantal_binnen_3_km
winkels_ov_dagel_levensm_gem_aantal_binnen_5_km
warenhuis_gemiddelde_afstand_in_km
warenhuis_gemiddeld_aantal_binnen_5_km
warenhuis_gemiddeld_aantal_binnen_10_km
warenhuis_gemiddeld_aantal_binnen_20_km
cafe_gemiddelde_afstand_in_km
cafe_gemiddeld_aantal_binnen_1_km
cafe_gemiddeld_aantal_binnen_3_km
cafe_gemiddeld_aantal_binnen_5_km
cafetaria_gemiddelde_afstand_in_km
cafetaria_gemiddeld_aantal_binnen_1_km
cafetaria_gemiddeld_aantal_binnen_3_km
cafetaria_gemiddeld_aantal_binnen_5_km
restaurant_gemiddelde_afstand_in_km
restaurant_gemiddeld_aantal_binnen_1_km
restaurant_gemiddeld_aantal_binnen_3_km
restaurant_gemiddeld_aantal_binnen_5_km
hotel_gemiddelde_afstand_in_km
hotel_gemiddeld_aantal_binnen_5_km
hotel_gemiddeld_aantal_binnen_10_km
hotel_gemiddeld_aantal_binnen_20_km
kinderdagverblijf_gemiddelde_afstand_in_km
kinderdagverblijf_gemiddeld_aantal_binnen_1_km
kinderdagverblijf_gemiddeld_aantal_binnen_3_km
kinderdagverblijf_gemiddeld_aantal_binnen_5_km
buitenschoolse_opvang_gem_afstand_in_km
buitenschoolse_opvang_gemiddeld_aantal_binnen_1_km
buitenschoolse_opvang_gemiddeld_aantal_binnen_3_km
buitenschoolse_opvang_gemiddeld_aantal_binnen_5_km
basisonderwijs_gemiddelde_afstand_in_km
basisonderwijs_gemiddeld_aantal_binnen_1_km
basisonderwijs_gemiddeld_aantal_binnen_3_km
basisonderwijs_gemiddeld_aantal_binnen_5_km
voortgezet_onderwijs_gem_afstand_in_km
voortgezet_onderwijs_gemiddeld_aantal_binnen_3_km
voortgezet_onderwijs_gemiddeld_aantal_binnen_5_km
voortgezet_onderwijs_gemiddeld_aantal_binnen_10_km
vmbo_gemiddelde_afstand_in_km
vmbo_gemiddeld_aantal_binnen_3_km
vmbo_gemiddeld_aantal_binnen_5_km
vmbo_gemiddeld_aantal_binnen_10_km
havo_vwo_gemiddelde_afstand_in_km
havo_vwo_gemiddeld_aantal_binnen_3_km
havo_vwo_gemiddeld_aantal_binnen_5_km
havo_vwo_gemiddeld_aantal_binnen_10_km
brandweerkazerne_gemiddelde_afstand_in_km
oprit_hoofdverkeersweg_gemiddelde_afstand_in_km
treinstation_gemiddelde_afstand_in_km
overstapstation_gemiddelde_afstand_in_km
zwembad_gemiddelde_afstand_in_km
kunstijsbaan_gemiddelde_afstand_in_km
bibliotheek_gemiddelde_afstand_in_km
poppodium_gemiddelde_afstand_in_km
bioscoop_gemiddelde_afstand_in_km
bioscoop_gemiddeld_aantal_binnen_5_km
bioscoop_gemiddeld_aantal_binnen_10_km
bioscoop_gemiddeld_aantal_binnen_20_km
sauna_gemiddelde_afstand_in_km
zonnebank_gemiddelde_afstand_in_km
attractiepark_gemiddelde_afstand_in_km
attractiepark_gemiddeld_aantal_binnen_10_km
attractiepark_gemiddeld_aantal_binnen_20_km
attractiepark_gemiddeld_aantal_binnen_50_km
theater_gemiddelde_afstand_in_km
theater_gemiddeld_aantal_binnen_5_km
theater_gemiddeld_aantal_binnen_10_km
theater_gemiddeld_aantal_binnen_20_km
gemiddelde_afstand_tot_museum
gemiddeld_aantal_musea_binnen_5_km
gemiddeld_aantal_musea_binnen_10_km
gemiddeld_aantal_musea_binnen_20_km"""
newcbsfields= pd.read_csv(io.StringIO(some_string), sep=",")


# +
def getgwbxlat(year,dbfnames,plkf):
    stryear=str(year)    
    df=pd.read_pickle(plkf)
    if dbfnames & (year>=2023):
        adj=wgrename.set_index('fieldname').to_dict()['dbfname']
        #print(adj)
        df.columns = [ adj.get(x, x) for x in df.columns]
    elif ( not dbfnames) & (year<2023):
        adj=wgrename.set_index('dbfname').to_dict()['fieldname']
        #print(adj)
        df.columns = [ adj.get(x, x) for x in df.columns]
    return (df)

def getgwb(year,dbfnames=True):
    stryear=str(year)    
    g=getgwbxlat(year,dbfnames,"../intermediate/CBS/gwb_gem_"+stryear+".pkl")    
    w=getgwbxlat(year,dbfnames,"../intermediate/CBS/gwb_wijk_"+stryear+".pkl") 
    b=getgwbxlat(year,dbfnames,"../intermediate/CBS/gwb_buurt_"+stryear+".pkl") 
    return ([g,w,b])
#test code
gemeentendata ,  wijkgrensdata ,    buurtendata = getgwb(2023)    
# -


buurtendata.columns


def getpc4stats(year):
    stryear=str(year)    
    data_pc4=pd.read_pickle("../intermediate/CBS/pc4stats_"+stryear+".pkl")    
    return (data_pc4)
#test code
#data_pc4_2020 = getpc4stats(2020)


def getpc6hnryr(year):
    stryear=str(year)    
    ngrp=pd.read_pickle("../intermediate/CBS/pc6hnryr_"+stryear+".pkl") 
    return(ngrp)
pc6hnryr =getpc6hnryr(2020) 
pc6hnryr.dtypes


