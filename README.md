# Predicting rain damage
Repository for the thesis of Teun de Mast

The goal of this project is predict rain damage and compare models that use different data sources. Use has been made of 2 data sources that provide historical instances of raindamages. One is the P2000 network, which is a network that is used by emergency services in The Netherlands. The other one is a dataset from a private insurance company in which insurance claims per district per day can be found. 

Some code is provided by Thijs Simons. His repository can be found here: <https://github.com/SimonsThijs/wateroverlast>. Some of his explanation in the README is also used in this README. 

## Structure

- The rain data is obtained from <https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=rad_nl25_rac_mfbs_5min&x-dataset-version=2.0>. Make sure the data is placed like: 'neerslag/data/{year}/{month}/RAD_NL25_RAC_MFBS_5min_XXXXXXXXXXXX_NL.h5'.

- AHN3 is used to obtain a height map of the area around the affected building. We use the public wcs from <https://nationaalgeoregister.nl> and thus does not require additional files to be downloaded.

Since the insurance data is private data, it can not be found in this repository. 

The process of scraping p2000 notifications can be found in the directory 'p2000'.

Scripts that are used to generate datasets and train and test models can be found in the directory 'scripts'.

Results can be found in the directory 'results'. 

## Generating the final dataset
We use the script 'scripts/generate_dataset.py' to generate the final dataset. This script also serves as an example on how to use the tools provided in the repo. The script works as follows:

1. Get all positive instances using the p2000 or insurance dataset.
2. Enrich positive instances with precipitation data.
3. Only keep the instances with enough precipitation, remove the others. 
4. Add a height layer to these instances. 
5. Sample a negative instance with the same height map but a somewhat random amount of rainfall. 

## Changing resolution
Changing the spatial resolution of the p2000 dataset can be done with the file 'scripts/changeresolution.py'. The BAG, a database with information on all residences in the Netherlands, was used in this process.  Make sure to fill in your own API key if you would like to do this. 

An API key can be requested here: <https://www.kadaster.nl/zakelijk/producten/adressen-en-gebouwen/bag-api-individuele-bevragingen> 

## The feature construction year

In my thesis we have also looked at the possibility to use the construction year of a house as a predictive feature. Unfortunately, the scripts to engineer this feature got lost due to a laptop that died. However, the BAG was also used for this. 



