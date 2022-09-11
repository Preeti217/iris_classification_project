# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sys
import pandas as pd
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def get_data_from_path():
    #checks if file exists at the location
    if os.path.isfile('../../data/raw/Iris.csv'):
        data_full_path = '../../data/raw/Iris.csv'
    else:
        #downloads the file at the given location
        sys.path.append('../../data/raw')
        import data.raw.import_data as iD
        iD.import_data_from_kaggle()
        data_full_path = '../../data/raw/Iris.csv'

    #reads the csv file
    data = pd.read_csv(data_full_path)
    return data

def get_or_create_interim_data():
    raw_data= get_data_from_path()
    #the raw data should be repurposed as interim data and should not be formatted in any way
    data_full_path = '../../data/interim/Iris.csv'
    if os.path.isfile('../../data/interim/Iris.csv'):
        print('Interim data found... creating copy from interim data')
    else:
        print('Interim data not found... creating a copy from raw data and saving at interim location')
        base_name = os.getcwd()
        os.chdir('../../data/interim')
        raw_data.to_csv('Iris.csv', index = False)
        os.chdir(base_name)
    interim_data = pd.read_csv(data_full_path)
    return interim_data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    first_data = get_or_create_interim_data()
    first_data.head()

    #main()
