'''
    Gera o arquivo CSV com as classes com base no CSV de amostras de treino
'''

import pandas as pd
import geopandas as gpd
import numpy as np
import random
import os
from os import path
import json
from shapely import wkt
import csv
from tqdm import tqdm
import logging
import sys


def load_dataframe(dataframe_path):
    logging.debug('Carregando dataframe: {}'.format(dataframe_path))
    df = pd.read_csv(dataframe_path, names=["img_path", "xmin", "ymin", "xmax", 'ymax', 'disaster_type'])

    return df


def dataframe_to_csv(df, file_path):
    logging.debug('Salvando dataframe: {}'.format(file_path))
    df.to_csv(file_path, index=False)


def generate_class_csv(df, output_path):
    disasters = df.disaster_type.unique()

    logging.debug('Número de classes encontradas: {}'.format( len(disasters) ))

    with open(output_path, mode='w') as csvfile:
        csv_writer = csv.writer(csvfile)

        for index, disaster in enumerate(disasters):
            csv_writer.writerow([disaster, index])


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """csv_class_generator.py: Gera um arquivo CSV mapeando as classes para valores a partir das amostras de treino\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/csv/samples.csv",
                        help='Caminho para o CSV de amostras')

    parser.add_argument('--output', 
                        required=True, 
                        metavar="/path/output/csv/classes.csv", 
                        help='Caminho até o arquivo CSV de saída')

    parser.add_argument("-v", "--verbose", help="Exibe mensagens durante a execução", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Modo verboso ativo')

    df = load_dataframe(args.input)
    generate_class_csv(df, args.output)

