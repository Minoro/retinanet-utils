'''
    Subamostras os exemplos em um arquivo CSV menor para testes
'''


import pandas as pd
import geopandas as gpd
import numpy as np
import random
import os
from os import path
import json
from shapely import wkt
from shapely.geometry import Polygon
import csv
from tqdm import tqdm
import logging
import sys

RANDOM_SEED = 42


def convert_to_polygon(row):   
    return Polygon([
        (row['xmin'], row['ymin']), 
        (row['xmax'], row['ymin']), 
        (row['xmax'], row['ymax']), 
        (row['xmax'], row['ymin'])
    ])

def extract_desaster(img_path):
    # extrai o nome do desastre no caminho da imagem
    folders = path.normpath(img_path)
    folders = folders.split(os.sep)

    file_name = folders[-1]

    return file_name.split('_')[0] 

def load_dataframe(dataframe_path):
    df = pd.read_csv(dataframe_path, names=["img_path", "xmin", "ymin", "xmax", 'ymax', 'disaster_type'])

    # colunas adicionais
    # df['shape'] = df.apply(lambda row : convert_to_polygon(row), axis=1)
    # df['disaster'] = df.apply(lambda row : extract_desaster(row.img_path), axis=1)
    return df

def subsample_per_disaster(df, num_samples=100):
    disasters = df.disaster.unique()
    #TODO - implementar amostragem por imagem
    for disaster in disasters:
        pass 


def subsample_per_disaster_type(df, num_samples=100):
    #TODO - implementar amostragem por tipo de desastre
    pass


def subsample_per_image(df, max_num_samples=100, max_samples_per_image=10):
    '''
        Subamostra um dataframe podendo-se limitar o número máximo de amostras por imagem
    '''
    # embaralha as amostras

    images = df.img_path.unique()

    #embaralha as imagens
    random.seed(RANDOM_SEED)
    random.shuffle(images)

    if max_num_samples is None:
        max_num_samples = len(df.index)

    if max_samples_per_image is None:
        max_samples_per_image = np.inf

    image_index = 0
    total_samples = 0
    sampled_dataframes = []
    while total_samples < max_num_samples and image_index < len(images):
        image = images[image_index]
        
        sub_df = df[ df['img_path'] == image]

        # evita obeter mais amostras do que há para a imagem
        num_sample_img = min(max_samples_per_image, len(sub_df.index)) 
        sampled_dataframes.append(sub_df.sample(num_sample_img))

        total_samples += num_sample_img
        image_index += 1

    return  pd.concat(sampled_dataframes, ignore_index=True)


def subsamples_to_csv(df, file_path):
    df.drop(['disaster', 'shape'], errors='ignore')
    df.to_csv(file_path, index=False)






if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """convert_csv_to_samples.py: Transforma os arquivos em CSV com poligonos de danos das contruções em um CSV de amostras para a Retinanet\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/csv/datraframe.csv",
                        help='Caminho o CSV contendo todas as amostras para a Retinanet')


    parser.add_argument('--output', required=True, metavar="/path/output/csv/sub_samples.csv", help='Caminho até o arquivo CSV de subamostras para treino da Retinanet')

    parser.add_argument('--num-samples', 
                        default=100,
                        help='Número de (sub)amostras')

    parser.add_argument('--max-samples-per-image', 
                        default=10,
                        help='Número máximo de amostras por imagens')


    parser.add_argument("-v", "--verbose", help="Exibe mensagens durante a execução", action="store_true")

    args = parser.parse_args()


    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Modo verboso ativo')

    df = load_dataframe(args.input)

    subsample_df = subsample_per_image(df, args.num_samples, args.max_samples_per_image)
    subsamples_to_csv(subsample_df, args.output)