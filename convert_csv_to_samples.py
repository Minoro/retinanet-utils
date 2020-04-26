'''
    Converte os arquivos CSV contentendo as informações dos Polygonos para amostras da Retinanet
'''

import pandas as pd
import geopandas as gpd
import numpy as np
import random
import os
from os import path
from skimage.io import imread
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
from shapely import wkt
import csv
from tqdm import tqdm
import logging
import sys

RANDOM_SEED = 42

# BORDER = 10 #pixels de borda
MIN_WIDTH = 32
MIN_HEIGHT = 32 

IMAGE_SIZE = (1024, 1024)


def sample_coords(df, images_path, border=10):
    '''
        Amostra as coordenadas do CSV e transforma em amostra para a Retinanet
        
        :param df - dataframe dos poligonos
        :images_path - caminho até o diretório de imagens para treino do modelo
        :border - tamanho em pixels para expandir as amostras
        :return - lista com as amostras geradas no formato [path_image, x1, y1, x2, y2, classe]
    '''

    disaster_type = df.iloc[0].disaster_type

    samples = []
    
    images = df.img_name.unique()
    for img_name in tqdm(images):
        
        # amostra os poligonos associados a imagem 
        polygons = df[ df['img_name'] == img_name].geometry.values

        for polygon in polygons:
            coords =  np.array(list(polygon.exterior.coords))
            xmin, xmax, ymin, ymax = get_sample_coords(coords, border)

            #amostra com coordenadas inválida
            if xmin is None:
                continue
                
            #caminho para a imagem, utilizado no treino
            img_path = os.path.join(images_path, img_name)

            sample = (img_path, xmin, ymin, xmax, ymax, disaster_type)
            samples.append(sample)
    
    return samples

def get_sample_coords(coords, border=10):
    '''
        Extrai as coordenadas do retângulo que envolve o poligono
        Expande o retangulo pelo tamanho das bordas
    '''
    xcoords = coords[:, 0]
    ycoords = coords[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)
    

    xdiff = xmax - xmin
    ydiff = ymax - ymin
    
    if xdiff <= MIN_WIDTH or ydiff <= MIN_HEIGHT:
        return None, None, None, None
    
    xmin = max(int(xmin - border), 0)
    xmax = min(int(xmax + border), IMAGE_SIZE[1])
    ymin = max(int(ymin - border), 0)
    ymax = min(int(ymax + border), IMAGE_SIZE[0])

    return (xmin, xmax, ymin, ymax)


def write_samples_to_csv(samples, file_name):
    '''
        Adiciona as amostras ao CSV informado
    '''
    with open(file_name, 'a+') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(samples)


def export_dataframe_to_samples(df, images_path, output_csv, border=10):
    # embaralha o dataframe
    df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    disasters_types = df['disaster_type'].unique()

    for disaster_type in disasters_types:
        sub_sampled_df = df[ df['disaster_type'] == disaster_type]
        
        logging.debug('Amostrando {}...'.format(disaster_type))
        samples = sample_coords(sub_sampled_df, images_path, border)

        logging.debug('Total de amostras {}'.format(len(samples)))
        logging.debug('Exportando para CSV: {}...'.format(output_csv))

        write_samples_to_csv(samples, output_csv)



if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """convert_csv_to_samples.py: Transforma os arquivos em CSV com poligonos de danos das contruções em um CSV de amostras para a Retinanet\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/csv/datraframe.csv",
                        help='Caminho o CSV contendo os polygonos de danos. Cada poligono deve ser uma linha do csv')


    parser.add_argument('--images-path',
                        required=True,
                        metavar="/path/to/images/",
                        help='Caminho para as imagens de treino para serem encontradas pelo modelo')


    parser.add_argument('--output', required=True, metavar="/path/output/csv/samples.csv", help='Caminho até o arquivo CSV de amostras para treino da Retinanet')

    parser.add_argument('--border', 
                        default=10,
                        help='Tamanho da borda (em pixels) para expandir os poligonos das construções')

    parser.add_argument("-v", "--verbose", help="Exibe mensagens durante a execução", action="store_true")

    args = parser.parse_args()

    if not path.isdir(args.images_path):
        print(
            "[ERROR] Diretório não encontrado {}.\n\n"
            .format(args.images_path),
            file=sys.stderr)
        sys.exit(2)



    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Modo verboso ativo')


    # Se for informado
    output_csv = args.output
    if path.isdir(output_csv):
        output_csv = path.join(output_csv, 'samples.csv')


    df = pd.read_csv(args.input)
    df['geometry'] = df['geometry'].apply(wkt.loads)

    export_dataframe_to_samples(df, args.images_path, output_csv, args.border)