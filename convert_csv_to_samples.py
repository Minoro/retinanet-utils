'''
    Converte os arquivos CSV contentendo as informações dos Polygonos para amostras da Retinanet
    O Arquivo CSV será convertido em um dataframe e os dados dos poligonos (columa geometry) 
    será transformada em objeto do tipo Pologyon

    Os poligonos serão transformados em um retangulo envolvente que pode ser expandido em um determinado
    número de pixels

    O CSV de entrada deve conter apenas o nome das imagens, o caminho para elas será fornecido como parâmetro
    para este script e adicionado ao dataframe

    Amostras menores que os tamanhos definidos serão desconsideradas
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

# Tamanho mínimo dos retângulos envolventes
MIN_WIDTH = 32
MIN_HEIGHT = 32 

IMAGE_SIZE = (1024, 1024)

# Tamanho padrão das ancoras de treino
ANCHOR_SIZES = [32, 64, 128, 512]




def export_dataframe_to_samples(df, output_csv, border=10, min_distance=0, fit_anchors=False, overlap=True):
    # embaralha o dataframe
    df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    disasters_types = df['disaster_type'].unique()
    logging.debug('Número de desastres {}...'.format(len(disasters_types)))
    
    for disaster_type in disasters_types:

        sub_sampled_df = df[ df['disaster_type'] == disaster_type]
        
        logging.debug('Amostrando {}...'.format(disaster_type))
        samples = sample_coords(sub_sampled_df, border, min_distance, fit_anchors, overlap)

        logging.debug('Total de amostras: {}'.format(len(samples)))

        logging.debug('Exportando para CSV: {}...'.format(output_csv))
        write_samples_to_csv(samples, output_csv)


def sample_coords(df, border=10, min_distance=0, fit_anchors=False, overlap=True):
    '''
        Amostra as coordenadas do CSV e transforma em amostra para a Retinanet
        
        :param df - dataframe dos poligonos
        :border - tamanho em pixels para expandir as amostras
        :fit_anchors - Se verdadeiro aproxima o tamanho da amostra aos tamanhos das ancoras
        :return - lista com as amostras geradas no formato [path_image, x1, y1, x2, y2, classe]
    '''

    disaster_type = df.iloc[0].disaster_type

    samples = []
    
    images = df.img_name.unique()
    for img_name in tqdm(images):
        #polygonos contidos nessa imagem
        samples_polygon = []

        # amostra os poligonos associados a imagem 
        polygons = df[ df['img_name'] == img_name].geometry.values

        for polygon in polygons:
            coords =  np.array(list(polygon.exterior.coords))
            xmin, xmax, ymin, ymax = get_sample_coords(coords, border, min_distance, fit_anchors)

            #amostra com coordenadas inválida
            if xmin is None or xmax is None or ymin is None or ymax is None:
                continue

            if not overlap:
                # Converte as coordenadas das amostras em um poligono
                sample_polygon = convert_to_polygon(xmin, xmax, ymin, ymax)

                # desconsidera a amostra se estiver sobrepondo outras
                if is_sample_overlaping(sample_polygon, samples_polygon):
                    continue

                samples_polygon.append(sample_polygon)

            sample = (img_name, xmin, ymin, xmax, ymax, disaster_type)
            samples.append(sample)
    
    return samples

def get_sample_coords(coords, border=10, min_distance=0, fit_anchors=False, overlap=True):
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
    
    if xdiff < MIN_WIDTH or ydiff < MIN_HEIGHT:
        return None, None, None, None


    if fit_anchors:
        return get_fited_anchor(xmin, xmax, ymin, ymax)

    xmin, xmax, ymin, ymax = expand_borders(xmin, xmax, ymin, ymax, border)

    return (xmin, xmax, ymin, ymax)


def get_fited_anchor(xmin, xmax, ymin, ymax):
    '''
        Aproxima o tamanho da ancora ao tamanho padrão maior mais pŕoximo
        Se as dimensões forem superiores a maior caixa desconsidera a amostra
    '''
    xmin, xmax = get_expanded_anchor_min_max(xmin, xmax)
    ymin, ymax = get_expanded_anchor_min_max(ymin, ymax)
    
    # Descarta coordenadas que não se enquadram nos tamanhos das ancoras
    if xmin is None or xmax is None or ymin is None or ymax is None:
        return None, None, None, None
    
    return xmin, xmax, ymin, ymax

def expand_borders(xmin, xmax, ymin, ymax, border):
    '''
        Expande as coordenadas informadas pelo tamanho da borda informada
        É limitado as dimensões da imagem (min = 0 e max = image_size)
    '''
    xmin = reduce_border(xmin, border)
    xmax = augment_border(xmax, border)
    ymin = reduce_border(ymin, border)
    ymax = augment_border(ymax, border)

    return xmin, xmax, ymin, ymax


def get_expanded_anchor_min_max(min_coord, max_coord):
    '''
        Recupera as coordenadas minimas e máximas após apróximar ao tamanho da amostra
    '''

    # Recalcula após a adição das bordas
    diff = max_coord - min_coord

    close_anchor_size = get_close_anchor_size(diff)
    # caixa maior que o limite das ancoras
    if close_anchor_size is None:
        return None, None

    diff_to_anchor_size = close_anchor_size - diff
    exprand_anchor_size = int( (diff_to_anchor_size) / 2)
    min_coord = reduce_border(min_coord, exprand_anchor_size)

    #ajuste para diferenças impares para compensar na caixa maior
    if diff_to_anchor_size%2 != 0:
        exprand_anchor_size += 1

    max_coord = augment_border(max_coord, exprand_anchor_size)

    return min_coord, max_coord

def reduce_border(position, border):
    ''' 
        Reduz a coordenada pelo valor da borda, limitado a 0
    '''
    return max(int(position - border), 0)

def augment_border(position, border):
    '''
        Aumenta a coordenada pelo valor da borda limitado ao tamanho da imagem 
    '''
    return min(int(position + border), IMAGE_SIZE[0])


def get_close_anchor_size(anchor_size):
    '''
        Recupera o tamanho padrão da ancora maior mais próxima do tamanho informado
        Se maior que o maior tamanho padrão retorna None
    '''
    i = 0
    while i < len(ANCHOR_SIZES) and ANCHOR_SIZES[i] < anchor_size:
        i += 1
    
    # Desconsidera tamanhos maiores que o limite das ancoras
    if i < len(ANCHOR_SIZES):
        return ANCHOR_SIZES[i]
    
    return None




def convert_to_polygon(xmin, xmax, ymin, ymax):   
    '''
        Converte as coordenadas informadas em um poligono
    '''
    return Polygon([
        (xmin, ymin), 
        (xmax, ymin), 
        (xmax, ymax), 
        (xmin, ymax)
    ])

def is_sample_overlaping(sample, samples):
    ''' 
        Verifica se há sobreposição entre a amostra informada e o conjunto de amostra informado
        Tanto a amostra quando os exemplos do conjunto de amostras informado devem ser objetos
        do tipo Polygon
    '''
    for s in samples:
        if s.intersects(sample):
            return True

    return False


def write_samples_to_csv(samples, file_name):
    '''
        Adiciona as amostras ao CSV informado
    '''
    with open(file_name, 'a+') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(samples)

def load_dataframes(input_csv, merge_csv=None):
    '''
        Carrega o CSV para um dataframe na memória
    '''
    
    logging.debug('Carregando dataframe: {}'.format(input_csv))
    df = pd.read_csv(input_csv)
    
    if merge_csv:
        logging.debug('Combinando com dataframe: {}'.format(merge_csv))
        df = df.append(pd.read_csv(merge_csv), ignore_index=True, sort=False)

    logging.debug('Convertendo geometria')
    df['geometry'] = df['geometry'].apply(wkt.loads)

    return df

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

    parser.add_argument('--merge',
                        required=False,
                        metavar="/path/to/csv/datraframe_to_merge.csv",
                        help='Caminho o CSV contendo os polygonos de danos para ser mesclado com o dataframe de entrada.')

    parser.add_argument('--output', required=True, metavar="/path/output/csv/samples.csv", help='Caminho até o arquivo CSV de amostras para treino da Retinanet')

    parser.add_argument('--border', 
                        default=10,
                        help='Tamanho da borda (em pixels) para expandir os poligonos das construções')

    #TODO - Implementar
    parser.add_argument('--min-distance', 
                        default=0,
                        help='Distância mínima (em px) das amostras')

    parser.add_argument("--fit-anchors", help="Cria as caixas de treino no tamanho da ancora maior mais próxima", action="store_true")

    parser.add_argument("--no-overlap", help="Evita que haja sobreposição das amostras", action="store_true")

    parser.add_argument("-v", "--verbose", help="Exibe mensagens durante a execução", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Modo verboso ativo')

    df = load_dataframes(args.input, args.merge)

    output_csv = args.output
    if path.isdir(output_csv):
        output_csv = path.join(output_csv, 'samples.csv')
        logging.debug('Arquivo de saida: {}',format(output_csv))

    if path.exists(output_csv):
        os.remove(output_csv)

    overlap = not args.no_overlap

    export_dataframe_to_samples(df, output_csv, args.border, args.min_distance, args.fit_anchors, overlap)
