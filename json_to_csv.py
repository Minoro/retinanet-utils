''' 
	Json to CSV
	Converte os arquivos em JSON em CSV, transformando cada poligono em uma linha
	Os metadados do json são transformados em colunas e repetidos em todas as linhas
'''

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from os import path
import json
from shapely import wkt
from tqdm import tqdm
import logging
import sys


# valor numérico para o nível do dano
DAMAGE_LEVEL = {
    'un-classified' : 0,
    'no-damage' : 1,
    'minor-damage' : 2,
    'major-damage' : 3,
    'destroyed' : 4,
}

# Cores padrões para o nível de dano
DAMAGE_COLORS = [
    (1.0, 1.0, 1.0),            # Branco
    (0.0, 1.0, 0.0),            # Verde
    (1.0, 1.0, 148/255.0),      # Amarelo
    (1.0, 0.5, 0.0),            # Laranja
    (1.0, 0.0, 0.0),            # Vermelho
]

def read_json(json_file):
    '''
        Lê o arquivo json passado por parâmetro
    '''

    with open(json_file) as f:
        return json.loads(f.read(), encoding='utf-8')

    
def read_properties(polygon_data):
    '''
		Transforma as proprieadades dos polygonos lidas do arquivo JSON em um
		dicionário para ser manipulado pelo posteriormente
		Se não houver informação sobre o nível do dano marca como não classificado
		Adiciona as informações do nivel do dado e uma cor para o mesmo
    '''

    data = dict()
    
    data['feature_type'] = polygon_data['properties']['feature_type']
    data['uid'] = polygon_data['properties']['uid']
    data['geometry'] = polygon_data['wkt']

    if 'subtype' not in polygon_data['properties']:
        data['damage'] = 'un-classified'
    else:
        data['damage'] = polygon_data['properties']['subtype']
    
    data['damage-level'] = DAMAGE_LEVEL[data['damage']]
    data['color'] = DAMAGE_COLORS[data['damage-level']]
    
    return data


def convert_to_dataframe(labels_path, images_path, dataset, split_pre_and_post=True):
    ''' Converte as labels do arquivo JSON para dataframe transformando cada poligono
        em uma linha no dataframe;
        Os metadados são transformados em colunas.
        São criados três dataframes distrintos, um com todas as imagens, um com o pré-desastre e outro com o pós-desastre

        :param labels_path - caminho até a pasta onde se encontram as labels em arquivo JSON
        :param images_path - caminho até a pasta onde se encontram as images para ser adicionado ao nome das imagems
        :param dataset - especifica qual parte do dataset está utilizando (train, tier3, test, holdout)
        :param split_pre_and_post - se verdadeiro cria três dataframes distintos, um com todas as imagens, um com o pré-desastre e outro com o pós-desastre
    '''  

    df_data = []
    df_pre_data = []
    df_post_data = []

    for file_name in tqdm(os.listdir(labels_path)):
        file_name_parts = file_name.split('_')

        disaster = file_name_parts[0]
        file_number = file_name_parts[1]
        pre_post_disaster = file_name_parts[2]
        sufix = file_name_parts[3]

        json_file_path = os.path.join(labels_path, file_name)
        json_data = read_json(json_file_path)


        df_content = {
            'disaster': disaster, 
            'file_number': file_number, 
            'file_name': file_name,
            'pre_or_post_type': pre_post_disaster,
            'dataset': dataset,
        }

        #combina os dicionarios
        df_content = {**df_content, **json_data['metadata']}
        #adiciona o caminho das images ao nome do arquivo
        df_content['img_name'] = path.join(images_path, df_content['img_name']) 

        polygons_data = json_data['features']['xy']

        for polygon_data in polygons_data:
            data = read_properties(polygon_data)
            row = { **df_content, **data }
            df_data.append(row)

            if split_pre_and_post:
                # separa em dataframes distintos
                if pre_post_disaster == 'pre':
                    df_pre_data.append(row)
                else:
                    df_post_data.append(row)

    df = pd.DataFrame(df_data)
    df_pre = pd.DataFrame(df_pre_data)
    df_post = pd.DataFrame(df_post_data)

    return df, df_pre, df_post


def dataframe_to_csv(df, file_path):
    '''
        Exporta o dataframe para csv
    '''

    logging.debug('Salvando dataframe no diretorio: {}'.format( file_path ))
    df.to_csv(file_path)



if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """json_to_csv.py: Transforma os arquivos json do xBD dataset em um csv, transformando cada polygono em uma linha do csv\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/xBD/",
                        help="""Caminho até o diretório base do dataset, pode ser o de treino, tier3, test ou holdout.\n
                        A pasta de imagens e labels deve estar dentro do diretório""")

    parser.add_argument('--output', required=True, metavar="/path/output/csv/", help='Caminho para salvar os arquivos em csv')

    parser.add_argument('--split-pre-post', 
                        default=True,
                        help='Separa os dados em três datasets distintos, com todas as amostras, com as amostras pré-desastre e com pós-desastre')

    parser.add_argument("-v", "--verbose", help="Exibe mensagens durante a execução", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)


    if not path.isdir(args.input):
            print(
                "[ERROR] Diretório não encontrado {}.\n\n"
                .format(args.input),
                file=sys.stderr)
            sys.exit(2)


    logging.debug('Modo verboso ativo')

    images_path = path.join(args.input, 'images')
    labels_path = path.join(args.input, 'labels')
    target_path = path.join(args.input, 'targets')
    has_target = path.isdir(target_path)

    logging.debug('Diretório base: {}'.format(args.input))
    logging.debug('Diretório de imagens: {}'.format(images_path))
    logging.debug('Num. Imagens: {}'.format( len(os.listdir(images_path)) ))

    logging.debug('Labels Path: {}'.format( labels_path ))
    logging.debug('Num. Labels: {}'.format( len(os.listdir(labels_path)) ))


    if has_target:
        logging.debug('Target Path: {}'.format( labels_path ))
        logging.debug('Num. Targets: {}'.format( len(os.listdir(labels_path)) ))

    dataset_path = path.normpath(args.input)
    folders = dataset_path.split(os.sep)
    dataset_type = folders[-1]

    logging.debug('Parte do dataset')
    df, df_pre, df_post = convert_to_dataframe(labels_path, images_path, dataset_type, args.split_pre_post)
    
    # cria o diretório de saída se não existir
    if not path.isdir(args.output):
        logging.debug('Criando diretório de saída: {}'.format(args.output))
        os.mkdir(args.output)

    output_csv = path.join(args.output, '{}.csv'.format(dataset_type))
    dataframe_to_csv(df, output_csv)

    if args.split_pre_post:
        output_csv = path.join(args.output, '{}_pre.csv'.format(dataset_type))
        dataframe_to_csv(df_pre, output_csv)

        output_csv = path.join(args.output, '{}_post.csv'.format(dataset_type))
        dataframe_to_csv(df_post, output_csv)


