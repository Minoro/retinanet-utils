'''
    Exibe as amostras geradas para a retinanet
'''


import pandas as pd
import os
from os import path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import logging
import sys

DEFAULT_COLOR = (0, 255, 0, 100)

def load_dataframe(csv_path):
    return pd.read_csv(csv_path, names=['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'disaster_type'])

def show(samples_csv, image_name=None, show_text=False):
    df = load_dataframe(samples_csv)
    logging.debug('Total de amostras: {}'.format( len(df.index) ))

    # Seleciona uma imagem aleatória
    if image_name is None:
        image_name = df.sample().iloc[0].img_path

    # filtra as imagens pelo nome
    df = df[ df['img_path'].str.endswith(image_name) ]
    
    logging.debug('Exibindo imagem: {}'.format(df.iloc[0].img_path))
    logging.debug('Número de amostras: {}'.format( len(df.index) ))
    
    logging.debug('Abrindo imagems...')
    img = Image.open(image_name)

    logging.debug('Plotando poligonos...')
    draw = ImageDraw.Draw(img, 'RGBA')
    fig, ax = plt.subplots(figsize=(18, 10))
    for index, row in df.iterrows():
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']

        coords = [(xmin, ymin), (xmax, ymax)]
        draw.rectangle(coords, outline=DEFAULT_COLOR)

        if show_text:
            draw.text((xmin, ymin), "xmin: {} \nymin: {}\nxmax: {} \nymax: {}  \nxdiff: {} \nydiff: {}".format(xmin, ymin, xmax, ymax, xmax-xmin, ymax-ymin))


    logging.debug('Exibindo Imagem...')
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """convertshow_samples.py: Exibe os poligonos de amostra para uma imagem\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/csv/samples.csv",
                        help='Caminho para o CSV contento as amostras')

    parser.add_argument('--name', metavar="nome_da_amostra.png", help='Nome da amostra para ser exibida')

    parser.add_argument('--show-text', help='Exibe texto com informações das dimensões dos poligonos', action="store_true")

    parser.add_argument("-v", "--verbose", help="Exibe mensagens durante a execução", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Modo verboso ativo')

    show(args.input, args.name, args.show_text)