#!/usr/bin/python3
import math
import os
import argparse
import json
from sys import argv

from PIL import Image
from PIL.TiffTags import TAGS
import cv2 as cv
import numpy as np
from py_tps import TPSFile, TPSImage, TPSCurve, TPSPoints

import matplotlib
import matplotlib.pyplot as plt


def get_files(dir: str, filter=None):
    if not os.path.isdir(dir):
        raise ValueError(f'path: {dir} is not valid directory.')
    files = []
    if filter is not None:
        for path, dirs, fs in os.walk(dir):
            files += [os.path.join(path, f) for f in fs if filter in f]
    else:
        for path, dirs, fs in os.walk(dir):
            files += [os.path.join(path, f) for f in fs]
    return files


def get_mask(path: str):
    if not os.path.isfile(path):
        raise ValueError(f'path: {path} is not valid file.')
    return np.load(path)


def measure(mask, img_path):
    bin_im = bin_img(mask)
    contours, _ = cv.findContours(bin_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = contours[0]
    A = tuple(contour[contour[:, :, 1].argmin()][0])  # Top point
    B = tuple(contour[contour[:, :, 1].argmax()][0])  # Bottom point
    length = math.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)
    with Image.open(img_path) as img:
        #print('dpi:', img.info['dpi'])
        meta_dict = {}
        for key in img.tag.keys():
            try:
                meta_dict[TAGS[key]] = img.tag[key]
            except KeyError:
                pass
        #for k,v in zip(meta_dict.keys(), meta_dict.values()):
        #    print(k,':',v)
    y_res = meta_dict['YResolution'][0][0]/meta_dict['YResolution'][0][1]
    x_res = meta_dict['XResolution'][0][0]/meta_dict['XResolution'][0][1]
    assert y_res == x_res
    res = 3
    if meta_dict['ResolutionUnit'][0] == 3:  # cm
        length = length/x_res # convert pixels to cm
    elif meta_dict['ResolutionUnit'][0] == 2:  # inches
        length = (length*2.54)/x_res # convert pixels to inches to cm
    else:
        res = 1
    return (A, B), length, res, contour


def bin_img(mask):
    bin_img = np.zeros(mask.shape[:2], dtype=np.uint8)
    mask = np.reshape(mask, bin_img.shape)
    bin_img[mask] = 255
    return bin_img

def segment(img, mask, name: str, path='/data/public/Bakalářka/outputs/'):
    zero = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = np.reshape(mask, img.shape[:2])
    zero[mask] = 255
    result = cv.bitwise_and(img, img, mask=zero)
    result[zero == 0] = (255, 255, 255)
    cv.imwrite(path + name, result)

if __name__ == '__main__':
    arg = argparse.ArgumentParser(description='Process output of MaskRCNN saves as .npy file formats.')
    arg.add_argument('--images', '-i', required=True, dest='img_dir', help='path to directory.', type=str)
    arg.add_argument('--dir', '-d', required=True, dest='npy_dir', help='path to .npy files.', type=str)
    arg.add_argument('--out', '-o', required=False, dest='out_path', default=None, help='Directory to store output.',
                     type=str)

    args = arg.parse_args(argv[1:])

    mask_files = get_files(args.npy_dir)
    img_files = get_files(args.img_dir, '.tif')
    mask_files.sort(key=lambda x: x.split('/')[-1])
    img_files.sort(key=lambda x: x.split('/')[-1])
    if len(mask_files) != len(img_files):
        print('Not same length')
        print('Masks: ', len(mask_files))
        print('Images: ', len(img_files))
    directories = {}
    #tps_images = []
    #json_out = []
    for mask_path, img_path in zip(mask_files, img_files):
        mask = get_mask(mask_path)
        img = bin_img(mask)
        A_B_points, length, res, c = measure(mask, img_path)
        c = np.reshape(c, (c.shape[0], 2))
        json_o = {
            'Image'       : img_path.split('/')[-1],
            'TopPoint'    : (int(A_B_points[0][0]),int(A_B_points[0][1])),
            'BottomPoint' : (int(A_B_points[1][0]),int(A_B_points[1][1])),
            'Length'      : length,
            'LengthUnit'  : 'cm' if res == 3 else 'pixels',
            'Contour'     : c.tolist()
        }
        points = TPSPoints(c)
        #tps_images.append(TPSImage(img_path.split('/')[-1], landmarks=points))
        #json_out.append(json_o)
        if (img_path.split('/')[-2] + '_' + img_path.split('/')[-3]) in directories.keys(): # Add json and tps image
            directories[img_path.split('/')[-2] + '_' + img_path.split('/')[-3]].append((json_o, TPSImage(img_path.split('/')[-1], landmarks=points)))
        else: # Create key
            directories[img_path.split('/')[-2] + '_' + img_path.split('/')[-3]] = [(json_o, TPSImage(img_path.split('/')[-1], landmarks=points))]

    for k in directories.keys():
        file_name = 'output_'
        file_name += k
        json_o, tps_i = [], []
        for (j, t) in directories[k]:
            json_o.append(j)
            tps_i.append(t)
        tps_f = TPSFile(tps_i)
        tps_f.write_to_file(os.path.join(args.out_path,file_name + '.tps'))
        with open(os.path.join(args.out_path,file_name + '.json'), 'w') as fp:
            json.dump(json_o, fp, indent=4, separators=(',', ': '))
