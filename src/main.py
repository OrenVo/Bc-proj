#!/usr/bin/python3
#%%
import sys

from PIL import Image
from PIL.TiffTags import TAGS
import time
import numpy as np
import cv2 as cv
import argparse
from modules.detector import ObjectsDetector
from modules.files import FileReader

# Arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument('files', metavar='F', type=str,
                    nargs='*', help='Files to be processed')
parser.add_argument('--dir', '-d', dest='directory', default=None,
                    type=str, help="Directory with files to process")

if __name__ == '__main__':
    proccesed_images = 0
    args = parser.parse_args()
    detector = ObjectsDetector()
    file_reader = FileReader(directory=args.directory)
    file_reader.read_dir()
    times = []
    while True:
        start_time = time.time()
        img, file_path = file_reader.next_file()
        print(img.shape)
        break
        if '.txt' in file_path or '.tps' in file_path:
            continue

        if img is None:
            break
        proccesed_images += 1
        contours = detector.detect(img)
        contours.sort(key=cv.contourArea, reverse=True)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, contours, 0, (255, 255, 255), -1)
        result = cv.bitwise_and(img, img, mask=mask)
        result[mask == 0] = (255, 255, 255)
        cv.drawContours(img, contours, 0, (0, 255, 0), 2)
        cv.imwrite('/data/public/Bakalářka/outputs/'+file_path.split('/')[-1]+'f', result)
        times.append(time.time()-start_time)
    print(f'Proccesed {proccesed_images} files.')
    print(f'Time: {sum(times)} s')
    print(f'Min time (per image): {min(times)} ms')
    print(f'Max time (per image): {max(times)} ms')
    print(f'Mean time (per image): {sum(times)/len(times)} ms')