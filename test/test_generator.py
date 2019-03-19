import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='1'
sys.path.insert(0,'/clever/cabernet/wuwenhui/git_ocr_project/EAST/')
import cv2
import numpy as np
import tensorflow as tf
import model
import icdar
import lanms
from eval import resize_image, sort_poly, detect

import time

def main(argv=None):
    data_generator = icdar.get_batch(num_workers=4,
                                         input_size=512,
                                         batch_size=2)
    for i in range(100):
        data = next(data_generator)
        print(i)
        time.sleep(0.1)
    print('done')

if __name__ == '__main__':
    tf.app.run()
    
