'''
generating reference data1
'''

import os
import numpy as np
import pandas as pd
import sklearn.utils
from LIBS.Generator import my_images_generator_2d
SAMPLES_NUM = 64
ADD_BLACK_INTERVAL = 16
IMAGE_SHAPE = (299, 299, 3)

# filename_csv images after preprocessed
filename_csv = os.path.join(os.path.abspath('.'),  'split_patid_v1.csv')
df = pd.read_csv(filename_csv)
df = sklearn.utils.shuffle(df, random_state=22222)
imagefiles = df[0:SAMPLES_NUM]['images'].tolist()
generator_test = my_images_generator_2d.My_images_generator_2d_test(imagefiles,
            batch_size=SAMPLES_NUM, do_normalize=True, image_shape=IMAGE_SHAPE)
x_test = generator_test.gen().__next__()

#add black images
img_black = np.zeros(IMAGE_SHAPE)
from LIBS.ImgPreprocess.my_image_norm import input_norm
img_black = input_norm(img_black)
img_black = np.expand_dims(img_black, axis=0)

for i in range(SAMPLES_NUM):
    if (i % ADD_BLACK_INTERVAL == 0):
        x_test = np.append(x_test, img_black, axis=0)

x_test = np.asarray(x_test, dtype=np.float16)
save_filename = 'reference.npy'
np.save(save_filename, x_test)

# background = np.load(save_filename)

print('OK')

