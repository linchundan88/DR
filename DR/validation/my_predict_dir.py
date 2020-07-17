import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from LIBS.DataPreprocess.my_data import get_images_labels, write_csv_dir_nolabel
from LIBS.DataValidation.my_multi_class import op_files_multiclass
import pandas as pd
from LIBS.DLP.my_predict_helper import do_predict

DO_PREPROCESS = False
GEN_CSV = True
COMPUTE_DIR_FILES = True

dir_original = '/media/ubuntu/data1/screen/original'
dir_preprocess = '/media/ubuntu/data1/screen/preprocess384/'
dir_dest = '/media/ubuntu/data1/ROP项目/screen/results/DR_english'
pkl_prob = os.path.join(dir_dest, 'probs.pkl')

from LIBS.ImgPreprocess import my_preprocess_dir
if DO_PREPROCESS:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
            image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

dicts_models = []
model_dir = '/tmp5/models_2020_6_19/DR_english/v1'
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionV3-004-0.982.hdf5'),
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-004-0.984.hdf5'),
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)
dict_model1 = {'model_file': os.path.join(model_dir, 'Xception-004-0.984.hdf5'),
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

filename_csv = os.path.join(dir_dest, 'DR_english_predict_dir.csv')
if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    write_csv_dir_nolabel(filename_csv, dir_preprocess)
df = pd.read_csv(filename_csv)
all_files, all_labels = get_images_labels(filename_csv_or_pd=df)

prob_total, y_pred_total, prob_list, pred_list = \
    do_predict(dicts_models, filename_csv, argmax=True)

import pickle
os.makedirs(os.path.dirname(pkl_prob), exist_ok=True)
with open(pkl_prob, 'wb') as file:
    pickle.dump(prob_total, file)

if COMPUTE_DIR_FILES:
    op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
                        dir_dest=dir_dest, dir_original=dir_original, keep_subdir=True)

print('OK')