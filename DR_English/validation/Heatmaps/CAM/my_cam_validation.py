
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
import shutil

from LIBS.ImgPreprocess import my_preprocess
from LIBS.Neural_Networks.Heatmaps.CAM import my_helper_cam, my_helper_grad_cam, my_helper_grad_cam_plusplus
from tensorflow import keras
from LIBS.Generator import my_images_generator_2d

DIR_PREPROCESS = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/'
DIR_DEST_HEATMAP = '/tmp5/DR_english_2020_6_22_new/'

TRAIN_TYPE = 'DR_english'
DATA_VERSION = 'v1'

model_dir = '/tmp5/models_2020_6_19/DR_english/v1'
dicts_models = []
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-004-0.984.hdf5'),
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

for dict1 in dicts_models:
    print('prepare to load model:' + dict1['model_file'])
    dict1['model'] = keras.models.load_model(dict1['model_file'], compile=False)
    print('model load complete!')

for heatmap_type in ['CAM', 'grad_cam', 'gradcam_plus']:
    for predict_type_name in ['split_patid_train', 'split_patid_valid', 'split_patid_test']:
        filename_csv = os.path.join(os.path.abspath('../../../'),
                'datafiles', TRAIN_TYPE, '{}_{}.csv'.format(predict_type_name, DATA_VERSION))

        if heatmap_type == 'CAM':
            my_cam = my_helper_cam.My_cam(model=dicts_models[0]['model'])
        if heatmap_type == 'grad_cam':
            my_grad_cam = my_helper_grad_cam.My_grad_cam(model=dicts_models[0]['model'])
        if heatmap_type == 'gradcam_plus':
            my_grad_cam_plusplus = my_helper_grad_cam_plusplus.My_grad_cam_plusplus(model=dicts_models[0]['model'])

        df = pd.read_csv(filename_csv)
        for _, row in df.iterrows():
            image_file = row['images']
            image_label = int(row['labels'])
            # img_source = image_file.replace('/preprocess384/', '/original/')

            assert DIR_PREPROCESS in image_file, 'preprocess directory error'
            preprocess = False
            input_shape = (299, 299, 3)
            if preprocess:
                img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
                img_input = my_images_generator_2d.my_gen_img_tensor(img_preprocess,
                                                                    image_shape=input_shape)
            else:
                img_source = image_file
                img_input = my_images_generator_2d.my_gen_img_tensor(image_file,
                                                                    image_shape=input_shape)

            model1 = dicts_models[0]['model']
            probs = model1.predict(img_input)
            class_predict = np.argmax(probs)

            if (class_predict == 1 and image_label == 1) or (class_predict == 1 and image_label == 0):
                if heatmap_type == 'CAM':
                    filename_heatmap = my_cam.gen_heatmap(img_input=img_input, pred_class=class_predict,
                                                          cam_relu=True, blend_original_image=True)
                if heatmap_type == 'grad_cam':
                    filename_heatmap = my_grad_cam.gen_heatmap(img_input=img_input, pred_class=class_predict,
                                                               blend_original_image=True)
                if heatmap_type == 'gradcam_plus':
                    filename_heatmap = my_grad_cam_plusplus.gen_heatmap(img_input=img_input, pred_class=class_predict,
                                                                        blend_original_image=True)

                save_dir = os.path.join(DIR_DEST_HEATMAP, heatmap_type, predict_type_name)

                #jpeg, gif, file extension may be fifferent.
                _, file_name = os.path.split(filename_heatmap)
                save_dir = os.path.join(DIR_DEST_HEATMAP, predict_type_name)
                if class_predict == 1 and image_label == 1:
                    dest_dir = os.path.dirname(image_file.replace(DIR_PREPROCESS, os.path.join(save_dir, '1_1/')))
                if class_predict == 1 and image_label == 0:
                    dest_dir = os.path.dirname(image_file.replace(DIR_PREPROCESS, os.path.join(save_dir, '0_1/')))
                os.makedirs(dest_dir, exist_ok=True)
                file_dest = os.path.join(dest_dir, file_name)
                shutil.copy(filename_heatmap, file_dest)
                print(file_dest)

print('OK!')