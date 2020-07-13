'''

'''
import tensorflow.compat.v1 as tf  #because of DeepShap is not compatible with tf2
tf.disable_v2_behavior()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import LIBS.Generator.my_images_generator_2d
import numpy as np
import LIBS.ImgPreprocess.my_image_helper
from LIBS.Neural_Networks.Heatmaps.deepshap.my_helper_deepshap import My_deepshap
import pandas as pd
from LIBS.ImgPreprocess import my_preprocess
import shutil

reference_file = 'reference.npy'
num_reference = 24

model_dir = '/tmp5/models_2020_6_19/DR_english/v1'
dicts_models = []
# xception batch_size:6, inception-v3 batch_size:24, InceptionResnetV2 batch_size:12
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-004-0.984.hdf5'),
               'input_shape': (299, 299, 3), 'batch_size': 12}
dicts_models.append(dict_model1)
# dict_model1 = {'model_file': os.path.join(model_dir, 'Xception-004-0.984.hdf5'),
#                'input_shape': (299, 299, 3), 'batch_size': 8}
# dicts_models.append(dict_model1)
# dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionV3-004-0.982.hdf5.hdf5'),
#                'input_shape': (299, 299, 3),  'batch_size': 24}
# dicts_models.append(dict_model1)

my_deepshap = My_deepshap(dicts_models, reference_file=reference_file, num_reference=num_reference)

dir_save_tmp = '/tmp/deepshap'
dir_save_results = '/tmp5/heatmap_2020_5_22/Stage/Deepshap/InceptionResnetV2/no_blend'
dir_preprocess = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384'

blend_image = False

model_no = 0
image_shape = dicts_models[0]['input_shape']

train_type = 'DR_english'
data_version = 'v1'

for model_no in range(len(dicts_models)):
    for predict_type_name in ['split_patid_train', 'split_patid_valid', 'split_patid_test']:
        save_dir = os.path.join(dir_save_results, predict_type_name)
        filename_csv = os.path.join(os.path.abspath('../../../'),
                'datafiles', train_type, '{}_{}.csv'.format(predict_type_name, data_version))

        df = pd.read_csv(filename_csv)
        for _, row in df.iterrows():
            image_file = row['images']
            image_label = int(row['labels'])

            # region predict label
            preprocess = False
            if preprocess:
                img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
                img_input = LIBS.Generator.my_images_generator_2d.my_gen_img_tensor(img_preprocess, image_shape=image_shape)
            else:
                img_input = LIBS.Generator.my_images_generator_2d.my_gen_img_tensor(image_file, image_shape=image_shape)

            prob = dicts_models[model_no]['model'].predict(img_input)
            class_predict = np.argmax(prob)
            #endregion

            if (class_predict == 1 and image_label == 1) or\
                    (class_predict == 1 and image_label == 0):
                list_classes, list_images = my_deepshap.shap_deep_explain(
                    model_no=model_no, num_reference=num_reference,
                    img_input=img_input, ranked_outputs=1,
                    blend_original_image=blend_image, base_dir_save=dir_save_tmp)

                if class_predict == 1 and image_label == 1:
                    file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, str(model_no), '1_1/'))
                if class_predict == 1 and image_label == 0:
                    file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, str(model_no), '0_1/'))

                assert dir_preprocess not in file_dest, 'heatmap file should not overwrite preprocess image file'

                if blend_image:
                    filename, file_ext = os.path.splitext(file_dest)
                    file_dest = filename + '.gif'
                os.makedirs(os.path.dirname(file_dest), exist_ok=True)
                shutil.copy(list_images[0], file_dest)
                print(file_dest)

print('OK')