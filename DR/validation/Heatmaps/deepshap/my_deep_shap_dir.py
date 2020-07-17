'''

'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import LIBS.Generator.my_images_generator_2d
import numpy as np
import LIBS.ImgPreprocess.my_image_helper
from LIBS.Neural_Networks.Heatmaps.deepshap.my_helper_deepshap import My_deepshap
import pandas as pd
from LIBS.ImgPreprocess import my_preprocess
import shutil

reference_file = 'ref_dr.npy'
num_reference = 24
dir_save_tmp = '/tmp/deepshap'

do_preprocess = False
gen_csv = True

dir_original = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/original'
dir_preprocess = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384'
dir_dest = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/results/CAM'

from LIBS.ImgPreprocess import my_preprocess_dir
if do_preprocess:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
                                        image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

filename_csv = os.path.join(dir_dest, 'csv', 'predict_dir.csv')
if gen_csv:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel
    write_csv_dir_nolabel(filename_csv, dir_preprocess)


model_dir = '/tmp5/models_2020_6_19/DR_english/v1'
dicts_models = []
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-004-0.984.hdf5'),
               'input_shape': (299, 299, 3), 'batch_size': 12}
dicts_models.append(dict_model1)

my_deepshap = My_deepshap(dicts_models, reference_file=reference_file, num_reference=num_reference)


model_no = 0
image_shape = dicts_models[model_no]['input_shape']
blend_original_image = False

df = pd.read_csv(filename_csv)
for _, row in df.iterrows():
    image_file = row['images']
    preprocess = False
    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
        img_input = LIBS.Generator.my_images_generator_2d.my_gen_img_tensor(img_preprocess,
                                                                            image_shape=image_shape)
    else:
        img_input = LIBS.Generator.my_images_generator_2d.my_gen_img_tensor(image_file,
                                                                            image_shape=image_shape)
    prob = dicts_models[model_no]['model'].predict(img_input)
    class_predict = np.argmax(prob)

    if class_predict == 1:
        list_classes, list_images = my_deepshap.shap_deep_explainer(
            model_no=model_no, num_reference=num_reference,
            img_input=img_input, ranked_outputs=1,
            blend_original_image=blend_original_image, norm_reverse=True,
            base_dir_save=dir_save_tmp)

        file_dest = image_file.replace(dir_preprocess, os.path.join(dir_dest, 'deepshap'))
        assert dir_preprocess in file_dest, 'heatmap file overwrite preprocess image file'
        if blend_original_image:
            filename, file_ext = os.path.splitext(file_dest)
            file_dest = filename + '.gif'
        os.makedirs(os.path.dirname(file_dest), exist_ok=True)
        shutil.copy(list_images[0], file_dest)
        print(file_dest)


print('OK')