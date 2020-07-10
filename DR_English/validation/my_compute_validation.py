import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from LIBS.DataPreprocess.my_data import get_images_labels
from LIBS.DataValidation.my_multi_class import compute_confusion_matrix, op_files_multiclass
import pandas as pd
import pickle
from LIBS.DLP.my_predict_helper import do_predict

COMPUTE_CONFUSIN_MATRIX = True
COMPUTE_DIR_FILES = True

dir_original = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/original/'
dir_preprocess = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384'

dir_dest = '/tmp3/DR_english/2020_6_22'

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

train_type = 'DR_english'
data_version = 'v1'
filename_csv_train = os.path.join(os.path.abspath('..'),
                'datafiles', train_type, 'split_patid_train_{}.csv'.format(data_version))
filename_csv_valid = os.path.join(os.path.abspath('..'),
                'datafiles', train_type, 'split_patid_valid_{}.csv'.format(data_version))
filename_csv_test = os.path.join(os.path.abspath('..'),
                'datafiles', train_type, 'split_patid_test_{}.csv'.format(data_version))

# for filename_csv in [filename_csv_train, filename_csv_valid, filename_csv_test]:
for filename_csv in [filename_csv_test]:
    df = pd.read_csv(filename_csv)
    all_files, all_labels = get_images_labels(filename_csv_or_pd=df)

    prob_total, y_pred_total, prob_list, pred_list =\
        do_predict(dicts_models, filename_csv, argmax=True)

    dir_dest_confusion = os.path.join(dir_dest, train_type, 'confusion_matrix', 'files')
    dir_dest_predict_dir = os.path.join(dir_dest, train_type, 'dir')
    pkl_prob = os.path.join(dir_dest, train_type + '_prob.pkl')
    pkl_confusion_matrix = os.path.join(dir_dest, train_type + '_cf.pkl')

    os.makedirs(os.path.dirname(pkl_prob), exist_ok=True)
    with open(pkl_prob, 'wb') as file:
        pickle.dump(prob_total, file)

    # pkl_file = open(prob_pkl', 'rb')
    # prob_total = pickle.load(pkl_file)

    if COMPUTE_CONFUSIN_MATRIX:
        (cf_list, not_match_list, cf_total, not_match_total) = \
            compute_confusion_matrix(prob_list, dir_dest_confusion,
                all_files, all_labels, dir_preprocess=dir_preprocess, dir_original=dir_original)

        print(cf_total)
        os.makedirs(os.path.dirname(pkl_confusion_matrix), exist_ok=True)
        with open(pkl_confusion_matrix, 'wb') as file:
            pickle.dump(cf_total, file)

    if COMPUTE_DIR_FILES:
        op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
            dir_dest=dir_dest_predict_dir, dir_original=dir_original, keep_subdir=True)


print('OK')