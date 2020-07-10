import os
from LIBS.DataPreprocess import my_data
import pandas as pd


#region read files, extract labels based on subdirectories, write to csv file.
TRAIN_TYPE = 'DR_english'
DATA_VERSION = 'v1'

dir_original = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/original/'
dir_preprocess = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384'

filename_csv = os.path.join(os.path.abspath('..'),
               'datafiles', TRAIN_TYPE, 'split_patid_{}.csv'.format(DATA_VERSION))
dict_mapping = {'激光斑': 1, 'R0': 0, 'R1': 0, 'R2': 1, 'R3': 1, 'R3s': 1}
my_data.write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping, match_type='partial')

#endregion

#region split dataset

train_files, train_labels, valid_files, valid_labels, test_files, test_labels = \
    my_data.split_dataset(filename_csv, valid_ratio=0.1, test_ratio=0.15)
filename_csv_train = os.path.join(os.path.abspath('..'),
                'datafiles', TRAIN_TYPE, 'split_patid_train_{}.csv'.format(DATA_VERSION))
filename_csv_valid = os.path.join(os.path.abspath('..'),
                'datafiles', TRAIN_TYPE, 'split_patid_valid_{}.csv'.format(DATA_VERSION))
filename_csv_test = os.path.join(os.path.abspath('..'),
                'datafiles', TRAIN_TYPE, 'split_patid_test_{}.csv'.format(DATA_VERSION))

my_data.write_images_labels_csv(train_files, train_labels, filename_csv_train)
my_data.write_images_labels_csv(valid_files, valid_labels, filename_csv_valid)
my_data.write_images_labels_csv(test_files, test_labels, filename_csv_test)

'''
filename_pkl_train = os.path.join(os.path.abspath('.'),
                    'pat_id_pkl', 'split_patid_train.pkl')
list_patient_id_train = pickle.load(open(filename_pkl_train, 'rb'))
filename_pkl_valid = os.path.join(os.path.abspath('.'),
                    'pat_id_pkl', 'split_patid_valid.pkl')
list_patient_id_valid = pickle.load(open(filename_pkl_valid, 'rb'))
filename_pkl_test = os.path.join(os.path.abspath('.'),
                    'pat_id_pkl', 'split_patid_test.pkl')
list_patient_id_test = pickle.load(open(filename_pkl_test, 'rb'))

from LIBS.DataPreprocess.my_data_patiend_id import write_csv_list_patient_id
write_csv_list_patient_id(filename_csv, filename_csv_train, list_patient_id_train,
                          field_columns=['images', 'labels'])
write_csv_list_patient_id(filename_csv, filename_csv_valid, list_patient_id_valid,
                          field_columns=['images', 'labels'])
write_csv_list_patient_id(filename_csv, filename_csv_test, list_patient_id_test,
                          field_columns=['images', 'labels'])
'''

for file_csv in [filename_csv_train, filename_csv_valid, filename_csv_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    for label in [0, 1]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))


#endregion



print('OK')