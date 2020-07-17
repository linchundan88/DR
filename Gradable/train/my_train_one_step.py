
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import collections
from LIBS.DLP.my_train_multiclass_helper import train_task_one_step

#region setting train type, data source
TRAIN_TYPE = 'Gradable'
DATA_VERSION = 'v1'
FILENAME_CSV_TRAIN = os.path.join(os.path.abspath('..'),
                'datafiles', TRAIN_TYPE, 'split_patid_train_{}.csv'.format(DATA_VERSION))
FILENAME_CSV_VALID = os.path.join(os.path.abspath('..'),
                'datafiles', TRAIN_TYPE, 'split_patid_valid_{}.csv'.format(DATA_VERSION))
FILENAME_CSV_TEST = os.path.join(os.path.abspath('..'),
                'datafiles', TRAIN_TYPE, 'split_patid_test_{}.csv'.format(DATA_VERSION))

MODEL_SAVE_BASEDIR = os.path.join('/tmp5/models_2020_6_19/', TRAIN_TYPE, DATA_VERSION)

#endregion

#region training parameters
'''
dataset1 
38716
0 33668
1 5048
5162
0 4459
1 703
7744
0 6688
1 1056
'''

WEIGHT_CLASS_START = np.array([1, 4])
WEIGHT_CLASS_END = np.array([1, 4])
BALANCE_RATIO = 0.93

CLASS_WEIGHT = {0: 1., 1: 1.3}
LABEL_SMOOTHING = 0.1

from imgaug import augmenters as iaa
IMGAUG_TRAIN_SEQ = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images

    iaa.Sometimes(0.9, iaa.ContrastNormalization((0.9, 1.1))),
    iaa.Sometimes(0.9, iaa.Add((-6, 6))),
    iaa.Sometimes(0.9, iaa.Affine(
        scale=(0.98, 1.02),
        translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
        rotate=(-15, 15),  # rotate by -10 to +10 degrees
    )),
])

BATCH_SIZE_TRAIN, BATCH_SIZE_VALID = 32, 64

USE_MULTIPROCESSING = True
WORKER = 4

VERBOSE = 1
#endregion

#region training
TRANSFER_SOURCE = 'Fundus'  #Fundus ImageNet
# TRANSFER_SOURCE = 'ImageNet'  # Fundus ImageNet

TRAIN_TIMES = 3
for i in range(TRAIN_TIMES):
    for MODEL_NAME in ['Xception', 'InceptionResnetV2', 'InceptionV3']:
        print("train time:{0}, model name:{1}".format(i, MODEL_NAME))
        MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_BASEDIR, str(i), MODEL_NAME)

        PLT_HISTORY_IMAGE_FILE = os.path.join(MODEL_SAVE_DIR, 'history_' + str(i)+MODEL_NAME+'.jpg')

        if MODEL_NAME == 'InceptionV3':
            IMAGE_SHAPE = (299, 299, 3)
            if TRANSFER_SOURCE == 'ImageNet':
                PRE_DEFINE_MODEL = MODEL_NAME
                MODEL_FILE = None
                ADD_TOP, CHANGE_TOP = True, False
            if TRANSFER_SOURCE == 'Fundus':
                MODEL_FILE = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'
                PRE_DEFINE_MODEL = None
                ADD_TOP, CHANGE_TOP = False, True

        if MODEL_NAME == 'InceptionResnetV2':
            IMAGE_SHAPE = (299, 299, 3)
            if TRANSFER_SOURCE == 'ImageNet':
                PRE_DEFINE_MODEL = MODEL_NAME
                MODEL_FILE = None
                ADD_TOP, CHANGE_TOP = True, False
            if TRANSFER_SOURCE == 'Fundus':
                MODEL_FILE = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'
                PRE_DEFINE_MODEL = None
                ADD_TOP, CHANGE_TOP = False, True

        if MODEL_NAME == 'Xception':
            IMAGE_SHAPE = (299, 299, 3)
            if TRANSFER_SOURCE == 'ImageNet':
                PRE_DEFINE_MODEL = MODEL_NAME
                MODEL_FILE = None
                ADD_TOP, CHANGE_TOP = True, False
            if TRANSFER_SOURCE == 'Fundus':
                MODEL_FILE = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'
                PRE_DEFINE_MODEL = None
                ADD_TOP, CHANGE_TOP = False, True

        EPOCH_FINETUNING = 5
        DICT_LR_FINETUNING = collections.OrderedDict()
        DICT_LR_FINETUNING['0'] = 1e-3
        DICT_LR_FINETUNING['1'] = 1e-4
        DICT_LR_FINETUNING['2'] = 1e-5
        DICT_LR_FINETUNING['3'] = 1e-6

        train_task_one_step(model_file=MODEL_FILE, pre_define_model=PRE_DEFINE_MODEL,
                            filename_csv_train=FILENAME_CSV_TRAIN,
                            filename_csv_valid=FILENAME_CSV_VALID, filename_csv_test=FILENAME_CSV_TEST,
                            add_top=ADD_TOP, change_top=CHANGE_TOP,
                            input_shape=IMAGE_SHAPE, imgaug_train_seq=IMGAUG_TRAIN_SEQ,
                            optimizer='adam', lookahead=False,
                            batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_TRAIN,
                            epoch=EPOCH_FINETUNING, dict_lr=DICT_LR_FINETUNING,
                            class_weight=CLASS_WEIGHT, label_smoothing=LABEL_SMOOTHING,
                            weight_class_start=WEIGHT_CLASS_START, weight_class_end=WEIGHT_CLASS_END,
                            balance_ratio=BALANCE_RATIO,
                            # specificity=0.9, sensitivity=0.9,
                            verbose=VERBOSE,
                            model_save_dir=MODEL_SAVE_DIR, model_name=MODEL_NAME,
                            config_file_realtime='config_file_realtime.json',
                            use_multiprocessing=USE_MULTIPROCESSING, workers=WORKER,
                            plt_history_image_file=PLT_HISTORY_IMAGE_FILE)

#endregion

print('OK!')


