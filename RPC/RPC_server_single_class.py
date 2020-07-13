'''
  RPC Service
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
from xmlrpc.server import SimpleXMLRPCServer
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
from LIBS.ImgPreprocess.my_preprocess import do_preprocess
from LIBS.Generator.my_images_generator_2d import my_gen_img_tensor
import my_config

def predict_softmax(img_preprocessed):
    prob_np = []
    prob = []
    pred = []

    for dict1 in dicts_models:
        img_tensor = my_gen_img_tensor(img_preprocessed,
                                       image_shape=dict1['input_shape'])
        prob1 = dict1['model'].predict_on_batch(img_tensor)
        prob1 = np.mean(prob1, axis=0)  # batch mean, test time img aug
        pred1 = prob1.argmax(axis=-1)

        prob_np.append(prob1)  #  numpy  weight avg prob_total

        prob.append(prob1.tolist())    #担心XMLRPC numpy
        pred.append(int(pred1))   # numpy int64, int  XMLRPC

    list_weights = []  # the prediction weights of models
    for dict1 in dicts_models:
        list_weights.append(dict1['model_weight'])

    prob_total = np.average(prob_np, axis=0, weights=list_weights)
    pred_total = prob_total.argmax(axis=-1)

    prob_total = prob_total.tolist()  #RPC Service can not pass numpy variable
    pred_total = int(pred_total)     # 'numpy.int64'  XMLRPC

    # correct_model_no is used for choosing which model to generate CAM
    # on extreme condition: average softmax prediction class is not in every model's prediction class
    correct_model_no = 0
    for i, pred1 in enumerate(pred):
        if pred1 == pred_total:
            correct_model_no = i    #start from 0
            break

    return prob, pred, prob_total, pred_total, correct_model_no

#command parameters: predict class type no and port number
if len(sys.argv) == 3:  # sys.argv[0]  exe file itself
    reference_class = str(sys.argv[1])
    port = int(sys.argv[2])
else:
    reference_class = '1'  # DR
    port = 5001

#region define models
model_dir = my_config.dir_deploy_models
dicts_models = []

#left right eye
if reference_class == '-4':
    dict1 = {'model_file': model_dir + 'LeftRight/MobileNetV2-005-0.997.hdf5',
              'input_shape': (224, 224, 3), 'model_weight': 1}
    dicts_models.append(dict1)

    # dict1 = {'model_file': DIR_MODELS + 'LeftRight/NasnetMobile-007-0.991.hdf5',
    #          'model_weight': 1, 'input_shape': (224, 224, 3)}
    # models.append(dict1)

#gradable
if reference_class == '-3':
    dict1 = {'model_file': model_dir + 'Gradable/MobileNetV2-005-0.946.hdf5',
              'input_shape': (224, 224, 3), 'model_weight': 1}
    dicts_models.append(dict1)

    # dict1 = {'model_file': DIR_MODELS + 'Gradable/NasnetMobile-006-0.945.hdf5',
    #          'input_shape': (224, 224, 3), 'model_weight': 1}
    # models.append(dict1)

#DR
if reference_class == '1':
    dict1 = {'model_file': os.path.join(model_dir, 'DR_english_2classes/InceptionResnetV2-006-0.980.hdf5'),
              'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict1)

    dict1 = {'model_file': os.path.join(model_dir, 'DR_english_2classes/Xception-006-0.980.hdf5'),
              'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict1)

#endregion

#load models
for dict1 in dicts_models:
    model_file = dict1['model_file']
    print('%s load start!' % (model_file))
    # ValueError: Unknown activation function:relu6  MobileNet V2
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        dict1['model'] = keras.models.load_model(model_file, compile=False)

    if 'input_shape' not in dict1:
        if len(dict1['model'].input_shape) == 4: #(batch, height, width, channel)
            dict1['input_shape'] = dict1['model'].input_shape[1:]
        else:
            dict1['input_shape'] = (299, 299, 3)

    print('%s load complte!' % (model_file))


#region test mode
if my_config.debug_mode:
    img_source = '/tmp1/66b17a1e-a74d-11e8-94f6-6045cb817f5b.jpg'
    if os.path.exists(img_source):
        img_file_preprocessed = '/tmp1/preprocessed.jpg'
        img1 = do_preprocess(img_source, my_config.preprocess_img_size, img_file_dest=img_file_preprocessed)
        prob_list, pred_list, prob_total, pred_total, correct_model_no = predict_softmax(img_file_preprocessed)
        print(prob_total)
    else:
        print('file:', img_source, ' does not exist.')
#endregion


server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
server.register_function(predict_softmax, "predict_softmax")
server.serve_forever()

