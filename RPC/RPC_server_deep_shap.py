'''
provide RPC service for shap_deep_Explainer
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  #DeepShap do not compatible with tf2.2 so far
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import numpy as np
from LIBS.ImgPreprocess.my_preprocess import do_preprocess
from LIBS.Generator.my_images_generator_2d import my_gen_img_tensor

#command parameters: gpu_no  port no
#one server provides service for different types
if len(sys.argv) != 3:  # sys.argv[0]  exe file itself
    gpu_no = '0'
    port = 5100
else:
    gpu_no = str(sys.argv[1])
    port = int(sys.argv[2])

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

#limit gpu memory usage
# import tensorflow as tf
# helper = tf.ConfigProto()
# helper.gpu_options.per_process_gpu_memory_fraction = 0.5
# from keras.backend.tensorflow_backend import set_session
# set_session(tf.Session(helper=helper))

from xmlrpc.server import SimpleXMLRPCServer
from LIBS.Neural_Networks.Heatmaps.deepshap.my_helper_deepshap import My_deepshap

reference_file = os.path.join(os.path.abspath('.'), 'reference.npy')
num_reference = 24  # background  24

import my_config
dir_tmp = os.path.join(my_config.dir_tmp, 'deep_shap')

model_dir = my_config.dir_deploy_models
dicts_models = []
#xception batch_size:6, inception-v3 batch_size:24, InceptionResnetV2 batch_size:12
dict1 = {'model_file': os.path.join(model_dir, 'DR_english_2classes/InceptionResnetV2-006-0.980.hdf5'),
         'input_shape': (299, 299, 3), 'batch_size': 8}
dicts_models.append(dict1)
dict1 = {'model_file': os.path.join(model_dir, 'DR_english_2classes/Xception-006-0.980.hdf5'),
         'input_shape': (299, 299, 3), 'batch_size': 6}
dicts_models.append(dict1)

my_deepshap = My_deepshap(dicts_models, reference_file=reference_file, num_reference=num_reference)


def server_shap_deep_explainer(model_no, img_source,
                     ranked_outputs=1, blend_original_image=False):
    image_shape = dicts_models[model_no]['input_shape']
    img_input = my_gen_img_tensor(
            img_source, image_shape=image_shape)

    list_classes, list_images = my_deepshap.shap_deep_explainer(
        model_no=model_no, num_reference=num_reference,
        img_input=img_input, ranked_outputs=ranked_outputs,
        blend_original_image=blend_original_image, base_dir_save=dir_tmp)

    return list_classes, list_images


if my_config.debug_mode:
    import time
    img_source = '/tmp1/66b17a1e-a74d-11e8-94f6-6045cb817f5b.jpg'
    img_file_preprocessed = '/tmp1/preprocessed.jpg'
    img1 = do_preprocess(img_source, my_config.preprocess_img_size, img_file_dest=img_file_preprocessed)

    if os.path.exists(img_source):
        input_shape = dicts_models[0]['input_shape']
        img_input = my_gen_img_tensor(
            img_file_preprocessed, image_shape=input_shape)
        prob = dicts_models[0]['model'].predict(img_input)
        pred = np.argmax(prob)
        print(pred)

        print('model1')
        #first time take longer
        for i in range(1):
            print(time.time())
            list_classes, list_images = server_shap_deep_explainer(model_no=0,
                img_source=img_file_preprocessed, ranked_outputs=1,
                blend_original_image=True, norm_reverse=True)
            print(time.time())
            print(list_images)


# server = SimpleXMLRPCServer(("localhost", port))
server = SimpleXMLRPCServer(("0.0.0.0", port))
print("Listening on port: ", str(port))
server.register_function(server_shap_deep_explainer, "server_shap_deep_explainer")
server.serve_forever()
