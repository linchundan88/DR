'''
    RPC Service for CAM, grad-cam  and Grad-CAM++
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from xmlrpc.server import SimpleXMLRPCServer
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import shutil
import uuid
import my_config
from LIBS.Generator.my_images_generator_2d import my_gen_img_tensor
from LIBS.Neural_Networks.Heatmaps.CAM import my_helper_cam, my_helper_grad_cam, my_helper_grad_cam_plusplus
from LIBS.ImgPreprocess.my_preprocess import do_preprocess
from tensorflow import keras

DIR_MODELS = my_config.dir_deploy_models
DIR_TMP = os.path.join(my_config.dir_tmp, 'CAM')
if not os.path.exists(DIR_TMP):
    os.makedirs(DIR_TMP)

# this method is deployed to RPC service
def server_cam(model_no, image_file_preprocessed, pred, cam_relu=True,
               blend_original_image=True):

    img_input = my_gen_img_tensor(image_file_preprocessed,
                                image_shape=dicts_models[model_no]['input_shape'])
    if heatmap_type == 'CAM':
        filename_heatmap = list_my_cam[model_no].gen_heatmap(img_input=img_input, pred_class=pred,
                                    cam_relu=cam_relu, blend_original_image=blend_original_image)
    if heatmap_type == 'grad_cam':
        filename_heatmap = list_my_grad_cam[model_no].gen_heatmap(img_input=img_input, pred_class=pred,
                                                blend_original_image=blend_original_image)
    if heatmap_type == 'gradcam_plus':
        filename_heatmap = list_my_grad_cam_plusplus[model_no].gen_heatmap(img_input=img_input, pred_class=pred,
                                                blend_original_image=blend_original_image)

    str_uuid = str(uuid.uuid1())
    if blend_original_image:
        filename_CAM_dst = os.path.join(DIR_TMP, str_uuid, 'CAM{}.gif'.format(pred))
    else:
        filename_CAM_dst = os.path.join(DIR_TMP, str_uuid, 'CAM{}.jpg'.format(pred))
    if not os.path.exists(os.path.dirname(filename_CAM_dst)):
        os.makedirs(os.path.dirname(filename_CAM_dst))
    shutil.copy(filename_heatmap, filename_CAM_dst)

    return filename_CAM_dst


#region command parameters: class type no and port no
if len(sys.argv) != 4:  # sys.argv[0]  exe file itself
    reference_class = '1'  # DR english 2 classes
    port = 5100
    heatmap_type = 'CAM' #CAM grad_cam, gradcam_plus
else:
    reference_class = str(sys.argv[1])
    port = int(sys.argv[2])
    heatmap_type = sys.argv[3]
#endregion

dicts_models = []
if reference_class == '1':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'DR_english_2classes/InceptionResnetV2-006-0.980.hdf5'),
              'input_shape': (299, 299, 3)}
    dicts_models.append(dict1)
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'DR_english_2classes/Xception-006-0.980.hdf5'),
              'input_shape': (299, 299, 3)}
    dicts_models.append(dict1)

#Gradable
if reference_class == '11':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'Gradable/MobileNetV2-005-0.946.hdf5'),
              'input_shape': (224, 224, 3)}
    dicts_models.append(dict1)
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'Gradable/NasnetMobile-006-0.945.hdf5'),
             'input_shape': (224, 224, 3)}
    dicts_models.append(dict1)

#left right eye
if reference_class == '12':
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'LeftRight/MobileNetV2-005-0.997.hdf5'),
              'input_shape': (224, 224, 3)}
    dicts_models.append(dict1)
    dict1 = {'model_file': os.path.join(DIR_MODELS, 'LeftRight/NasnetMobile-007-0.991.hdf5'),
              'input_shape': (224, 224, 3)}
    dicts_models.append(dict1)


for dict1 in dicts_models:
    print('prepare to load model:' + dict1['model_file'])
    dict1['model'] = keras.models.load_model(dict1['model_file'], compile=False)
    print('model load complete!')

if heatmap_type == 'CAM':
    list_my_cam = []
    for dict_model in dicts_models:
        my_cam = my_helper_cam.My_cam(model=dict_model['model'])
        list_my_cam.append(my_cam)
if heatmap_type == 'grad_cam':
    list_my_grad_cam = []
    for dict_model in dicts_models:
        my_grad_cam = my_helper_grad_cam.My_grad_cam(model=dict_model['model'])
        list_my_grad_cam.append(my_grad_cam)
if heatmap_type == 'gradcam_plus':
    list_my_grad_cam_plusplus = []
    for dict_model in dicts_models:
        my_grad_cam_plusplus = my_helper_grad_cam_plusplus.My_grad_cam_plusplus(model=dict_model['model'])
        list_my_grad_cam_plusplus.append(my_grad_cam_plusplus)
#endregion

# region test code
if my_config.debug_mode:
    img_source = '/tmp1/66b17a1e-a74d-11e8-94f6-6045cb817f5b.jpg'
    img_file_preprocessed = '/tmp1/aaa.jpg'
    img1 = do_preprocess(img_source, my_config.preprocess_img_size, img_file_dest=img_file_preprocessed)
    if os.path.exists(img_source):
        filename_CAM1 = server_cam(model_no=0, image_file_preprocessed=img_file_preprocessed,
            pred=1, cam_relu=True, blend_original_image=True)
        print('OK')

#endregion

#region start service
# server = SimpleXMLRPCServer(("localhost", port))
server = SimpleXMLRPCServer(("0.0.0.0", port))
print("Listening on port: ", str(port))
server.register_function(server_cam, "server_cam")
server.serve_forever()

#endregion