#based on https://github.com/totti0223/gradcamplusplus/blob/master/gradcamutils.py

import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
import cv2
import uuid
from scipy.ndimage.interpolation import zoom
from LIBS.Neural_Networks.Utils.my_utils import get_last_conv_layer_name
from tensorflow.python.framework.ops import disable_eager_execution

# in the future, I will replace K.gradients function with tf.GradientTape, g.gradnent.
disable_eager_execution()

class My_grad_cam_plusplus():
    def __init__(self, model):
        if isinstance(model, str):
            print('prepare to load model!')
            self.model = keras.models.load_model(model, compile=False)
            print('loading model complete!')
        else:
            self.model = model

        layer_name = get_last_conv_layer_name(self.model)
        self.conv_output = self.model.get_layer(layer_name).output

    def __grad_cam_plusplus(self, img_input, pred_class):
        height, width = img_input.shape[1], img_input.shape[2]

        y_c = self.model.output[0, pred_class]
        grads = K.gradients(y_c, self.conv_output)[0]

        first = K.exp(y_c)*grads
        second = K.exp(y_c)*grads*grads
        third = K.exp(y_c)*grads*grads

        gradient_function = K.function([self.model.input], [y_c, first, second, third, self.conv_output, grads])
        y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img_input])
        global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
        alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))

        deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = zoom(cam, height / cam.shape[0])
        cam = cam / np.max(cam) # scale 0 to 1.0

        return cam


    def gen_heatmap(self, img_input, pred_class,
                    gen_gif=True, gif_fps=1,
                    blend_original_image=True, norm_reverse=True, base_dir_save='/tmp'):

        # gradcamplus: 0-1
        gradcamplus = self.__grad_cam_plusplus(img_input, pred_class=pred_class)
        # cam: 0-255
        cam = cv2.applyColorMap(np.uint8(255 * gradcamplus), cv2.COLORMAP_JET)

        str_uuid = str(uuid.uuid1())
        os.makedirs(os.path.join(base_dir_save, str_uuid), exist_ok=True)

        if gen_gif:
            image_original = img_input[0, :]
            from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
            if norm_reverse:
                image_original = input_norm_reverse(image_original)
            image_original = image_original.astype(np.uint8)

            filename_original = os.path.join(base_dir_save, str_uuid, 'original.jpg')
            cv2.imwrite(image_original, filename_original)

            filename_CAM = os.path.join(base_dir_save, str_uuid, 'GradCAM_PlusPlus{}.jpg'.format(pred_class))
            cv2.imwrite(filename_CAM, cam)

            import imageio
            mg_paths = [filename_original, filename_CAM]
            gif_images = []
            for path in mg_paths:
                gif_images.append(imageio.imread(path))
            img_file_gif = os.path.join(base_dir_save, str_uuid, 'GradCAM_PlusPlus{}.gif'.format(pred_class))
            imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
            return img_file_gif

        else:
            if blend_original_image:
                image_original = img_input[0, :]
                from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
                if norm_reverse:
                    image_original = input_norm_reverse(image_original)
                image_original = image_original.astype(np.uint8)

                image_original -= np.min(image_original)
                image_original = np.minimum(image_original, 255)

                cam = np.float32(cam) + np.float32(image_original)
                cam = 255 * cam / np.max(cam)

            filename_CAM = os.path.join(base_dir_save, str_uuid, 'GradCAM_PlusPlus{}.jpg'.format(pred_class))
            cv2.imwrite(filename_CAM, cam)

            return filename_CAM

