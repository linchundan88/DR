#based on https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
import cv2
import uuid
from LIBS.Neural_Networks.Utils.my_utils import get_last_conv_layer_name
from tensorflow.python.framework.ops import disable_eager_execution

# in the future, I will replace K.gradient function with tf.GradientTape, g.gradnent.
disable_eager_execution()


class My_grad_cam():
    def __init__(self, model):
        if isinstance(model, str):
            print('prepare to load model!')
            self.model = keras.models.load_model(model, compile=False)
            print('loading model complete!')
        else:
            self.model = model

        self.layer_name = get_last_conv_layer_name(self.model)

    def __grad_cam(self, input_model, image, cls, layer_name):

        """GradCAM method for visualizing input saliency."""
        y_c = input_model.output[0, cls]
        conv_output = input_model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]
        # Normalize if necessary
        # grads = normalize(grads)
        gradient_function = K.function([input_model.input], [conv_output, grads])
        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # Process CAM
        cam = cv2.resize(cam, (image.shape[1], image.shape[2]), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max != 0:
            cam = cam / cam_max
        return cam

    def gen_heatmap(self, img_input, pred_class, gen_gif=True, gif_fps=1,
                    blend_original_image=True, norm_reverse=True, base_dir_save='/tmp'):

        gradcam = self.__grad_cam(self.model, img_input, pred_class, self.layer_name)
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)

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

            filename_CAM = os.path.join(base_dir_save, str_uuid, 'Grad_CAM{}.jpg'.format(pred_class))
            cv2.imwrite(filename_CAM, jetcam)

            import imageio
            mg_paths = [filename_original, filename_CAM]
            gif_images = []
            for path in mg_paths:
                gif_images.append(imageio.imread(path))
            img_file_gif = os.path.join(base_dir_save, str_uuid, 'Grad_CAM{}.gif'.format(pred_class))
            imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
            return img_file_gif
        else:
            if blend_original_image:
                image_original = img_input[0, :]
                from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
                if norm_reverse:
                    image_original = input_norm_reverse(image_original)
                image_original = image_original.astype(np.uint8)

                jetcam = (np.float32(jetcam) + image_original) / 2

            filename_CAM = os.path.join(base_dir_save, str_uuid, 'Grad_CAM{}.jpg'.format(pred_class))
            cv2.imwrite(filename_CAM, jetcam)

            return filename_CAM

