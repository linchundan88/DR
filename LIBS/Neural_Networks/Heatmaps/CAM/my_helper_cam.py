
import os
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow import keras
import cv2
import uuid

class My_cam():
    def __init__(self, model):
        if isinstance(model, str):
            print('prepare to load model!')
            self.model = keras.models.load_model(model, compile=False)
            print('loading model complete!')
        else:
            self.model = model

        self.model_last_conv, self.all_amp_layer_weights = self.__modify_model()

    def __modify_model(self, last_layer=-1):
        # get the last conv layer before global average pooling
        for i in range(len(self.model.layers) - 1, -1, -1):
            if isinstance(self.model.layers[i], Conv2D) or \
                    isinstance(self.model.layers[i], Activation) or \
                    isinstance(self.model.layers[i], SeparableConv2D) or \
                    isinstance(self.model.layers[i], Concatenate):  # inception v3 Concatenate
                last_conv_layer = i
                break

        # get AMP layer weights
        last_layer_weights = self.model.layers[last_layer].get_weights()[0]

        # extract wanted output
        output_model = Model(inputs=self.model.input,
                             outputs=(self.model.layers[last_conv_layer].output))

        return output_model, last_layer_weights

    def gen_heatmap(self, img_input, pred_class, cam_relu=True,
                    gen_gif=True, gif_fps=1,
                    blend_original_image=True, norm_reverse=True, base_dir_save='/tmp'):

        height, width = img_input.shape[1], img_input.shape[2]

        last_conv_output = self.model_last_conv.predict(img_input)
        last_conv_output = np.squeeze(last_conv_output)

        # get AMP layer weights
        amp_layer_weights = self.all_amp_layer_weights[:, pred_class]  # dim: (2048,)

        if cam_relu:
            amp_layer_weights = np.maximum(amp_layer_weights, 0)

        cam_small = np.dot(last_conv_output, amp_layer_weights)  # dim: 224 x 224
        cam = cv2.resize(cam_small, (width, height))  # 14*14 224*224
        cam = np.maximum(cam, 0)  # ReLU
        heatmap = cam / np.max(cam)  # heatmap:0-1

        # cam:0-255
        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        str_uuid = str(uuid.uuid1())
        os.makedirs(os.path.join(base_dir_save, str_uuid), exist_ok=True)

        if gen_gif:
            image_original = img_input[0, :]
            from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
            if norm_reverse:
                image_original = input_norm_reverse(image_original)
            image_original = image_original.astype(np.uint8)

            filename_original = os.path.join(base_dir_save, str_uuid, 'original.jpg')
            cv2.imwrite(filename_original, image_original)

            filename_CAM = os.path.join(base_dir_save, str_uuid, 'CAM{}.jpg'.format(pred_class))
            cv2.imwrite(filename_CAM, cam)

            import imageio
            mg_paths = [filename_original, filename_CAM]
            gif_images = []
            for path in mg_paths:
                gif_images.append(imageio.imread(path))
            img_file_gif = os.path.join(base_dir_save, str_uuid, 'CAM{}.gif'.format(pred_class))
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

                cam = (np.float32(cam) + image_original) / 2

            filename_CAM = os.path.join(base_dir_save, str_uuid, 'CAM{}.jpg'.format(pred_class))
            cv2.imwrite(filename_CAM, cam)

        return filename_CAM



