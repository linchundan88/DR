'''based on https://keras.io/examples/vision/integrated_gradients/'''

import os
import numpy as np
import tensorflow as tf
import cv2
import uuid

class My_gradients():
    def __init__(self, model):
        if isinstance(model, str):
            print('prepare to load model!')
            self.model = keras.models.load_model(model, compile=False)
            print('loading model complete!')
        else:
            self.model = model

    def __get_gradients(self, img_input, top_pred_idx):
        """Computes the gradients of outputs w.r.t input image.

        Args:
            img_input: 4D image tensor
            top_pred_idx: Predicted label for the input image

        Returns:
            Gradients of the predictions w.r.t img_input  4D
        """
        images = tf.cast(img_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(images)
            preds = self.model(images)
            top_class = preds[:, top_pred_idx]

        grads = tape.gradient(top_class, images)
        return grads


    def __get_integrated_gradients(self, img_input, top_pred_idx, baseline=None, num_steps=50):
        """Computes Integrated Gradients for a predicted label.

        Args:
            img_input (ndarray): 4d image
            top_pred_idx: Predicted label for the input image
            baseline (ndarray): The baseline image to start with for interpolation
            num_steps: Number of interpolation steps between the baseline
                and the input used in the computation of integrated gradients. These
                steps along determine the integral approximation error. By default,
                num_steps is set to 50.

        Returns:
            Integrated gradients w.r.t input image
        """

        if img_input.ndim == 4:
            img_input = img_input[0]

        # If baseline is not provided, start with a black image
        # having same size as the input image.
        if baseline is None:
            baseline = np.zeros(img_input.shape).astype(np.float32)
        else:
            baseline = baseline.astype(np.float32)

        # 1. Do interpolation.
        img_input = img_input.astype(np.float32)
        interpolated_image = [
            baseline + (step / num_steps) * (img_input - baseline)
            for step in range(num_steps + 1)
        ]
        interpolated_image = np.array(interpolated_image).astype(np.float32)

        # 2. Preprocess the interpolated images
        # interpolated_image = np.expand_dims(interpolated_image, axis=0)

        # 3. Get the gradients
        grads = []
        for i, img in enumerate(interpolated_image):
            img = tf.expand_dims(img, axis=0)
            grad = self.__get_gradients(img, top_pred_idx=top_pred_idx)
            grads.append(grad[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        # 4. Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        # 5. Calculate integrated gradients and return
        integrated_grads = (img_input - baseline) * avg_grads
        return integrated_grads

    def __random_baseline_integrated_gradients(self,
                                               img_input, top_pred_idx, num_steps=50, num_runs=2):
        """Generates a number of random baseline images.

        Args:
            img_input (ndarray): 4D image
            top_pred_idx: Predicted label for the input image
            num_steps: Number of interpolation steps between the baseline
                and the input used in the computation of integrated gradients. These
                steps along determine the integral approximation error. By default,
                num_steps is set to 50.
            num_runs: number of baseline images to generate

        Returns:
            Averaged integrated gradients for `num_runs` baseline images
        """
        # 1. List to keep track of Integrated Gradients (IG) for all the images
        integrated_grads = []

        # 2. Get the integrated gradients for all the baselines
        for run in range(num_runs):
            baseline = np.random.random(img_input[0].shape) * 255
            igrads = self.__get_integrated_gradients(
                img_input=img_input,
                top_pred_idx=top_pred_idx,
                baseline=baseline,
                num_steps=num_steps,
            )
            integrated_grads.append(igrads)

        # 3. Return the average integrated gradients for the image
        integrated_grads = tf.convert_to_tensor(integrated_grads)
        return tf.reduce_mean(integrated_grads, axis=0)


    def gen_gradients(self, img_input, pred_class,
                    gen_gif=True, gif_fps=1,
                     norm_reverse=True, base_dir_save='/tmp'):

        gradients = self.__get_gradients(img_input, pred_class).numpy()
        if gradients.ndim == 4:
            gradients = gradients[0]

        str_uuid = str(uuid.uuid1())
        os.makedirs(os.path.join(base_dir_save, str_uuid), exist_ok=True)

        gradients = np.mean(gradients, axis=-1) #(299,299,3) ->(299,299)
        gradients = np.maximum(gradients, 0)  # ReLU
        gradients /= np.max(gradients)  # heatmap:0-1
        # cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        filename_gradient = os.path.join(base_dir_save, str_uuid, 'gradient{}.jpg'.format(1))
        cv2.imwrite(filename_gradient, gradients * 255)

        if gen_gif:
            image_original = img_input[0, :]
            from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
            if norm_reverse:
                image_original = input_norm_reverse(image_original)
            image_original = image_original.astype(np.uint8)

            filename_original = os.path.join(base_dir_save, str_uuid, 'original.jpg')
            cv2.imwrite(filename_original, image_original)

            import imageio
            mg_paths = [filename_original, filename_gradient]
            gif_images = []
            for path in mg_paths:
                gif_images.append(imageio.imread(path))
            img_file_gif = os.path.join(base_dir_save, str_uuid, 'gradient{}.gif'.format(pred_class))
            imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
            return img_file_gif

        else:
            return filename_gradient

    def gen_integrated_gradients(self, img_input, pred_class,
                    gen_gif=True, gif_fps=1,
                     norm_reverse=True, base_dir_save='/tmp'):

        integrated_gradients = self.__get_integrated_gradients(img_input, pred_class).numpy()
        if integrated_gradients.ndim == 4:
            integrated_gradients = integrated_gradients[0]

        str_uuid = str(uuid.uuid1())
        os.makedirs(os.path.join(base_dir_save, str_uuid), exist_ok=True)

        integrated_gradients = np.mean(integrated_gradients, axis=-1)  # (299,299,3) ->(299,299)
        integrated_gradients = np.maximum(integrated_gradients, 0)  # ReLU
        integrated_gradients /= np.max(integrated_gradients)  # heatmap:0-1
        # cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        filename_gradient = os.path.join(base_dir_save, str_uuid, 'integrated_gradients{}.jpg'.format(1))
        cv2.imwrite(filename_gradient, integrated_gradients * 255)

        if gen_gif:
            image_original = img_input[0, :]
            from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
            if norm_reverse:
                image_original = input_norm_reverse(image_original)
            image_original = image_original.astype(np.uint8)

            filename_original = os.path.join(base_dir_save, str_uuid, 'original.jpg')
            cv2.imwrite(filename_original, image_original)

            import imageio
            mg_paths = [filename_original, filename_gradient]
            gif_images = []
            for path in mg_paths:
                gif_images.append(imageio.imread(path))
            img_file_gif = os.path.join(base_dir_save, str_uuid, 'integrated_gradients{}.gif'.format(pred_class))
            imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
            return img_file_gif

        else:
            return filename_gradient


if __name__ =='__main__':
    model_file = '/tmp5/models_2020_6_19/DR_english/v1/InceptionResnetV2-004-0.984.hdf5'
    from tensorflow import keras
    model = keras.models.load_model(model_file, compile=False)

    import LIBS.Generator.my_images_generator_2d
    img_preprocess ='/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R2/清晰可见/68006683-a74d-11e8-94f6-6045cb817f5b.jpg'
    input_shape = (299, 299, 3)
    img_input = LIBS.Generator.my_images_generator_2d.my_gen_img_tensor(img_preprocess,
                        image_shape=input_shape)

    my_gradient = My_gradients(model)
    import time
    print(time.time())
    file_name_aa = my_gradient.gen_gradients(img_input, 1,
                  gen_gif=True, gif_fps=1,
                  norm_reverse=True, base_dir_save='/tmp')
    # file_name_aa = my_gradient.gen_integrated_gradients(img_input, 1,
    #               gen_gif=True, gif_fps=1,
    #               norm_reverse=True, base_dir_save='/tmp')
    print(time.time())
    print('aaaa')

    print(time.time())
    # file_name_aa = my_gradient.gen_gradients(img_input, 1,
    #               gen_gif=True, gif_fps=1,
    #               norm_reverse=True, base_dir_save='/tmp')
    file_name_aa = my_gradient.gen_integrated_gradients(img_input, 1,
                  gen_gif=True, gif_fps=1,
                  norm_reverse=True, base_dir_save='/tmp')
    print(time.time())

    print('bbb')

