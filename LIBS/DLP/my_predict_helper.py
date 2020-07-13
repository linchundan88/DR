import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import LIBS.Generator.my_images_generator_2d
import LIBS.ImgPreprocess.my_image_helper
from LIBS.DataPreprocess import my_data
from LIBS.ImgPreprocess.my_preprocess import do_preprocess

def do_predict_single(model1, img1, img_size=299, preproess=False, cuda_visible_devices=""):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if isinstance(model1, str):
        model1 = keras.models.load_model(model1, compile=False)

    if preproess:
        img1 = do_preprocess.my_preprocess(img1, 512)

    # /= 255. etc.
    img_tensor = LIBS.Generator.my_images_generator_2d.my_gen_img_tensor(img1,
                image_shape=(img_size, img_size, 3))

    prob1 = model1.predict_on_batch(img_tensor)
    prob1 = np.mean(prob1, axis=0)  # batch mean
    # pred1 = prob1.argmax(axis=-1)

    return prob1

#used by computing confusion matrix, etc.
def do_predict(dicts_models, files, devices=None,
               batch_size_test=64, argmax=False,
               use_multiprocessing=True, workers=4):

    if isinstance(files, str):  #csv file
        files, _ = my_data.get_images_labels(filename_csv_or_pd=files)
    assert len(files) > 0, 'No Data'

    prob_lists = []  #each element contain all probabilities  multiple batch, np.vstack
    preds_list = []

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        for dict_model in dicts_models:
            if ('model' not in dict_model) or (dict_model['model'] is None):
                print('prepare to load model:', dict_model['model_file'])
                model1 = keras.models.load_model(dict_model['model_file'], compile=False)
                print('load model:', dict_model['model_file'], ' complete')

                dict_model['model'] = model1
            else:
                model1 = dict_model['model']  # avoid loading models multiple times

            if 'input_shape' in dict_model:
                input_shape = dict_model['input_shape']
            elif len(model1.input_shape) == 4: #  [batch, height, width, channel]
                input_shape = model1.input_shape[1:]
            else:
                input_shape = (299, 299, 3)

            from LIBS.Generator.my_images_generator_2d import My_images_generator_2d_test
            list1 = [None] + list(input_shape)  # the last batch of an epoch may be smaller.
            shape_x = tuple(list1)  # (batch_size_train, 299, 299, 3)
            generator_test = My_images_generator_2d_test(files,
                        batch_size=batch_size_test, image_shape=input_shape)
            data_test = tf.data.Dataset.from_generator(generator=generator_test.gen,
                        output_types=(tf.float16), output_shapes=(shape_x))

            class My_callback(keras.callbacks.Callback):
                def on_predict_batch_end(self, batch, logs=None):
                    print('batch:', batch)
            probs = model1.predict(data_test,
                    callbacks=[My_callback()],
                    use_multiprocessing=use_multiprocessing, workers=workers)

            ''' #old version
            j = 0 # batch
            for x in generator_test.gen():
                probabilities = model1.predict_on_batch(x)
                if j == 0:    #'probs' not in locals().keys():
                    probs = probabilities
                else:
                    probs = np.vstack((probs, probabilities))
                j += 1
                print('batch:', j)
            '''

            prob_lists.append(probs)
            if argmax:
                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()
                preds_list.append(y_preds)

    sum_models_weights = 0
    for i, prob1 in enumerate(prob_lists):
        if 'model_weight' not in dicts_models[i]:
            model_weight = 1
        else:
            model_weight = dicts_models[i]['model_weight']

        if i == 0:
            prob_total = prob1 * model_weight
        else:
            prob_total += prob1 * model_weight

        sum_models_weights += model_weight

    prob_total /= sum_models_weights

    if argmax:
        y_pred_total = prob_total.argmax(axis=-1)
        return prob_total, y_pred_total, prob_lists, preds_list
    else:
        return prob_total, prob_lists


if __name__ ==  '__main__':

    model_file1 = '/home/ubuntu/dlp/deploy_models_2019/ocular_surface/3class/InceptionV3-021-train0.9985_val0.9996.hdf5'

    img_file1 = '/tmp2/fundus2.JPG'
    prob1 = do_predict_single(model_file1, img_file1, preproess=False, img_size=299)

    print('OK')


