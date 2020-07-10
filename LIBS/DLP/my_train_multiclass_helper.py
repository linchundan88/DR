
import os
import pandas as pd
import numpy as np
from LIBS.DataPreprocess import my_data
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from LIBS.Neural_Networks.Utils import my_transfer_learning
from LIBS.Generator.my_images_generator_2d import My_images_generator_2d, My_images_weight_generator_2d
import math
from LIBS.Neural_Networks.optimization.lookahead import Lookahead
from LIBS.Neural_Networks.optimization.adabound import AdaBound
import json

def train_task_one_step(model_file, filename_csv_train, filename_csv_valid, filename_csv_test=None,
                        num_classes=None,
                        pre_define_model=None,
                        add_top=False, change_top=False,
                        input_shape=(299, 299, 3), imgaug_train_seq=None,
                        optimizer="adam", lookahead=False,
                        epoch=None, dict_lr=None,
                        batch_size_train=32, batch_size_valid=64,
                        label_smoothing=0, class_weight=None,
                        weight_class_start=None, weight_class_end=None, balance_ratio=None,
                        sensitivity=None, specificity=None,
                        devices=None, verbose=1,
                        model_save_dir='/tmp', model_name='model1',
                        config_file_realtime='config_file_realtime.json',
                        use_multiprocessing=True, workers=5,
                        plt_history_image_file=None):

    #region read csv, split train validation set
    if num_classes is None:
        df = pd.read_csv(filename_csv_train)
        num_classes = df['labels'].nunique(dropna=True)

    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(filename_csv_valid)
    #endregion

    #region data sequence
    if weight_class_start is not None:
        generator_train = My_images_weight_generator_2d(train_files, train_labels, num_output=num_classes,
                                                        batch_size=batch_size_train,
                                                        weight_class_start=weight_class_start,
                                                        weight_class_end=weight_class_end,
                                                        balance_ratio=balance_ratio,
                                                        imgaug_seq=imgaug_train_seq,
                                                        label_smoothing=label_smoothing,
                                                        image_shape=input_shape,
                                                        )
    else:
        generator_train = My_images_generator_2d(train_files, train_labels, num_output=num_classes,
                                                 batch_size=batch_size_train, image_shape=input_shape,
                                                 imgaug_seq=imgaug_train_seq, label_smoothing=label_smoothing)

    list1 = [None] + list(input_shape)  # the last batch of an epoch may be smaller.
    shape_x = tuple(list1)     #(batch_size_train, 299, 299, 3)
    shape_y = (None, num_classes)  #(batch_size, num_classes)
    data_train = tf.data.Dataset.from_generator(generator=generator_train.gen,
                output_types=(tf.float32, tf.float32),
                output_shapes=(shape_x, shape_y))

    generator_valid = My_images_generator_2d(valid_files, valid_labels, num_output=num_classes,
                                             batch_size=batch_size_valid, image_shape=input_shape)
    data_valid = tf.data.Dataset.from_generator(generator=generator_valid.gen,
                output_types=(tf.float32, tf.float32),
                output_shapes=(shape_x, shape_y))

    if filename_csv_test is not None:
        test_files, test_labels = my_data.get_images_labels(filename_csv_test)

    #endregion

    #region callbacks

    os.makedirs(model_save_dir, exist_ok=True)
    save_filepath_finetuning = os.path.join(model_save_dir, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")
    checkpointer_finetuning = keras.callbacks.ModelCheckpoint(save_filepath_finetuning,
              verbose=1, save_weights_only=False, save_best_only=False)

    class My_callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            try:
                with open(config_file_realtime, 'r') as json_file:
                    data = json.load(json_file)

                    if data['epoch_compute_cf_train'] == 1:
                        compute_cf_train = True
                    else:
                        compute_cf_train = False

                    if data['epoch_compute_cf_valid'] == 1:
                        compute_cf_valid = True
                    else:
                        compute_cf_valid = False

                    if data['epoch_compute_cf_test'] == 1:
                        compute_cf_test = True
                    else:
                        compute_cf_test = False
            except:
                print('read realtime helper file error!')
                compute_cf_train = True
                compute_cf_valid = True
                compute_cf_test = True

            if compute_cf_train:
                print('calculate confusion matrix of training dataset...')
                generator_cf_train = My_images_generator_2d(train_files, train_labels, num_output=num_classes,
                                                            batch_size=batch_size_train, image_shape=input_shape)
                i = 0
                for x_train, y_train in generator_cf_train.gen():
                    probabilities = self.model.predict(x_train)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(train_files) / batch_size_train):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, num_classes)]
                confusion_matrix_train = sk_confusion_matrix(train_labels, y_preds, labels=labels)

                print(confusion_matrix_train)

            if compute_cf_valid:
                print('calculate confusion matrix of validation dataset...')
                generator_cf_valid = My_images_generator_2d(valid_files, valid_labels, num_output=num_classes,
                                                            batch_size=batch_size_valid, image_shape=input_shape)
                i = 0
                for x_valid, y_valid in generator_cf_valid.gen():
                    probabilities = self.model.predict(x_valid)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(valid_files) / batch_size_valid):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, num_classes)]
                confusion_matrix_valid = sk_confusion_matrix(valid_labels, y_preds, labels=labels)
                print(confusion_matrix_valid)

            if compute_cf_test:
                print('calculate confusion matrix of test dataset...')
                generator_cf_test = My_images_generator_2d(test_files, test_labels, num_output=num_classes,
                                                           batch_size=batch_size_valid, image_shape=input_shape)
                i = 0
                for x_test, y_test in generator_cf_test.gen():
                    probabilities = self.model.predict(x_test)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(test_files) / batch_size_valid):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, num_classes)]
                confusion_matrix_test = sk_confusion_matrix(test_labels, y_preds, labels=labels)
                print(confusion_matrix_test)

    my_callback = My_callback()

    if epoch is None:
        if len(df) > 10000:
            epoch = 20
        elif len(df) > 5000:
            epoch = 25
        elif len(df) > 2000:
            epoch = 30
        else:
            epoch = 40

    def scheduler_finetuning(epoch):
        if optimizer == 'adabound':
            return K.get_value(model1.optimizer.lr)
        try:
            with open(config_file_realtime, 'r') as json_file:
                data = json.load(json_file)
                if data['lr_rate'] > 0:
                    lr_rate = data['lr_rate']

                    print("epoch：%d, current learn rate:  %f by realtime helper file" % (epoch, lr_rate))
                    K.set_value(model1.optimizer.lr, lr_rate)
                    return K.get_value(model1.optimizer.lr)
        except Exception:
            print('read realtime helper file error!')

        if dict_lr is not None:
            for (k, v) in dict_lr.items():
                if epoch >= int(k):
                    lr_rate = v

            print("epoch：%d, set  learn rate:  %f according to pre-defined policy." % (epoch, lr_rate))
            K.set_value(model1.optimizer.lr, lr_rate)

        return K.get_value(model1.optimizer.lr)

    change_lr_finetuning = keras.callbacks.LearningRateScheduler(scheduler_finetuning)

    #endregion

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    print("Number of devices: {} used".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        if pre_define_model is None:
            print('loading model...')
            model1 = keras.models.load_model(model_file, compile=False)
            print('loading model complete!')

            if add_top:
                model1 = my_transfer_learning.add_top(model1, num_output=num_classes, activation_function='softmax')
            if change_top:
                model1 = my_transfer_learning.convert_model_transfer(model1, clsss_num=num_classes,
                                                                     change_top=change_top, activation_function='SoftMax',
                                                                     freeze_feature_extractor=False)
            model1 = my_transfer_learning.convert_trainable_all(model1)
        else:
            print('creating model...')
            if pre_define_model.lower() == 'inceptionv3':
                model1 = keras.applications.InceptionV3(include_top=False)
            elif pre_define_model.lower() == 'xception':
                model1 = keras.applications.Xception(include_top=False)
            elif pre_define_model.lower() == 'inceptionresnetv2':
                model1 = keras.applications.InceptionResNetV2(include_top=False)
            else:
                raise Exception('predefine model error!')

            model1 = my_transfer_learning.add_top(model1, num_output=num_classes, activation_function='softmax')
            print('creating model complete!')

        assert optimizer in ['adam', 'SGD', 'adabound'], 'optimizer type  error'
        if optimizer == 'adam':
            op_finetuning = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if optimizer == 'SGD':
            op_finetuning = keras.optimizers.sgd(lr=1e-3, momentum=0.9, nesterov=True)
        if optimizer == 'adabound':
            op_finetuning = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, amsbound=False)

        if sensitivity is not None and specificity is not None:
            sensivity = keras.metrics.SensitivityAtSpecificity(specificity=sensitivity)
            specificity = keras.metrics.SpecificityAtSensitivity(sensitivity=specificity)
            my_metric = ['acc', sensivity, specificity]
        else:
            my_metric = ['acc']

        # loss1 = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing)
        loss1 = 'categorical_crossentropy'
        model1.compile(loss=loss1,
                       optimizer=op_finetuning, metrics=my_metric)

        if lookahead:
            lookahead = Lookahead(k=5, alpha=0.5)
            lookahead.inject(model1)

        history = model1.fit(
            data_train,
            steps_per_epoch=math.ceil(len(train_files) / batch_size_train),
            epochs=epoch,
            verbose=verbose,
            validation_data=data_valid,
            validation_steps=math.ceil(len(valid_files) / batch_size_valid),
            callbacks=[checkpointer_finetuning, change_lr_finetuning, my_callback],
            class_weight=class_weight,
            use_multiprocessing=use_multiprocessing, workers=workers
        )

        if plt_history_image_file is not None:
            import matplotlib.pyplot as plt
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            plt.savefig(plt_history_image_file)
            plt.close()

    K.clear_session()

