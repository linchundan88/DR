import cv2
import numpy as np
import collections

from tensorflow.keras.utils import to_categorical
import math

from LIBS.ImgPreprocess.my_image_norm import input_norm


def get_balance_class(files, labels, weights):
    y = np.array(labels)
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        p[y == i] = weight

    # random sampling
    random_sampling = np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                                       p=(np.array(p) / p.sum()))

    random_sampling_list = random_sampling.tolist()  # ndarray to list

    r_files = []
    r_labels = []
    for i in random_sampling_list:
        r_files.append(files[i])
        r_labels.append(labels[i])

    # 增加每一种label的样本数   比采样类权重直观
    list_num = [0 for x in range(36)]
    for label1 in r_labels:
        list_num[label1] = list_num[label1] + 1

    print(list_num)

    return r_files, r_labels

# 根据每个类的样本数，进行一个运算,生成动态采样的概率
def _get_class_weight(list_class_samples_num, file_weight_power, epoch=0):

    file_object = open(file_weight_power)
    try:
        line = file_object.readline()
        line = line.strip('\n')  # 删除换行符
        weight_power = float(line)

        print('set weight_power from file weight_power.txt')

    except Exception:
        print('read weight_power file error')
        print('set weight_power automatcally')

        dict_weight_power = collections.OrderedDict()
        dict_weight_power['0'] = 0.76
        dict_weight_power['1'] = 0.75
        dict_weight_power['2'] = 0.73
        dict_weight_power['3'] = 0.72
        dict_weight_power['4'] = 0.70
        dict_weight_power['5'] = 0.69
        dict_weight_power['6'] = 0.68
        dict_weight_power['7'] = 0.67
        dict_weight_power['8'] = 0.66
        dict_weight_power['9'] = 0.65
        dict_weight_power['10'] = 0.63
        dict_weight_power['12'] = 0.62

        for (k, v) in dict_weight_power.items():
            if epoch >= int(k):
                weight_power = v

    finally:
        file_object.close()

    '''
    获取每种类别的样本数目, 然后根据weight_power进行运算
    根据目录和dict_mapping 自动生成 class_samples_num
    '''

    list_weights = op_class_weight(list_class_samples_num, weight_power)
    weight_class = np.array(list_weights)

    print("epoch：%d, weight_power:  %f" % (epoch, weight_power))
    print('resampling ratio:', np.round(weight_class, 2))

    return weight_class

def op_class_weight(class_samples, weight_power=0.7):
    list_sample_weights = []

    max_class_samples = max(class_samples)

    for _, class_samples1 in enumerate(class_samples):
        class_samples1 = (max_class_samples ** weight_power) / (class_samples1 ** weight_power)
        list_sample_weights.append(class_samples1)

    return list_sample_weights

def smooth_labels(y, label_smoothing):
    # https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        label_smoothing: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''

    y1 = y.copy()
    y1 = y1.astype(np.float32)
    assert len(y1.shape) == 2 and 0 <= label_smoothing <= 1

    y1 *= 1 - label_smoothing
    y1 += label_smoothing / y1.shape[1]

    # y1[y1 == 1] = 1-label_smoothing + label_smoothing / num_classes
    # y1[y1 == 0] = label_smoothing / num_classes

    # np.multiply(y1, 1-label_smoothing, out=y1, casting="unsafe")

    return y1


def load_resize_images(image_files, image_shape=None, grayscale=False):
    list_image = []

    if isinstance(image_files, list):   # list of image files
        for image_file in image_files:
            image_file = image_file.strip()

            if grayscale:
                image1 = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image1 = cv2.imread(image_file)

            try:
                if (image_shape is not None) and (image1.shape[:2] != image_shape[:2]):
                        image1 = cv2.resize(image1, image_shape[:2])
            except:
                raise Exception("Image shape error:" + image_file)

            if image1 is None:
                raise Exception("Invalid image:" + image_file)

            if image1.ndim == 2:
                image1 = np.expand_dims(image1, axis=-1)

            list_image.append(image1)
    else:
        if isinstance(image_files, str):
            if grayscale:
                image1 = cv2.imread(image_files, cv2.IMREAD_GRAYSCALE)
            else:
                image1 = cv2.imread(image_files)
        else:
            if grayscale and image_files.ndim == 3:
                image1 = cv2.cvtColor(image_files, cv2.COLOR_BGR2GRAY)
            else:
                image1 = image_files

        try:
            if (image_shape is not None) and (image1.shape[:2] != image_shape[:2]):
                image1 = cv2.resize(image1, image_shape[:2])
        except:
            raise Exception("Invalid image:" + image_files)

        if image1 is None:
            raise Exception("Invalid image:" + image_files)

        if image1.ndim == 2:
            image1 = np.expand_dims(image1, axis=-1)

        list_image.append(image1)

    return list_image


#my_images_generator is used by multi-class, multi-label and regression
class My_images_generator_2d():
    def __init__(self, files, labels,
                 image_shape=(299, 299, 3),  num_output=1,
                 multi_labels=False, regression=False,  batch_size=32,
                 imgaug_seq=None,  do_normalize=True):

        self.files, self.labels = files, labels
        self.image_shape = image_shape

        self.batch_size = batch_size
        self.num_output = num_output

        self.multi_labels = multi_labels
        self.regression = regression

        self.imgaug_seq = imgaug_seq

        self.do_normalize = do_normalize

    def gen(self):
        n_samples = len(self.files)

        while True:
            for i in range(math.ceil(n_samples / self.batch_size)):
                sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
                files_batch = self.files[sl]
                labels_batch = self.labels[sl]

                x_train = load_resize_images(files_batch, self.image_shape)
                if self.imgaug_seq is not None:
                    x_train = self.imgaug_seq.augment_images(x_train)
                x_train = np.asarray(x_train, dtype=np.float16)
                if self.do_normalize:
                    x_train = input_norm(x_train)

                if not self.regression:
                    if not self.multi_labels:
                        y_train = np.asarray(labels_batch, dtype=np.uint8)
                        y_train = to_categorical(y_train, num_classes=self.num_output)
                    else:
                        y_train = []

                        for labels_str in labels_batch:
                            # print(labels_str)
                            labels = str(labels_str).split('_')
                            # 0_1_1 or [1,2]
                            list_labels = []
                            for _, label in enumerate(labels):
                                if label == '':
                                    continue

                                list_labels.append(int(label))

                            # print(list_labels)
                            # print('\n')
                            y_train.append(list_labels)

                            # convert '4_8_28' to [4,8,28]
                            '''
                            list_labels = []
                            for label1 in labels_str.split('_'):
                                if label1 == '':
                                    continue
                                list_labels.append(int(label1))

                            # convert [1,4]  to  [0,1,0,0,1,0,0...]
                            list_labels_convert = []
                            for j in range(self.num_output):
                                if j in list_labels:
                                    list_labels_convert.append(1)
                                else:
                                    list_labels_convert.append(0)
                                    
                            y_train.append(list_labels_convert)                                    
                           '''

                        y_train = np.asarray(y_train, dtype=np.uint8)
                else:  # regression
                    y_train = np.asarray(labels_batch, dtype=np.float16)

                yield x_train, y_train

# my_images_weight_generator is used by only multi-class
# before every epoch -resampling_dynamic and _get_balance_class
class My_images_weight_generator_2d():
    def __init__(self, files, labels, weight_class_start, weight_class_end, balance_ratio, label_smoothing=0,
                 image_shape=(299, 299), do_normalize=True,
                 batch_size=32, num_output=1, imgaug_seq=None):
        self.train_files = files
        self.train_labels = labels
        self.image_shape = image_shape

        self.do_normalize = do_normalize

        self.weight_class_start = weight_class_start
        self.weight_class_end = weight_class_end
        self.balance_ratio = balance_ratio

        self.batch_size = batch_size
        self.num_output = num_output

        self.imgaug_seq = imgaug_seq

        self.label_smoothing = label_smoothing

    def resampling_dynamic(self, weight_class_start, weight_class_end, balance_ratio, epoch):
        alpha = balance_ratio ** epoch
        class_weights = weight_class_start * alpha + weight_class_end * (1 - alpha)
        class_weights = np.around(class_weights, decimals=2)  # 保留两位小数

        print('resampling ratio:', class_weights)
        return class_weights

    def gen(self):
        n_samples = len(self.train_files)

        current_batch_num = 0
        current_epoch = 0

        while True:
            weights = self.resampling_dynamic(weight_class_start=self.weight_class_start, weight_class_end=self.weight_class_end,
                     balance_ratio=self.balance_ratio, epoch=current_epoch)

            train_files_balanced, train_labels_balanced = get_balance_class(
                    self.train_files, self.train_labels, weights=weights)

            # print('\nlabels:', train_labels_balanced)

            for i in range(math.ceil(n_samples / self.batch_size)):
                sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
                files_batch, labels_batch = train_files_balanced[sl], train_labels_balanced[sl]

                x_train = load_resize_images(files_batch, self.image_shape)
                if self.imgaug_seq is not None:
                    x_train = self.imgaug_seq.augment_images(x_train)

                # imgs_aug返回的x_train 的是list，每个元素(299,299,3) float32
                x_train = np.asarray(x_train, dtype=np.float16)
                if self.do_normalize:
                    x_train = input_norm(x_train)

                y_train = np.asarray(labels_batch, dtype=np.uint8)  # 64*1
                y_train = to_categorical(y_train, num_classes=self.num_output)

                if self.label_smoothing > 0:
                    y_train = smooth_labels(y_train, self.label_smoothing)

                current_batch_num = current_batch_num + 1


                yield x_train, y_train

            current_epoch = current_epoch + 1


#different from My_images_generator_2d, do not use labels, do not iterate many times
class My_images_generator_2d_test():
    def __init__(self, files,
                 image_shape=(299, 299, 3), batch_size=32,
                 imgaug_seq=None, do_normalize=True):

        self.files = files
        self.image_shape = image_shape
        self.batch_size = batch_size

        self.imgaug_seq = imgaug_seq
        self.do_normalize = do_normalize

    def gen(self):
        n_samples = len(self.files)

        for i in range(math.ceil(n_samples / self.batch_size)):
            sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
            files_batch = self.files[sl]

            x_train = load_resize_images(files_batch, self.image_shape)
            if self.imgaug_seq is not None:
                x_train = self.imgaug_seq.augment_images(x_train)
            x_train = np.asarray(x_train, dtype=np.float16)
            if self.do_normalize:
                x_train = input_norm(x_train)

            yield x_train

class My_images_generator_2d_test_imgaug():
    def __init__(self, files,
                 image_shape=(299, 299, 3), batch_size=32,
                 dx=10, dy=10,
                 do_flip=True,
                 do_normalize=True):

        self.files = files
        self.image_shape = image_shape
        self.batch_size = batch_size

        self.dx = dx
        self.dy = dy
        self.do_flip = do_flip
        self.do_normalize = do_normalize

    def gen(self):
        n_samples = len(self.files)

        for i in range(math.ceil(n_samples / self.batch_size)):
            sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
            files_batch = self.files[sl]
            from LIBS.ImgPreprocess.my_test_time_img_aug import load_resize_images_imgaug
            x_train = load_resize_images_imgaug(files_batch, self.image_shape,
                    dx=self.dx, dy=self.dy, do_flip=self.do_flip)
            x_train = np.asarray(x_train, dtype=np.float16)
            if self.do_normalize:
                x_train = input_norm(x_train)

            yield x_train

#  a simple format of My_images_generator_2d_test
def my_gen_img_tensor(files, image_shape=(299, 299, 3)):
    images = load_resize_images(files, image_shape)
    x_test = np.asarray(images, dtype=np.float16)
    x_test = input_norm(x_test)

    return x_test



if __name__ == '__main__':
   pass


