
import numpy as np
import math
import cv2
import os

# detect vessel G channel


def get_green_channel(img1, img_file_dest=None):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    img1 = img1[:, :, 1] # BGR
    img1 = np.expand_dims(img1, axis=-1)

    img_zero = np.zeros(img1.shape)
    img2 = np.concatenate((img_zero, img1, img_zero), axis=-1)

    if img_file_dest is not None:
        cv2.imwrite(img_file_dest, img2)
    else:
        return img2


def resize_images_dir(source_dir, dest_dir, convert_image_to_square=False, image_size=299):
    if not source_dir.endswith('/'):
        source_dir += '/'
    if not dest_dir.endswith('/'):
        dest_dir += '/'

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            img1 = cv2.imread(image_file_source)
            if img1 is None:
                print('error file:', image_file_source)

            if convert_image_to_square:
                img1 = image_to_square(img1)

            img1 = cv2.resize(img1, (image_size, image_size))

            image_file_dest = image_file_source.replace(source_dir, dest_dir)
            os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
            cv2.imwrite(image_file_dest, img1)
            print(image_file_source)

# 1.square, 2.resize
def image_to_square(image1, image_size=None, grayscale=False):
    if isinstance(image1, str):
        image1 = cv2.imread(image1)

    height, width = image1.shape[:-1]

    if width > height:
        #original size can be odd or even number,
        padding_top = math.floor((width - height) / 2)
        padding_bottom = math.ceil((width - height) / 2)

        image_padding_top = np.zeros((padding_top, width, 3), dtype=np.uint8)
        image_padding_bottom = np.zeros((padding_bottom, width, 3), dtype=np.uint8)

        image1 = np.concatenate((image_padding_top,image1,image_padding_bottom), axis=0)
    elif width < height:
        padding_left = math.floor((height - width) / 2)
        padding_right = math.ceil((height - width) / 2)

        image_padding_left = np.zeros((height, padding_left, 3), dtype=np.uint8)
        image_padding_right = np.zeros((height, padding_right, 3), dtype=np.uint8)

        image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)


    if image_size is not None:
        height, width = image1.shape[:-1] #image1 is square now

        if height > image_size:
            image1 = cv2.resize(image1, (image_size, image_size))
        elif height < image_size:
            if image_size > width:
                padding_left = math.floor((image_size - width) / 2)
                padding_right = math.ceil((image_size - width) / 2)

                image_padding_left = np.zeros((height, padding_left, 3), dtype=np.uint8)
                image_padding_right = np.zeros((height, padding_right, 3), dtype=np.uint8)

                image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)
                height, width = image1.shape[:-1]

            if image_size > height:
                padding_top = math.floor((image_size - height) / 2)
                padding_bottom = math.ceil((image_size - height) / 2)

                image_padding_top = np.zeros((padding_top, width, 3), dtype=np.uint8)
                image_padding_bottom = np.zeros((padding_bottom, width, 3), dtype=np.uint8)

                image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)
                height, width = image1.shape[:-1]

    if grayscale:
        #cv2.cvtColor only support unsigned int (8U, 16U) or 32 bit float (32F).
        # image_output = np.uint8(image_output)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    return image1

def image_border_padding(image1,
                         padding_top, padding_bottom, padding_left, padding_right):

    if image1.ndim == 2:
        image1 = np.expand_dims(image1, axis=-1)
    (height, width) = image1.shape[:-1]

    image_padding_top = np.zeros((padding_top, width, 3), dtype=np.uint8)
    image_padding_bottom = np.zeros((padding_bottom, width, 3), dtype=np.uint8)

    image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)

    (height, width) = image1.shape[:-1]

    image_padding_left = np.zeros((height, padding_left, 3), dtype=np.uint8)
    image_padding_right = np.zeros((height, padding_right, 3), dtype=np.uint8)

    image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)

    return image1


def crop_image(img1, bottom, top, left, right):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    img2 = img1[bottom:top, left:right, :]

    return img2


if __name__ ==  '__main__':
    img1 = np.ones((50, 100, 3))
    img1 = img1 * 255
    cv2.imwrite('/tmp1/111.jpg', img1)

    img2 = image_to_square(img1, imgsize=150)
    cv2.imwrite('/tmp1/122.jpg', img2)
    exit(0)


