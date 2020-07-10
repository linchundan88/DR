# test time image augmentation,

import cv2
import numpy as np

def img_move(img1, dx, dy):
    if isinstance(img1, str):
        img2 = cv2.imread(img1)
    else:
        img2 = img1

    rows, cols = img2.shape[0:2]

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(img2, M, (cols, rows))

    return dst

def img_flip(img1, flip_direction=1): #horizonal:1, vertical:0
    if isinstance(img1, str):
        img2 = cv2.imread(img1)
    else:
        img2 = img1

    img3 = cv2.flip(img2, flip_direction)

    return img3

def load_resize_images_imgaug(image_files, image_shape=None, grayscale=False,
        dx=10, dy=10, do_flip=True):
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

            if not do_flip:
                image2 = img_move(-dx, -dy)
                image3 = img_move(-dx, -dy)
                list_image.append(image2)
                list_image.append(image3)
            if do_flip:
                image2 = img_move(-dx, -dy)
                image2 =img_flip(image2,flip_direction=0)
                image3 = img_move(-dx, -dy)
                image2 = img_flip(image2, flip_direction=1)
                list_image.append(image2)
                list_image.append(image3)

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

        if not do_flip:
            image2 = img_move(-dx, -dy)
            image3 = img_move(-dx, -dy)
            list_image.append(image2)
            list_image.append(image3)
        if do_flip:
            image2 = img_move(-dx, -dy)
            image2 = img_flip(image2, flip_direction=0)
            image3 = img_move(-dx, -dy)
            image2 = img_flip(image2, flip_direction=1)
            list_image.append(image2)
            list_image.append(image3)

    return list_image



if __name__ == '__main__':
    img_file = '/tmp1/ouzel1.jpg'

    img1 = cv2.imread(img_file)

    image2 = cv2.flip(img1,1)  #horizonal
    # image=cv2.flip(image,0)  #vertical

    cv2.imshow('aaa', image2)

    # cv2.imshow( 'dst', img_move(img_file, 50,3))
    cv2.waitKey(0)


