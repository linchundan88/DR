
from LIBS.ImgPreprocess import my_preprocess_dir

dir_original = '/media/ubuntu/data1/糖网项目/图像质量/original/'
dir_preprocess = '/media/ubuntu/data1/糖网项目/图像质量/preprocess384'

my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess, image_size=384,
                                    convert_jpg=False, add_black_pixel_ratio=0.07)

print('OK')
