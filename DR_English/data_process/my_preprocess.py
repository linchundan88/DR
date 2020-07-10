
from LIBS.ImgPreprocess import my_preprocess_dir

dir_original = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/original/'
dir_preprocess = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384'

my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess, image_size=384,
                                    convert_jpg=False, add_black_pixel_ratio=0.07)


print('OK')

