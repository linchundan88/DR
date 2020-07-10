'''do_predict_dir()调用多个模型，预测目录中的每一个文件，返回总该率和总预测类别
虽然预测目录不需要csv文件，但是为了和文件处理一致，还是使用同一个csv文件
'''

import os

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from LIBS.DataPreprocess import my_data
import pandas as pd
import shutil
import heapq


'''
  top 1 class/  top1 prob 
  for example: /0/prob35#aaaa.jpg
'''
def op_files_multiclass(all_files, prob_total, dir_preprocess,
                    dir_original='', dir_dest='/tmp', keep_subdir=False):

    if isinstance(all_files, str):  #csv file
        all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=all_files)

    for i in range(len(all_files)):
        #region  get top class, and prob of top class
        prob_list = prob_total[i].tolist()

        top_class_n = heapq.nlargest(5, range(len(prob_total[i])), prob_total[i].take)
        #top_n[0]  Softmax argmax  class no
        prob_top_class0 = round(prob_list[top_class_n[0]] * 100, 0)
        prob_top_class0 = int(prob_top_class0)
        #endregion

        img_file_source = all_files[i]

        filename = os.path.basename(img_file_source)

        if '#' in filename:    #obtain filename after #,
            filename = filename.split('#')[-1]

        filename_dest = 'prob' + str(prob_top_class0) + '#' + filename

        if keep_subdir:
            basedir = os.path.dirname(img_file_source)
            if not dir_preprocess.endswith('/'):
                dir_preprocess += '/'
            if not basedir.endswith('/'):
                basedir += '/'

            img_file_dest = os.path.join(dir_dest, basedir.replace(dir_preprocess, ''),
                                         str(top_class_n[0]), filename_dest)
        else:
            img_file_dest = os.path.join(dir_dest, str(top_class_n[0]), filename_dest)

        os.makedirs(os.path.dirname(img_file_dest), exist_ok=True)

        # copy original files instead of preprocessed images
        if dir_original != '':
            if dir_original.endswith('/'):
                dir_original = dir_original[:-1]
            if dir_preprocess.endswith('/'):
                dir_preprocess = dir_preprocess[:-1]

            img_file_source = img_file_source.replace(dir_preprocess, dir_original)

        if not os.path.exists(img_file_source):
            raise RuntimeError(img_file_source + ' not found!')

        shutil.copy(img_file_source, img_file_dest)

        print(img_file_dest)



def compute_confusion_matrix(probs_list, dir_dest,
                             all_files, all_labels,
                             dir_preprocess='', dir_original='', ):
    cf_list = []
    not_match_list = []

    # every model's confusion matrix and not match files
    for probs in probs_list:

        y_preds = probs.argmax(axis=-1)
        y_preds = y_preds.tolist()

        NUM_CLASSES = probs[0].shape[0]  #len(probs[0])

        labels = [x for x in range(0, NUM_CLASSES)]
        cf1 = sk_confusion_matrix(all_labels, y_preds, labels=labels)
        cf_list.append(cf1)

        not_match1 = []
        for i in range(len(all_files)):
            if all_labels[i] != y_preds[i]:
                dict_predict = {'filename': all_files[i], 'pred_level': y_preds[i], 'label_level': all_labels[i]}
                not_match1.append(dict_predict)
        not_match_list.append(not_match1)


    # region total confusion matrix and not match files
    for i, probs in enumerate(probs_list):
        if i == 0:
            prob_total = probs
        else:
            prob_total += probs

    prob_total /= len(probs_list)
    y_pred_total = prob_total.argmax(axis=-1)
    cf_total = sk_confusion_matrix(all_labels, y_pred_total, labels=labels)

    not_match_total = []
    for i in range(len(all_files)):
        if all_labels[i] != y_pred_total[i]:
            prob_max = np.max(prob_total[i]) * 100  # find maximum probability value
            dict_predict = {'filename': all_files[i], 'pred_level': y_pred_total[i],
                            'prob_max': prob_max,
                            'label_level': all_labels[i]}
            not_match_total.append(dict_predict)

    # endregion

    #  export confusion files
    if dir_dest != '':
        for dict1 in not_match_total:

            img_file_source = str(dict1['filename']).strip()

            # some files are deleted for some reasons.
            if not os.path.exists(img_file_source):
                raise RuntimeError(img_file_source + ' not found!')

            _, filename = os.path.split(img_file_source)
            file_dest = os.path.join(dir_dest, str(dict1['label_level']) + '_' + str(dict1['pred_level'])
                                     , str(int(dict1['prob_max'])) + '__' + filename)

            #copy original files instead of preprocessed files
            if dir_preprocess != '' and dir_original != '':
                if not dir_preprocess.endswith('/'):
                    dir_preprocess = dir_preprocess + '/'
                if not dir_original.endswith('/'):
                    dir_original = dir_original + '/'

                img_file_source = img_file_source.replace(dir_preprocess, dir_original)

            if not os.path.exists(img_file_source):
                raise RuntimeError(img_file_source + ' not found!')

            os.makedirs(os.path.dirname(file_dest), exist_ok=True)
            shutil.copyfile(img_file_source, file_dest)
            print('copy file:', file_dest)


    return cf_list, not_match_list, cf_total, not_match_total