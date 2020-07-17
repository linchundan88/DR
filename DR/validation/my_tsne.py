# can not do model ensemble, because of different models have different output dimensions.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from LIBS.DataPreprocess import my_data


nb_classes = 2
train_type = 'DR_english'
data_version = 'V1'
filename_csv_test = os.path.join(os.path.abspath('..'),
        'datafiles', train_type, 'split_patid_test_{}.csv'.format(data_version))

df = pd.read_csv(filename_csv_test)
files, labels = my_data.get_images_labels(filename_csv_or_pd=df)

model_file = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_5_19/InceptionResnetV2-008-0.985.hdf5'
input_shape = (299, 299, 3)
save_tsne_image = "/tmp5/t_sne_2020_5_21/Stage_tsne.png"


from LIBS.Neural_Networks.TSNE.my_tsne_helper import compute_features, gen_tse_features, draw_tsne
features = compute_features(model_file, files, input_shape=input_shape)
X_tsne = gen_tse_features(features)

# save_npy_file = "/tmp5/probs_test1.npy"
# import numpy as np
# np.save(save_npy_file, X_tsne)
# X_tsne = np.load(save_npy_file)

draw_tsne(X_tsne, labels, nb_classes=nb_classes, save_tsne_image=save_tsne_image,
          labels_text=['Referable DR', 'Non-Referable'], colors=['g', 'r'])

print('OK')