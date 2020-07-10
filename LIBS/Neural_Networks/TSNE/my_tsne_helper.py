import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalAveragePooling3D, AveragePooling3D
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# colors: b--blue, c--cyan, g--green, k--black, r--red, w--white, y--yellow, m--magenta
def draw_tsne(X_tsne, labels, nb_classes, labels_text, colors=['g', 'r', 'b'], save_tsne_image=None):

    y = np.array(labels)
    colors_map = y

    plt.figure(figsize=(10, 10))
    for cl in range(nb_classes):
        indices = np.where(colors_map == cl)
        # plt.ylabel('aaaaaaaaa')
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=colors[cl], label=labels_text[cl])
    plt.legend()

    os.makedirs(os.path.dirname(save_tsne_image), exist_ok=True)
    plt.savefig(save_tsne_image)
    # plt.show()


def compute_features(model_file, files, input_shape, batch_size=32, gen_tsne_features=True):

    model1 = keras.models.load_model(model_file, compile=False)

    for i in range(len(model1.layers) - 1, -1, -1):
        if isinstance(model1.layers[i], GlobalAveragePooling2D) or \
                isinstance(model1.layers[i], GlobalAveragePooling3D):
            layer_num_GAP = i
            break
        if isinstance(model1.layers[i], AveragePooling2D) or \
                isinstance(model1.layers[i], AveragePooling3D):
            layer_num_GAP = i
            is_avgpool = True
            break

    # Inception-Resnet V2 1536
    output_model = Model(inputs=model1.input,
                         outputs=model1.layers[layer_num_GAP].output)
    # output_model.summary()

    batch_no = 0
    from LIBS.Generator.my_images_generator_2d import My_images_generator_2d_test
    generator_test = My_images_generator_2d_test(files,
                    batch_size=batch_size, image_shape=input_shape)
    for x in generator_test.gen():
        y = output_model.predict_on_batch(x)

        print('batch_no:', batch_no)
        batch_no += 1

        if 'features' not in locals().keys():
            features = y
        else:
            features = np.vstack((features, y))

    return features


def gen_tse_features(features):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(features)

    return X_tsne