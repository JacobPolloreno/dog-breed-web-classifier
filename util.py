import numpy as np
import os

from keras.preprocessing import image
from operator import itemgetter

IMAGE_DIM = (224, 224)
LABELS_PATH = os.path.abspath('labels.txt')

global labels

with open(LABELS_PATH) as f:
    labels = f.readlines()

labels = np.array([label.strip() for label in labels])


def img_to_tensor(img):
    """Loads a PIL image and outputs a 4d tensor

    :param img, PIL Image object
    :return 4D numpy array/tensor
    """
    if img.size != IMAGE_DIM:
        img = img.resize(IMAGE_DIM)
    # convert to 3d array
    x = image.img_to_array(img)
    # conver to 4d
    return np.expand_dims(x, axis=0)


def preprocess_inception(img):
    """Prepare image for Inception model

    :param img, PIL Image object
    :return preprocssed input for inception model
    """
    from keras.applications.inception_v3 import preprocess_input
    img = preprocess_input(img_to_tensor(img))
    return img


def extract_bottleneck_features_resnet(tensor):
    from keras.applications.resnet50 import ResNet50
    return ResNet50(weights='imagenet', include_top=False).predict(tensor)


def preprocess_resnet(img):
    """Prepare image for Resnet50 model
    """
    from keras.applications.resnet50 import preprocess_input
    img = preprocess_input(img_to_tensor(img))
    return extract_bottleneck_features_resnet(img)


def decode_prob(output_arr, top_probs=5):
    """Label class probabilities with class names

    :param output_arr, list: class probabilities
    :param top_probs, int: number of class probabilities to return out of 133

    :return list[dict]:
    """
    results = []
    for row in output_arr:
        entries = []
        for name, prob in zip(labels, row[0]):
            entries.append({'name': name,
                            'prob': prob})

        entries = sorted(entries,
                         key=itemgetter('prob'),
                         reverse=True)[:top_probs]

        for entry in entries:
            entry['prob'] = '{:.5f}'.format(entry['prob'])

        results.append(entries)
    return results
