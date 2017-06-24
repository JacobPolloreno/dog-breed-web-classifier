from typing import Tuple
from keras.models import model_from_json
import tensorflow as tf


def init_model(model: str, weights: str) -> Tuple:
    """Initialize the keras model

    :param model, str: path to model.json
    :param weights, str: path to weights for model
    :return keras model and graph
    """
    with open(model, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load woeights into new model
    loaded_model.load_weights(weights)

    # compile and evaluate loaded model
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    graph = tf.get_default_graph()

    return loaded_model, graph
