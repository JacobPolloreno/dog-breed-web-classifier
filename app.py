import base64
import logging
import io
import os

from flask import Flask, render_template, request
from load import init_model
from PIL import Image
from util import decode_prob

logger = logging.getLogger("dog_breed_classifier")
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

# Initialize
MODEL_DIR = os.path.abspath("./models")

RESNET_CONFIG = {'arch':
                 os.path.join(MODEL_DIR,
                              'model.Resnet50.json'),
                 'weights':
                 os.path.join(MODEL_DIR,
                              'weights.Resnet50.hdf5')}

INCEPTION_CONFIG = {'arch':
                    os.path.join(MODEL_DIR,
                                 'model.inceptionv3.json'),
                    'weights':
                    os.path.join(MODEL_DIR,
                                 'weights.inceptionv3.h5')}

MODELS = {'resnet': RESNET_CONFIG,
          'inception': INCEPTION_CONFIG}


@app.route('/index')
@app.route('/')
def index():
    return render_template('settings.html')


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Select Model Architecture and Initialize
    """
    global model, graph, preprocess

    # grab model selected
    model_name = request.form['model']
    config = MODELS[model_name]

    # init the model with pre-trained architecture and weights
    model, graph = init_model(config['arch'], config['weights'])

    # use the proper preprocessing method
    if model_name == 'inception':
        from util import preprocess_inception
        preprocess = preprocess_inception
    else:
        from util import preprocess_resnet
        preprocess = preprocess_resnet

    return render_template('select_files.html', model_name=model_name)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """File selection and display results
    """

    if request.method == 'POST' and 'file[]' in request.files:
        inputs = []
        files = request.files.getlist('file[]')
        for file_obj in files:
            # Check if no files uploaded
            if file_obj.filename == '':
                if len(files) == 1:
                    return render_template('select_files.html')
                continue
            entry = {}
            entry.update({'filename': file_obj.filename})
            try:
                img_bytes = io.BytesIO(file_obj.stream.getvalue())
                entry.update({'data':
                              Image.open(
                                  img_bytes
                              )})
            except AttributeError:
                img_bytes = io.BytesIO(file_obj.stream.read())
                entry.update({'data':
                              Image.open(
                                  img_bytes
                              )})
            # keep image in base64 for later use
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
            entry.update({'img': img_b64})

            inputs.append(entry)

        outputs = []

        with graph.as_default():
            for input_ in inputs:
                # convert to 4D tensor to feed into our model
                x = preprocess(input_['data'])
                # perform prediction
                out = model.predict(x)
                outputs.append(out)

        # decode output prob
        outputs = decode_prob(outputs)

        results = []

        for input_, probs in zip(inputs, outputs):
            results.append({'filename': input_['filename'],
                            'image': input_['img'],
                            'predict_probs': probs})

        return render_template('results.html', results=results)

    # if no files uploaded
    return render_template('select_files.html')


if __name__ == '__main__':
    app.run(debug=True)
