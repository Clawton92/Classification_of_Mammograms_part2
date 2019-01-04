from flask import Flask, render_template, request, jsonify
import pickle
from build_model import TextClassifier
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
import pdb
from werkzeug.utils import secure_filename
import os
import numpy as np
from final_model_for_flask import MassPredictor
import pdb
import imageio

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('form/index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('form/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_loc = '/Users/christopherlawton/galvanize/module_3/cap_3_dir/flaskr/static/'+filename
    file.save(file_loc)

    img = imageio.imread(file_loc)
    model = MassPredictor()
    model.predict(img)
    figure_path = '/Users/christopherlawton/galvanize/module_3/cap_3_dir/flaskr/static/'+'dist_'+filename
    model.plot_dist(figure_path)
    pred_class = model.predicted_class
    prob_benign = model.prob_benign
    prob_malignant = model.prob_malignant
    K.clear_session()
    return render_template('form/predict.html', predicted='{}'.format(pred_class),\
                            proba_benign='{}%'.format(prob_benign * 100), proba_malignant='{}%'.format(prob_malignant * 100), \
                            upload_img=filename, upload_plot='dist_'+filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
