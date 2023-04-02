from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# load the pre-trained model
model = tf.keras.models.load_model('model.hdf5', compile=False)

# define the mapping of class indices to labels
class_labels = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

# helper function to preprocess audio data
def preprocess_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs_features.T,axis=0)
    
    return mfccs
    

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    audio_file = request.files.get('audio-file')

    # preprocess the audio data
    audio_data = preprocess_audio(audio_file)

    # make a predictioncdc
    audio_data = audio_data.reshape(1, -1)
    predictions = model.predict(audio_data)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return render_template('index.html', prediction_text='Predicted Class :{}'.format(predicted_class_label))

if __name__ == '__main__':
    app.run(debug=True)





