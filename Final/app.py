from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import os
import numpy as np
from tensorflow import keras
import pandas as pd
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import librosa

# Load the emotion recognition model
model = keras.models.load_model('E:\\Samsung\\Web\\models\\model.h5')

# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def transcribe_audio(mp3_file):
    # Convert audio file to WAV format using FFmpeg
    wav_file = 'temp_audio.wav'
    ffmpeg_path = 'ffmpeg'  # Update with the correct path to ffmpeg if necessary
    command = [ffmpeg_path, '-i', mp3_file, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, error = process.communicate()

    if process.returncode != 0:
        print(f"Audio conversion failed: {error.decode('utf-8').strip()}")
        return None

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)

    # Perform speech recognition
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from speech recognition service; {0}".format(e))
    finally:
        # Clean up the temporary WAV file
        os.remove(wav_file)

    return "Function passed"

def predict_emotion(text):
    # Load the CSV data
    df = pd.read_csv('E:\\Samsung\\Web\\emotions.csv')

    # Replace missing values with an empty string
    df['Text'].fillna('', inplace=True)

    # Extract input features (X) and labels (y) from the CSV
    X = df['Text'].values
    y = df['Emotion'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a model on the training data
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # Make predictions on new text inputs
    text_vectorized = vectorizer.transform([text])
    predicted_emotion = model.predict(text_vectorized)

    return predicted_emotion[0]

def predict_naturalness(audio_path):
    # Load the necessary variables
    max_length = 99840  # Max sequence length
    num_mfcc = 8  # Number of MFCC coefficients

    
    # Load and preprocess the new audio sample
    new_audio, sr = librosa.load(audio_path, sr=None, mono=True, res_type='audioread')
    new_audio = librosa.resample(new_audio, orig_sr=sr, target_sr=16000)

    # Extract MFCC features for the new audio sample
    mfcc = librosa.feature.mfcc(y=new_audio, sr=16000, n_mfcc=num_mfcc)

    # Pad or truncate the MFCC features to match the expected shape
    if mfcc.shape[1] < max_length:
        mfcc = pad_sequences([mfcc.T], padding='post', maxlen=max_length, dtype='float32').T
    elif mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]

    # Reshape the MFCC features to match the model's input shape
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.swapaxes(mfcc, 1, 2)  # Swap axes to match the expected shape

    # Perform prediction
    predictions = model.predict(mfcc)
    class_names = ['Very Unnatural', 'Unnatural', 'Neutral', 'Natural', 'Completely Natural']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    # Check if the audio file is present in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'})

    audio_file = request.files['audio']

    # Save the audio file temporarily
    audio_path = 'temp_audio.mp3'
    audio_file.save(audio_path)

    try:
        # Perform speech recognition
        transcription = transcribe_audio(audio_path)

        # Perform emotion recognition
        predicted_emotion = predict_emotion(transcription)

        # Perform naturalness classification
        predicted_naturalness = predict_naturalness(audio_path)

        # Clean up the temporary audio file
        os.remove(audio_path)

        # Return the results as JSON
        result = {
            'naturalness': predicted_naturalness,
            'emotion': predicted_emotion
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




