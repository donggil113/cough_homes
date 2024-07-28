import os
import numpy as np
import librosa
from pydub import AudioSegment
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 데이터 경로 설정
data_path = 'path_to_cough_sound_files'
labels = ['disease1', 'disease2', 'disease3']  # 예시 레이블

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    file_path_wav = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(file_path_wav, format='wav')
    return file_path_wav

def trim_audio(file_path, duration=60):
    audio = AudioSegment.from_wav(file_path)
    if len(audio) > duration * 1000:
        audio = audio[:duration * 1000]
    audio.export(file_path, format='wav')

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc
    
@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    file_path = f'./{audio_file.filename}'
    audio_file.save(file_path)
    
    mfcc = preprocess_audio(file_path)
    prediction = model.predict(np.array([mfcc]))
    predicted_class = np.argmax(prediction, axis=1)
    
    return jsonify({'prediction': int(predicted_class[0])})

@app.route('/train', methods=['POST'])
def train():
    audio_file = request.files['file']
    true_label = int(request.form['label'])
    file_path = f'./{audio_file.filename}'
    audio_file.save(file_path)
    
    mfcc = preprocess_audio(file_path)
    X_new = np.array([mfcc])
    y_new = np.array([true_label])
    
    model.fit(X_new, y_new, epochs=1, verbose=0)
    model.save('model.h5')
    
    return jsonify({'status': 'Model updated'})

def predict_disease(file_path):
    mfcc = preprocess_audio(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # 배치 차원 추가
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

features = []
labels_list = []

for label in labels:
    folder_path = os.path.join(data_path, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # 파일 확장자 변환 및 트리밍
        file_path_wav = convert_to_wav(file_path)
        trim_audio(file_path_wav)

        mfccs = extract_features(file_path_wav)
        features.append(mfccs)
        labels_list.append(label)

# 데이터 배열로 변환
X = np.array(features)
y = np.array([labels.index(label) for label in labels_list])

# 레이블을 원-핫 인코딩
y = to_categorical(y, num_classes=len(labels))

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

# 모델 구축
model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 형상 조정
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 모델 훈련
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 모델 저장
model.save('cough_classification_model.h5')

# 모델 불러오기
from tensorflow.keras.models import load_model
loaded_model = load_model('cough_classification_model.h5')

# Function to convert audio file to mel spectrogram
def extract_mel_spectrogram(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.repeat(spectrogram[..., np.newaxis], 3, -1)  # Repeat to simulate RGB channels
    return spectrogram

# Preprocess audio files and extract features
def preprocess_audio_files(data_path, labels):
    features = []
    labels_list = []

    for label in labels:
        folder_path = os.path.join(data_path, label)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            file_path_wav = convert_to_wav(file_path)
            trim_audio(file_path_wav)

            # Extract mel spectrogram
            spectrogram = extract_mel_spectrogram(file_path_wav)
            features.append(spectrogram)
            labels_list.append(label)

            # Data augmentation
            augmented_paths = augment_data(file_path_wav)
            for augmented_path in augmented_paths:
                spectrogram_aug = extract_mel_spectrogram(augmented_path)
                features.append(spectrogram_aug)
                labels_list.append(label)

    return np.array(features), np.array([labels.index(label) for label in labels_list])

import soundfile as sf

def augment_data(file_path):
    y, sr = librosa.load(file_path, sr=None)
    augmented_paths = []

    # 시간 축소 및 확대
    speed_change = np.random.uniform(0.9, 1.1)
    y_changed = librosa.effects.time_stretch(y, rate=speed_change)
    augmented_path = file_path.rsplit('.', 1)[0] + f'_aug_{speed_change:.2f}.wav'
    sf.write(augmented_path, y_changed, sr)
    augmented_paths.append(augmented_path)
    
    # 피치 변경
    pitch_change = np.random.uniform(-2, 2)
    y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_change)
    augmented_path_pitch = file_path.rsplit('.', 1)[0] + f'_pitch_{pitch_change:.2f}.wav'
    sf.write(augmented_path_pitch, y_pitched, sr)
    augmented_paths.append(augmented_path_pitch)
    
    return augmented_paths
    
features = []
labels_list = []

for label in labels:
    folder_path = os.path.join(data_path, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 파일 확장자 변환 및 트리밍
        file_path_wav = convert_to_wav(file_path)
        trim_audio(file_path_wav)
        
        mfccs = extract_features(file_path_wav)
        features.append(mfccs)
        labels_list.append(label)
        
        # 증강 데이터 특징 추출 (추가된 부분)
        augmented_paths = augment_data(file_path_wav)
        for augmented_path in augmented_paths:
            mfccs_aug = extract_features(augmented_path)
            features.append(mfccs_aug)
            labels_list.append(label)

from flask import Flask, request, jsonify

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    file_path = f'./{audio_file.filename}'
    audio_file.save(file_path)
    
    predicted_class = predict_disease(file_path)
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)

import React, { useState } from 'react';
import { View, Button, Text, Platform } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';

export default function App() {
  const [result, setResult] = useState(null);

  const pickAudio = async () => {
    let result = await DocumentPicker.getDocumentAsync({
      type: 'audio/*',
    });

    if (result.type === 'success') {
      const fileUri = result.uri;
      const formData = new FormData();
      formData.append('file', {
        uri: fileUri,
        name: 'audio.wav',
        type: 'audio/wav',
      });

      const response = await fetch('http://our-backend-server-url/predict', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();
      setResult(data.prediction);
    }
  };

  return (
    <View style={{ padding: 20 }}>
      <Button title="Pick an Audio File" onPress={pickAudio} />
      {result !== null && <Text>Prediction: {result}</Text>}
    </View>
  );
}

import tensorflow as tf

# 기존 모델 로드
model = tf.keras.models.load_model('model.h5')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 모델 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save the model
model.save('cough_classification_transformer.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('cough_classification_transformer.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the pre-trained model
model = load_model('cough_classification_transformer.h5')

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# SHAP expects the model output probabilities
model.probabilities = lambda x: model.predict(x, verbose=0)

# Function to preprocess a single audio file and get the mel spectrogram
def preprocess_single_audio(file_path):
    file_path_wav = convert_to_wav(file_path)
    trim_audio(file_path_wav)
    spectrogram = extract_mel_spectrogram(file_path_wav)
    spectrogram = spectrogram / 255.0  # Normalize
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    return spectrogram

# Define a function to predict the probabilities for LIME
def predict_proba(images):
    images = np.array([gray2rgb(img) for img in images])  # Convert to 3 channels
    images = images / 255.0  # Normalize
    return model.predict(images)

# Generate a background dataset
X_background = X_train[:100]  # Assuming X_train is the training data array

# Create SHAP explainer
explainer_shap = shap.KernelExplainer(model.probabilities, X_background)

# Function to explain predictions
def explain_prediction(file_path):
    spectrogram = preprocess_single_audio(file_path)
    shap_values = explainer_shap.shap_values(spectrogram)
    
    # Plot the explanation
    plt.figure(figsize=(10, 5))
    shap.image_plot(shap_values, -spectrogram)  # The '-' is used because SHAP expects the input to be image-like (2D)
    plt.show()

# Example of explaining a prediction
explain_prediction('our_audio_file.wav')

# Create a LIME image explainer
explainer_lime = lime_image.LimeImageExplainer()

# Function to explain predictions using LIME
def explain_prediction(file_path):
    # Preprocess the audio to get a spectrogram
    spectrogram = preprocess_single_audio(file_path)
    spectrogram_rgb = gray2rgb(spectrogram)  # Convert to RGB format expected by LIME

    # Explain the model's prediction
    explanation = explainer_lime.explain_instance(
        spectrogram_rgb, 
        classifier_fn=predict_proba,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Get the explanation for the top class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=10, 
        hide_rest=True
    )

    # Display the explanation
    plt.imshow(temp, interpolation='nearest')
    plt.show()

# Example of explaining a prediction
explain_prediction('our_audio_file.wav')

import React, { useState } from 'react';
import { View, Button, Text } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import * as tflite from '@tensorflow/tfjs-tflite';
import * as FileSystem from 'expo-file-system';
import * as Audio from 'expo-av';

export default function App() {
  const [result, setResult] = useState(null);
  const [model, setModel] = useState(null);

  const loadModel = async () => {
    const modelJson = await require('./model.tflite'); // 모델 로드
    const model = await tflite.loadTFLiteModel(modelJson);
    setModel(model);
  };

  const predict = async (audioUri) => {
    const response = await fetch(audioUri);
    const audioData = await response.arrayBuffer();
    const audioTensor = tf.tensor(new Float32Array(audioData)); // 오디오 데이터를 텐서로 변환

    const prediction = model.predict(audioTensor);
    setResult(prediction);
  };

  const pickAudio = async () => {
    // 오디오 녹음 및 파일 선택 로직 추가
    const audioUri = 'path/to/our/audio/file.wav';
    await predict(audioUri);
  };

  React.useEffect(() => {
    loadModel();
  }, []);

  return (
    <View style={{ padding: 20 }}>
      <Button title="Pick an Audio File" onPress={pickAudio} />
      {result && <Text>Prediction: {result}</Text>}
    </View>
  );
}

