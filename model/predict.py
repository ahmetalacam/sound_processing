from keras.models import load_model
import numpy as np
import librosa
import pickle

# Modeli ve LabelEncoder'ı yükleyin
model = load_model('model/emotion_recognition_model.h5')
with open('model/label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Ses dosyasını ön işleme fonksiyonu
def preprocess_audio_for_prediction(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = mfccs.reshape(1, -1, 1)  # Modelin giriş formatına uyacak şekilde genişletin
    return mfccs

# Tahmin yapma fonksiyonu
def predict_emotion(audio_file):
    # Ses dosyasını ön işleme tabi tutun
    features = preprocess_audio_for_prediction(audio_file)
    
    # Tahmin yapın
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    
    # Tahmin edilen etiketi dönüştürün
    predicted_emotion = label_encoder.inverse_transform([predicted_label])
    return predicted_emotion[0]


# Tahmin yapmak istediğiniz ses dosyasının yolunu belirtin
audio_file_path = 'data/tess_crema/OAF_dog_SAD_X.wav'
predicted_emotion = predict_emotion(audio_file_path)
print(f"Predicted emotion: {predicted_emotion}")
