import numpy as np
import librosa
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

def load_data(data_dir):
    global labels
    file_paths = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            file_paths.append(os.path.join(data_dir, filename))
            labels.append(filename.split('_')[2])  # Etiket dosya adında varsa
    features = []
    for file in file_paths:
        y, sr = librosa.load(file, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        features.append(mfccs)
    return np.array(features), np.array(labels)



# Veri setini yükleyin
X, y = load_data('data/AudioWAV')

# Etiketleri kodlayın
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Eğitim ve test veri setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Modeli oluşturun
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(rate=0.3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# EĞİTİM
model.fit(X_train, y_train, epochs=90, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# SAVE
model.save('model/emotion_recognition_model.h5')

with open('model/label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)



