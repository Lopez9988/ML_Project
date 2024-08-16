#Ivan Lopez
#Naive Bayes audio feature classifier


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import csv
import random
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
#Keras
import keras

'''
#Creating the dataset
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#calculate the features of the audio files
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
'''

#Reading the dataset and dropping unnecessary columns
data = pd.read_csv('data.csv')
data.head() 

data = data.drop(['filename'],axis=1)
data.head()

#Encoding genres into integers
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
#print(y)

# normalizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
#print(X)

#Split into training & testing data (test:train = 3:7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random.randint(0,1000000))

#Gaussian Naive Bayes Model
model = GaussianNB()
#Provide training inputs & expected to model
model.fit(X_train, y_train)

#Import model stats
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

#Get output predictions from test inputs
y_pred = model.predict(X_test)

#Get model stats
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
#Loss Function (TO do)
#Look into model parameters

labels = [
    "Blues",
    "Classical",
    "Country",
    "Disco",
    "Hiphop",
    "Jazz",
    "Metal",
    "Pop",
    "Reggae",
    "Rock",
]

#Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation='vertical')
plt.tight_layout()
plt.show()
#plt.savefig('NB_ConfMatrix.png')