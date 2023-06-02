import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Incarcarea imaginilor din folderele mentionate
folder1_path = r'E:\\Pycharm\\pozitie\\clase5Copie\\cameragoala\\'
folder2_path = r'E:\\Pycharm\\pozitie\\clase5Copie\\drept\\'
folder3_path = r'E:\\Pycharm\\pozitie\\clase5Copie\\pedreapta\\'
folder4_path = r'E:\\Pycharm\\pozitie\\clase5Copie\\pefata\\'
folder5_path = r'E:\\Pycharm\\pozitie\\clase5Copie\\pestanga\\'

folder_paths = [folder1_path, folder2_path, folder3_path, folder4_path, folder5_path]
image_data = []
labels = []

for i, folder_path in enumerate(folder_paths):
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        image_data.append(img)
        labels.append(i)

image_data = np.array(image_data)
labels = np.array(labels)

# 2. Definirea si antrenarea unui CNN
num_classes = len(np.unique(labels))

image_data = image_data.reshape(image_data.shape[0], 28, 28, 1)
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data=(X_test, y_test))

# Salvarea modelului
model.save("detection_model.h5")

# 3. Detectia posturii pe baza feedului live de la camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = np.array(gray).reshape(1, 28, 28, 1)
    posture_label = np.argmax(model.predict(gray), axis=-1)

    if posture_label == 0:
        posture_text = "Nicio persoana idendificata"
    elif posture_label == 1:
        posture_text = "Persoana sta drept"
    elif posture_label == 2:
        posture_text = "Persoana sta aplecata pe dreapta"
    elif posture_label == 3:
        posture_text = "Persoana sta aplecata pe fata"
    else:
        posture_text = "Persoana sta aplecata pe stanga"

    cv2.putText(frame, posture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Afișarea imaginii pe ecran
    cv2.imshow('Posture Detection', frame)

    # Așteptarea apăsării unei taste pentru a încheia programul
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break