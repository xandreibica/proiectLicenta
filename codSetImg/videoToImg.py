import cv2
import os

# Path-ul către videoclip
video_path = 'E:\Pycharm\pozitie\VIDEOPOZ\cameragoala.mp4'

# Directorul în care se vor salva imaginile
output_dir = 'E:\Pycharm\pozitie\clase5Copie\cameragoala'

# Intervalul de timp în secunde între extragerea de cadre
frame_interval = 0.5

# Creați directorul de ieșire dacă nu există deja
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Deschideți videoclipul folosind cv2.VideoCapture
cap = cv2.VideoCapture(video_path)

# calculeaza numarul total de cadre din videoclip
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# loop prin fiecare cadru și salvează-l ca imagine
for i in range(num_frames):
    # citeste cadru cu cadru
    ret, frame = cap.read()

    # verifica daca cadru a fost citit cu succes
    if ret:
        # construieste numele fișierului de ieșire
        output_file = os.path.join(output_dir, f"frame_{i:06d}.jpg")

        # salveaza cadru ca imagine
        cv2.imwrite(output_file, frame)

        # afișează progresul
        print(f"Salvare cadru {i + 1}/{num_frames}")
    else:
        print(f"Eroare la citirea cadru {i + 1}/{num_frames}")

# inchide fisierul video
cap.release()

# afișează mesaj de finalizare
print("Salvare imagini finalizata.")
