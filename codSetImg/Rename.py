import os

folder_path = "E:\Pycharm\pozitie\clase5Copie\pedreapta"
new_name = "pedreapta_"

# lista cu numele tuturor fișierelor din director
files = os.listdir(folder_path)

# iterează prin fiecare fișier și redenumește-l
for i, file_name in enumerate(files):
    # obține extensia fișierului
    file_ext = os.path.splitext(file_name)[1]
    # construiește noul nume de fișier
    new_file_name = new_name + str(i+1) + file_ext
    # construiește calea completă către fișierul vechi
    old_file_path = os.path.join(folder_path, file_name)
    # construiește calea completă către fișierul nou
    new_file_path = os.path.join(folder_path, new_file_name)
    # redenumește fișierul
    os.rename(old_file_path, new_file_path)
