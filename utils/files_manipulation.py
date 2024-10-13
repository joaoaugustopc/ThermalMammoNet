
from include.imports import *

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Pasta {folder_path} deletada.")
    else:
        print(f"Pasta {folder_path} não encontrada.")

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Arquivo {file_path} deletado.")
    else:
        print(f"Arquivo {file_path} não encontrado.")

def move_files_to_folder(file_list, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in file_list:
        if os.path.exists(file):
            shutil.move(file, destination_folder)