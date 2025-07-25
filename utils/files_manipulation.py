
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
            
def rename_file(current_path, new_path):
    if os.path.exists(current_path):
        os.rename(current_path, new_path)
        print(f"Arquivo {current_path} renomeado para {new_path}.")
    else:
        print(f"Arquivo {current_path} não encontrado.")
        
    
def rename_folder(current_folder_path, new_folder_path):
    if os.path.exists(current_folder_path):
        os.rename(current_folder_path, new_folder_path)
        print(f"Pasta {current_folder_path} renomeada para {new_folder_path}.")
    else:
        print(f"Pasta {current_folder_path} não encontrada.")

def move_files_within_folder(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file_name in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file_name)
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_folder)
            print(f"Arquivo {source_file} movido para {destination_folder}.")
        else:
            print(f"{source_file} não é um arquivo.")

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Pasta {folder_path} criada.")
    else:
        print(f"Pasta {folder_path} já existe.")


def copy_file(source_file, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_folder)
        print(f"Arquivo {source_file} copiado para {destination_folder}.")
    else:
        print(f"Arquivo {source_file} não encontrado.")

def move_folder(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if os.path.exists(source_folder):
        shutil.move(source_folder, destination_folder)
        print(f"Pasta {source_folder} movida para {destination_folder}.")
    else:
        print(f"Pasta {source_folder} não encontrada.")