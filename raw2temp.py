# É necessario instalar o exiftool para executar este script.
# Instalação no Ubuntu: sudo apt install libimage-exiftool-perl

import os
import cv2
import numpy as np
import subprocess

jpg_dir = "imgs-ufpe-frontal/Frontal/sick"
png_dir = "ufpe_thermal/Frontal/sick"
out_dir = "ufpe_temp/Frontal/sick"

os.makedirs(out_dir, exist_ok=True)


def get_exif_param(jpg_path, tag):
    cmd = ["exiftool", jpg_path]
    result = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in result.splitlines():
        if tag.lower() in line.lower():
            return float(line.split(":")[-1].strip().replace(" C", ""))
    return None


def raw_to_temp(raw, p):
    return (p["B"] / np.log(p["R1"] / (p["R2"] * (raw + p["O"])) + p["F"])) - 273.15

 
for filename in os.listdir(jpg_dir):
    if filename.lower().endswith(".jpg"):
        base = os.path.splitext(filename)[0]
        jpg_path = os.path.join(jpg_dir, filename)
        png_path = os.path.join(png_dir, f"{base}_thermal.png")
        out_path = os.path.join(out_dir, f"{base}_temp.txt")

        if not os.path.exists(png_path):
            print(f"[!] PNG não encontrado para: {base}")
            continue

        print(f"[→] Processando: {base}")

        try:
            params = {
                "R1": get_exif_param(jpg_path, "Planck R1"),
                "R2": get_exif_param(jpg_path, "Planck R2"),
                "B":  get_exif_param(jpg_path, "Planck B"),
                "F":  get_exif_param(jpg_path, "Planck F"),
                "O":  get_exif_param(jpg_path, "Planck O")
            }

            if None in params.values():
                print(f"[!] Parâmetros incompletos para: {base}")
                continue

            raw_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            temps = raw_to_temp(raw_img, params)
            np.savetxt(out_path, temps, fmt="%.2f")

            print(f"[✓] Temperatura salva em: {out_path}")
        except Exception as e:
            print(f"[X] Erro ao processar {base}: {e}")

