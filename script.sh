
# É necessario instalar o exiftool para executar este script.
# Instalação no Ubuntu: sudo apt install libimage-exiftool-perl

INPUT_DIR="./imgs-ufpe-frontal/Frontal/sick"                
OUTPUT_DIR="./ufpe_thermal/Frontal/sick"        


mkdir -p "$OUTPUT_DIR"

# Loop por cada arquivo .jpg
for IMG in "$INPUT_DIR"/*.jpg; do
  # Extrai o nome do arquivo sem extensão
  FILENAME=$(basename "$IMG" .jpg)
  
  # Caminho de saída
  OUTFILE="$OUTPUT_DIR/${FILENAME}_thermal.png"
  
  # Comando de extração
  exiftool -b -RawThermalImage "$IMG" > "$OUTFILE"
  
  # Feedback no terminal
  echo "Extraído: $OUTFILE"
done
