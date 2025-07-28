import cv2
import numpy as np
import pytesseract
import os

# --- Configurações de Depuração ---
DEBUG_SAVE_IMAGES = True # Mude para False quando não estiver depurando
DEBUG_OUTPUT_DIR = "debug_output" # Pasta para salvar as imagens de depuração
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
# --- Fim Configurações de Depuração ---


def find_color_bar_and_temps(image_path):
    """
    Tenta encontrar a barra de cores e os números de temperatura
    na imagem usando técnicas de visão computacional tradicionais e OCR.

    Args:
        image_path (str): Caminho para a imagem.

    Returns:
        tuple: (min_temp_val, max_temp_val, colormap_extracted, main_image_region_without_bar)
               Retorna None para qualquer um se não for encontrado.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, None, None, None

    img_h, img_w, _ = img.shape
    
    # 1. Definir a Região de Interesse (ROI) onde a barra é esperada (lado direito)
    roi_x_start = int(img_w * 0.90)
    roi_x_end = img_w - 2          
    roi_y_start = int(img_h * 0.05)
    roi_y_end = int(img_h * 0.95)   
    
    # Certificar-se que a ROI é válida
    if roi_x_start < 0: roi_x_start = 0
    if roi_x_end > img_w: roi_x_end = img_w
    if roi_y_start < 0: roi_y_start = 0
    if roi_y_end > img_h: roi_y_end = img_h

    roi_search_area = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end].copy()
    
    if roi_search_area.shape[0] == 0 or roi_search_area.shape[1] == 0:
        print(f"ROI de busca da barra inválida para {image_path}")
        return None, None, None, None

    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_01_roi_search_area.jpg"), roi_search_area) # cite: 1, 3, 5, 13

    # 2. Segmentar a Barra de Cores na ROI
    hsv_roi = cv2.cvtColor(roi_search_area, cv2.COLOR_BGR2HSV)

    # --- REVISÃO NAS FAIXAS DE HSV ---

    # ajustar aqui
    lower_red1 = np.array([0, 100, 80])    
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80]) 
    upper_red2 = np.array([179, 255, 255]) 

    lower_yellow = np.array([15, 80, 60]) 
    upper_yellow = np.array([40, 255, 255])

    lower_green = np.array([40, 100, 80])   
    upper_green = np.array([80, 255, 255]) 
    
    lower_cyan = np.array([80, 100, 100])   
    upper_cyan = np.array([130, 255, 255])

    lower_purple = np.array([140, 180, 120])  
    upper_purple = np.array([160, 255, 255]) 

    mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
    mask_cyan = cv2.inRange(hsv_roi, lower_cyan, upper_cyan)
    mask_purple = cv2.inRange(hsv_roi, lower_purple, upper_purple)
    
    combined_mask = mask_red1 | mask_red2 | mask_yellow | mask_green | mask_cyan | mask_purple

    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_02_combined_mask.jpg"), combined_mask) # cite: 4, 7, 8, 9, 10, 11, 12

    # Operações morfológicas para limpar a máscara e conectar partes da barra
    # Kernel ajustado para um valor que mostrou bons resultados antes
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size),np.uint8) 
    mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_03_mask_cleaned.jpg"), mask_cleaned) # cite: 2, 6

    # Encontrar contornos na máscara limpa
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_bar_bbox = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200: 
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)
            
            # Ajuste nos limites de w, se a barra for muito fina, ou muito grossa
            # Ajuste no aspect_ratio para ser mais flexível, se necessário.
            if aspect_ratio > 3.0 and w > 10 and w < 40: 
                x_original = x + roi_x_start
                y_original = y + roi_y_start

                if (x_original >= roi_x_start - 5 and x_original + w <= roi_x_end + 5 and
                    y_original >= roi_y_start - 5 and y_original + h <= roi_y_end + 5):
                    if area > max_area:
                        max_area = area
                        color_bar_bbox = (x_original, y_original, w, h)
    
    if color_bar_bbox is None:
        print(f"Não foi possível encontrar a barra de cores em {image_path}")
        return None, None, None, None
        
    x_bar, y_bar, w_bar, h_bar = color_bar_bbox
    
    if DEBUG_SAVE_IMAGES:
        img_with_bar_bbox = img.copy()
        cv2.rectangle(img_with_bar_bbox, (x_bar, y_bar), (x_bar + w_bar, y_bar + h_bar), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_04_bar_detected.jpg"), img_with_bar_bbox) # cite: 0

    # 3. Localizar os Números (27 e 36) próximos à barra
    
    # ROIs ajustadas para serem um pouco mais generosas ou alinhadas com a barra.
    # Ajustar estes valores com base na visualização de _04_bar_detected.jpg
    # para garantir que os números estejam dentro do retângulo.
    roi_top_num_x_start = x_bar + w_bar + 5 
    roi_top_num_x_end = x_bar + w_bar + 50 
    roi_top_num_y_start = y_bar - 20 
    roi_top_num_y_end = y_bar + 30    

    roi_top_num_x_start = max(0, min(roi_top_num_x_start, img_w - 1))
    roi_top_num_x_end = max(0, min(roi_top_num_x_end, img_w - 1))
    roi_top_num_y_start = max(0, min(roi_top_num_y_start, img_h - 1))
    roi_top_num_y_end = max(0, min(roi_top_num_y_end, img_h - 1))

    # ROI para o número inferior (Min Temp)
    roi_bottom_num_x_start = x_bar + w_bar + 5
    roi_bottom_num_x_end = x_bar + w_bar + 50 
    roi_bottom_num_y_start = y_bar + h_bar - 30
    roi_bottom_num_y_end = y_bar + h_bar + 20   

    roi_bottom_num_x_start = max(0, min(roi_bottom_num_x_start, img_w - 1))
    roi_bottom_num_x_end = max(0, min(roi_bottom_num_x_end, img_w - 1))
    roi_bottom_num_y_start = max(0, min(roi_bottom_num_y_start, img_h - 1))
    roi_bottom_num_y_end = max(0, min(roi_bottom_num_y_end, img_h - 1))

    roi_top_num = img[roi_top_num_y_start:roi_top_num_y_end, roi_top_num_x_start:roi_top_num_x_end]
    roi_bottom_num = img[roi_bottom_num_y_start:roi_bottom_num_y_end, roi_bottom_num_x_start:roi_bottom_num_x_end]
    
    if DEBUG_SAVE_IMAGES:
        if roi_top_num.shape[0] > 0 and roi_top_num.shape[1] > 0:
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_05_roi_top_num.jpg"), roi_top_num) # cite: 0
        if roi_bottom_num.shape[0] > 0 and roi_bottom_num.shape[1] > 0:
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_06_roi_bottom_num.jpg"), roi_bottom_num)
    
    # Pré-processamento e OCR para os números
    def ocr_number_from_roi(roi_img, image_base_name=""):
        if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
            return None
        
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Adicionar desfoque mediano para remover ruído antes do threshold
        gray_roi = cv2.medianBlur(gray_roi, 3) 
        
        # Tente diferentes valores para o segundo parâmetro se o Otsu não funcionar bem
        # _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV) 
        _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        thresh_roi = cv2.copyMakeBorder(thresh_roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0])

        if DEBUG_SAVE_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{image_base_name}_thresh_ocr.jpg"), thresh_roi)

        custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata/" --oem 3 --psm 6 outputbase digits'        
        text = pytesseract.image_to_string(thresh_roi, config=custom_config)
        
        cleaned_text = ''.join(filter(str.isdigit, text))
        try:
            return int(cleaned_text)
        except ValueError:
            return None

    base_name = os.path.basename(image_path).split('.')[0] 
    num_top = ocr_number_from_roi(roi_top_num, f"{base_name}_top") # cite: 0
    num_bottom = ocr_number_from_roi(roi_bottom_num, f"{base_name}_bottom")

    min_temp_val = None
    max_temp_val = None
    
    if isinstance(num_top, int) and isinstance(num_bottom, int):
        if num_top > num_bottom:
            max_temp_val = num_top
            min_temp_val = num_bottom
        elif num_top < num_bottom:
            print(f"Aviso: Números detectados indicam escala invertida ou erro de OCR: Top={num_top}, Bottom={num_bottom} para {image_path}. Tentando atribuir.")
            max_temp_val = max(num_top, num_bottom)
            min_temp_val = min(num_top, num_bottom)
        else:
            print(f"Aviso: Números detectados são iguais: Top={num_top}, Bottom={num_bottom} para {image_path}. Não é possível determinar a faixa completa.")
            return None, None, None, None
    elif isinstance(num_top, int):
        print(f"Aviso: Apenas o número superior foi detectado ({num_top}) para {image_path}. Não é possível determinar a faixa completa.")
        return None, None, None, None
    elif isinstance(num_bottom, int):
        print(f"Aviso: Apenas o número inferior foi detectado ({num_bottom}) para {image_path}. Não é possível determinar a faixa completa.")
        return None, None, None, None
    else:
        print(f"Erro: Não foi possível detectar nenhum valor de temperatura na imagem {image_path}. Detectado: Top={num_top}, Bottom={num_bottom}")
        return None, None, None, None

    if min_temp_val is None or max_temp_val is None:
        print(f"Erro inesperado na atribuição de valores de temperatura para {image_path}.")
        return None, None, None, None

    # 4. Extrair o Colormap Dinamicamente da barra detectada
    colormap_extracted = []
    actual_color_bar_img = img[y_bar : y_bar + h_bar, x_bar : x_bar + w_bar]
    
    cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_color_bar_extracted.jpg"), actual_color_bar_img)

    sampling_step = max(1, h_bar // 256) 

    for y_pixel in range(h_bar - 1, -1, -sampling_step): 
        avg_color_bgr = np.mean(actual_color_bar_img[y_pixel, :, :], axis=0).astype(np.uint8)
        colormap_extracted.append(avg_color_bgr.tolist()) 
    
    # 5. Definir a região principal da imagem (sem a barra de temperatura)
    main_image_region_without_bar = img[:, :x_bar].copy()
    
    main_image_gray = cv2.cvtColor(main_image_region_without_bar, cv2.COLOR_BGR2GRAY)
    
    # Após converter para escala de cinza
    main_image_gray = cv2.normalize(main_image_gray, None, 0, 255, cv2.NORM_MINMAX)
    main_image_gray = main_image_gray.astype(np.uint8)
    
    # 2. Aplica o colormap extraído
    img_termica = apply_thermal_colormap(main_image_gray, min_temp_val, max_temp_val, colormap_extracted)
    cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"{os.path.basename(image_path)}_thermal_result.jpg"), img_termica)
    
    
    return min_temp_val, max_temp_val, colormap_extracted, main_image_gray

def apply_thermal_colormap(main_image_region_gray, min_temp, max_temp, colormap_extracted):
    """
    Aplica o colormap extraído a uma imagem em tons de cinza.

    Args:
        main_image_region_gray (numpy.ndarray): A imagem em tons de cinza a ser termalizada.
        min_temp (int): Temperatura mínima detectada.
        max_temp (int): Temperatura máxima detectada.
        colormap_extracted (list): Lista de cores BGR que formam o colormap.

    Returns:
        numpy.ndarray: A imagem termalizada.
    """
    if main_image_region_gray is None or len(colormap_extracted) == 0:
        return None

    img_normalized = main_image_region_gray / 255.0

    if (max_temp - min_temp) == 0:
        print("Aviso: Faixa de temperatura é zero (max_temp == min_temp). Não é possível aplicar colormap.")
        return np.zeros_like(main_image_region_gray) 

    temperatures = min_temp + (img_normalized * (max_temp - min_temp))

    img_termica = np.zeros((main_image_region_gray.shape[0], main_image_region_gray.shape[1], 3), dtype=np.uint8)
    
    num_colors = len(colormap_extracted)

    for r in range(main_image_region_gray.shape[0]):
        for c in range(main_image_region_gray.shape[1]):
            temp_pixel = temperatures[r, c]

            color_index_float = (temp_pixel - min_temp) / (max_temp - min_temp) * (num_colors - 1)
            
            color_index_float = np.clip(color_index_float, 0, num_colors - 1)
            
            idx1 = int(color_index_float)
            idx2 = min(idx1 + 1, num_colors - 1)

            ratio = color_index_float - idx1

            color1 = np.array(colormap_extracted[idx1])
            color2 = np.array(colormap_extracted[idx2])

            interpolated_color = (color1 * (1 - ratio) + color2 * ratio).astype(np.uint8)
            img_termica[r, c] = interpolated_color

    return img_termica