import json
from transformers.models.tapas.modeling_tapas import ProductIndexMap
import layoutparser as lp
import cv2
import easyocr
from PIL import Image
import pytesseract
import numpy as np
# 1. Cargar imagen
image = "images/imagen1.jpeg"
lenguaje="spa"


# 2. Cargar modelo 
model = lp.Detectron2LayoutModel(
    config_path = "models/config_publay.yml",
    model_path  = "models/model_final.pth",
    extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    label_map   = {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
)
ocr_agent=lp.TesseractAgent(languages=lenguaje)


#funcion detectar layout
def detectar_layout(model, image, output_filename="coordenadas.json"):
    img=Image.open(image)
    # 3. Detectar
    layout = model.detect(img)
    #ordenar coordenadas
    layout.sort(key=lambda b: b.coordinates[1])
    #print(layout)
    data_export=[]
    for block in layout:
        x1,y1,x2,y2=block.coordinates
        item = {
            "label":block.type,
            "score":float(block.score),
            "bbox":[int(x1),int(y1),int(x2),int(y2)]
        }
        data_export.append(item)
    
    with open(output_filename,"w",encoding="utf-8")as f:
        json.dump(data_export,f,indent=4)

    print(f"Se exportaron {len(data_export)} coordenadas a {output_filename}")

    # 4. Dibujar
    viz = lp.draw_box(img, layout, box_width=3)

    # Guardar el resultado en el disco
    viz.save("resultado_detectado.jpg")

    print(f"¡Listo! Revisa el archivo {output_filename} en tu carpeta.")

#funcion para recorgar las bbox de detectados
def recortar_bounging_box(image, output_filename="coordenadas.json"):
    i=1
    image = Image.open(image)
    with open(output_filename,"r",encoding="utf-8")as f:
        data = json.load(f)
    for item in data:
        x1,y1,x2,y2 = item["bbox"]
        recorte = image.crop((x1,y1,x2,y2))
        recorte.save(f"recorte_{item['label']}{i}.jpg")
        i+=1
#se detecta si hay texto o no
# implica EasyOCR
def dectectar_texto(image):
    image = Image.open(image)
    reader = easyocr.Reader(['en'],gpu=False)
    result = reader.readtext(image)
    #print(result)
    return result
    #for detection in result:
    #    print(detection)

#procesar image con pasos
"""
este procesamiento es según la investación de Gemini
es un procesamiento tipo TOP HAT
"""
def preprocesar_pizarra_para_layout(image_path_or_array, debug_save=False):
    # 1. Cargar imagen
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path_or_array}")
    else:
        img = image_path_or_array

    # 2. Convertir a Escala de Grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug_save: cv2.imwrite("debug_1_gray.jpg", gray)

    # 3. Detección y Corrección de Polaridad (Pizarra Negra vs Blanca)
    mean_intensity = np.mean(gray)
    if mean_intensity < 127:
        gray_for_processing = cv2.bitwise_not(gray)
        print("Detectada pizarra oscura. Invirtiendo colores.")
    else:
        gray_for_processing = gray
        print("Detectada pizarra clara.")
    
    if debug_save: cv2.imwrite("debug_2_inverted.jpg", gray_for_processing)


    # --- NÚCLEO DEL PROCESAMIENTO: Operación Top-Hat ---

    # 4. Definir el "Kernel" (Elemento Estructurante)
    kernel_size = (10,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 5. Aplicar Morphological Top-Hat
    tophat = cv2.morphologyEx(gray_for_processing, cv2.MORPH_TOPHAT, kernel)
    
    if debug_save: cv2.imwrite("debug_3_tophat_raw.jpg", tophat)

    # 6. Normalización / Estiramiento de Contraste
    normalized = cv2.normalize(tophat, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    
    if debug_save: cv2.imwrite("debug_4_normalized.jpg", normalized)

    # 7. Binarización (Umbralización de Otsu)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 8. Inversión Final
    final_output_cv = cv2.bitwise_not(binary)
    
    if debug_save: cv2.imwrite("debug_5_final.jpg", final_output_cv)
    final_pil = Image.fromarray(final_output_cv)

    return final_pil

#hac un contraste en la imagen, pero paraece que no funciona mucho
def funcion_contraste(image):
    image = cv2.imread(image)
    ycrcb_image=cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y_channel, cr_channel, cb_channel=cv2.split(ycrcb_image)
    y_channel_stretched=cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
    contrast_stretched_ycrb=cv2.merge((y_channel_stretched, cr_channel, cb_channel))
    contrast_stretched=cv2.cvtColor(contrast_stretched_ycrb, cv2.COLOR_YCR_CB2BGR)
    
    cv2.imwrite('Contrast_stretched_image.jpg', contrast_stretched)

# este es un procesamiento de dos pasos
"""
En este procesamiento se tiene que hacer un procesamiento anterior y uno posterior
empezamos con preprocesar_global y seguimos con
imagen después model
"""
def preprocesar_global(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # 1. Denoising avanzado (crucial para OCR)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    
    # 2. Super-resolución con Real-ESRGAN (si imagen es borrosa)
    # Esto reduce errores de OCR en un 52% según PreP-OCR
    # Instalar: pip install realesrgan
    
    # 3. Mejora de contraste adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 4. Binarización adaptativa (mejor que global)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    img=cv2.bitwise_not(img)
    # Guardar imagen preprocesada
    cv2.imwrite("imagen_preprocesada.jpg", img)


def imagen_despues_modelo(image):
    img=cv2.imread(image)
    segmento_mejorado=cv2.resize(img,None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    segmento_mejorado=cv2.medianBlur(segmento_mejorado, 3)
    cv2.imwrite('segmento_mejorado.png', segmento_mejorado)

    
if __name__ == "__main__":
    detectar_layout(model, image)
    #preprocesar_global(image)
    #detectar_layout(model,"imagen_preprocesada.jpg", "coordenadas3.json")
    #recortar_bounging_box("imagen_preprocesada.jpg", "coordenadas3.json")

""" 
    detectar_layout(model, image)
    recortar_bounging_box(image, "coordenadas.json")


    mi_imagen = "recorte_Title2.jpg" 

    # --- NUEVO PASO: Preprocesar ---
    print("Preprocesando imagen...")
    imagen_procesada_pil = preprocesar_pizarra_para_layout(mi_imagen, debug_save=True)

    print("Detectando layout en imagen procesada...")
    layout = model.detect(imagen_procesada_pil)

    # Visualizar resultado sobre la imagen procesada para verificar
    viz = lp.draw_box(imagen_procesada_pil, layout, box_width=5)
    viz.save("resultado_final_layout.jpg")
    print("¡Listo! Revisa resultado_final_layout.jpg")
    
    detectar_layout(model, image)
    recortar_bounging_box(image, "coordenadas.json")

    
    
    nombres_imagenes=["recorte_Title1.jpg","recorte_Title2.jpg"]
    lista_no_vacios=[]
    for i in nombres_imagenes:
        result=dectectar_texto(image=i)
        if len(result)>0:
            print(f"Imagen {i} tiene los siguientes textos")
            #print(result)
            lista_no_vacios.append(i)

    for i in lista_no_vacios:
        detectar_layout(model, image=i, output_filename="coordenadas2.json")
        recortar_bounging_box(image=i, output_filename="coordenadas2.json")
"""
