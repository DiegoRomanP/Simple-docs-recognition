import cv2
import numpy as np
from PIL import Image
from layout import detectar_layout
def preprocesar_pizarra_para_layout(image_path_or_array, debug_save=False):
    """
    Preprocesa imágenes de pizarras (blanca o negra) con alto contraste para
    optimizar la detección con LayoutParser.

    Args:
        image_path_or_array (str o np.ndarray): Ruta de la imagen o array de OpenCV.
        debug_save (bool): Si es True, guarda pasos intermedios para visualizar.

    Returns:
        PIL.Image: Imagen procesada lista para LayoutParser (fondo blanco, texto negro).
    """

    # 1. Cargar imagen
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path_or_array}")
    else:
        img = image_path_or_array

    # 2. Convertir a Escala de Grises
    # El color distrae a los modelos de layout; solo nos importa la estructura.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug_save: cv2.imwrite("debug_1_gray.jpg", gray)

    # 3. Detección y Corrección de Polaridad (Pizarra Negra vs Blanca)
    # Calculamos la intensidad media. Si es baja (<127), es una pizarra oscura.
    # LayoutParser prefiere fondo claro y texto oscuro, así que si es oscura, invertimos.
    mean_intensity = np.mean(gray)
    if mean_intensity < 127:
        # Es pizarra negra/verde -> Invertir para que el texto sea oscuro
        gray_for_processing = cv2.bitwise_not(gray)
        print("Detectada pizarra oscura. Invirtiendo colores.")
    else:
        # Es pizarra blanca -> Dejar como está
        gray_for_processing = gray
        print("Detectada pizarra clara.")
    
    if debug_save: cv2.imwrite("debug_2_inverted.jpg", gray_for_processing)


    # --- NÚCLEO DEL PROCESAMIENTO: Operación Top-Hat ---

    # 4. Definir el "Kernel" (Elemento Estructurante)
    # Este cuadrado debe ser más grande que el grosor del trazo de tiza/plumón,
    # pero más pequeño que la pizarra entera. Un tamaño de 40x40 suele funcionar
    # bien para fotos de resolución estándar.
    kernel_size = (40, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 5. Aplicar Morphological Top-Hat
    # Esta operación hace matemáticamente esto: (Imagen Original) - (Apertura de la Imagen).
    # La "Apertura" borra los detalles pequeños (texto) y deja solo el fondo con su iluminación.
    # Al restar, el resultado es SOLO los detalles pequeños (texto) sobre un fondo negro plano.
    tophat = cv2.morphologyEx(gray_for_processing, cv2.MORPH_TOPHAT, kernel)
    
    if debug_save: cv2.imwrite("debug_3_tophat_raw.jpg", tophat)

    # 6. Normalización / Estiramiento de Contraste
    # La imagen 'tophat' resultante es muy oscura. Estiramos el histograma para que
    # lo que sea un poco brillante se vuelva blanco puro y lo oscuro, negro puro.
    normalized = cv2.normalize(tophat, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    
    if debug_save: cv2.imwrite("debug_4_normalized.jpg", normalized)

    # 7. Binarización (Umbralización de Otsu)
    # Convertimos a blanco y negro puro. Otsu encuentra automáticamente el mejor punto de corte.
    # Usamos normalized como entrada.
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 8. Inversión Final
    # El paso Top-Hat (paso 5) siempre genera texto blanco sobre fondo negro.
    # Como queremos que LayoutParser vea texto NEGRO sobre fondo BLANCO,
    # invertimos el resultado final.
    final_output_cv = cv2.bitwise_not(binary)
    
    if debug_save: cv2.imwrite("debug_5_final.jpg", final_output_cv)

    # Convertir de vuelta a PIL para LayoutParser
    final_pil = Image.fromarray(final_output_cv)

    return final_pil


# =========================================
# Ejemplo de Uso con tu script de LayoutParser
# =========================================

# Supongamos que tienes tu código anterior:
# ... import layoutparser as lp ...
# model = lp.Detectron2LayoutModel(...)

# Ruta de tu imagen (la de la pizarra verde)
mi_imagen = "recorte_Title2.jpg" 

# --- NUEVO PASO: Preprocesar ---
print("Preprocesando imagen...")
# Usamos debug_save=True la primera vez para ver qué hace en la carpeta del proyecto
imagen_procesada_pil = preprocesar_pizarra_para_layout(mi_imagen, debug_save=True)

# --- CONTINUAR CON LAYOUTPARSER ---
print("Detectando layout en imagen procesada...")
# NOTA: model.detect puede recibir directamente la imagen PIL
layout = model.detect(imagen_procesada_pil)

# Visualizar resultado sobre la imagen procesada para verificar
viz = lp.draw_box(imagen_procesada_pil, layout, box_width=5)
viz.save("resultado_final_layout.jpg")
print("¡Listo! Revisa resultado_final_layout.jpg")

# Para el recorte posterior, recuerda usar las coordenadas del layout
# sobre la imagen ORIGINAL (si quieres los colores) o sobre la PROCESADA (si quieres OCR limpio).
# Generalmente para OCR, es mejor usar la procesada.