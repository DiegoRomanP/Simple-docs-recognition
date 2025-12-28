"""
Módulo para preprocesar imágenes antes de la detección de layout.
"""

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Clase para preprocesar imágenes antes de la detección"""

    """Las funciones preprocesar_pizarra_para_layout y funcion_contraste son para etapa de detección"""

    @staticmethod
    def preprocesar_pizarra_para_layout(image_path_or_array, debug_save=False): 
        """ Preprocesa una imagen de pizarra usando operación Top-Hat Args:
            image_path_or_array: Ruta de imagen o array numpy
            debug_save: Si guardar imágenes intermedias para debug

        Returns:
            Imagen PIL procesada
        """
        # 1. Cargar imagen
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path_or_array}")
        else:
            img = image_path_or_array

        # 2. Convertir a Escala de Grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug_save:
            cv2.imwrite("debug_1_gray.jpg", gray)

        # 3. Detección y Corrección de Polaridad (Pizarra Negra vs Blanca)
        mean_intensity = np.mean(gray)
        if mean_intensity < 127:
            gray_for_processing = cv2.bitwise_not(gray)
            print("Detectada pizarra oscura. Invirtiendo colores.")
        else:
            gray_for_processing = gray
            print("Detectada pizarra clara.")

        if debug_save:
            cv2.imwrite("debug_2_inverted.jpg", gray_for_processing)

        # --- NÚCLEO DEL PROCESAMIENTO: Operación Top-Hat ---

        # 4. Definir el "Kernel" (Elemento Estructurante)
        kernel_size = (10, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        # 5. Aplicar Morphological Top-Hat
        tophat = cv2.morphologyEx(gray_for_processing, cv2.MORPH_TOPHAT, kernel)

        if debug_save:
            cv2.imwrite("debug_3_tophat_raw.jpg", tophat)

        # 6. Normalización / Estiramiento de Contraste
        normalized = cv2.normalize(tophat, None, 0, 255, norm_type=cv2.NORM_MINMAX)

        if debug_save:
            cv2.imwrite("debug_4_normalized.jpg", normalized)

        # 7. Binarización (Umbralización de Otsu)
        _, binary = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # 8. Inversión Final
        final_output_cv = cv2.bitwise_not(binary)

        if debug_save:
            cv2.imwrite("debug_5_final.jpg", final_output_cv)

        final_pil = Image.fromarray(final_output_cv)

        return final_pil

    @staticmethod
    def funcion_contraste(image_path):
        """
        Aplica mejora de contraste usando espacio de color YCrCb
        Esto es realce de imagen, mantiene los colores y hcae el texto resaltante

        Args:
            image_path: Ruta de la imagen
        """
        image = cv2.imread(image_path)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
        contrast_stretched_ycrb = cv2.merge(
            (y_channel_stretched, cr_channel, cb_channel)
        )
        contrast_stretched = cv2.cvtColor(contrast_stretched_ycrb, cv2.COLOR_YCR_CB2BGR)

        cv2.imwrite("Contrast_stretched_image.jpg", contrast_stretched)

    """Las funciones preprocesar_global y imagen_despues_modelo son para después de haber detectado el layout, es decir para la preparación del OCR"""

    @staticmethod
    def preprocesar_global(ruta_imagen, output_path="imagen_preprocesada.jpg"):
        """
        Preprocesamiento global avanzado para OCR

        Args:
            ruta_imagen: Ruta de la imagen a procesar
            output_path: Ruta de salida de la imagen procesada

        Returns:
            Ruta del archivo guardado
        """
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

        # 1. Denoising avanzado (crucial para OCR)
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        # 2. Mejora de contraste adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # 3. Binarización adaptativa (mejor que global)
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        img = cv2.bitwise_not(img)

        # Guardar imagen preprocesada
        cv2.imwrite(output_path, img)

        return output_path

    @staticmethod
    def imagen_despues_modelo(image_path, output_path="segmento_mejorado.png"):
        """
        Mejora la imagen después del modelo (upsampling + denoising)

        Args:
            image_path: Ruta de la imagen
            output_path: Ruta de salida
        """
        img = cv2.imread(image_path)
        segmento_mejorado = cv2.resize(
            img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        segmento_mejorado = cv2.medianBlur(segmento_mejorado, 3)
        cv2.imwrite(output_path, segmento_mejorado)
