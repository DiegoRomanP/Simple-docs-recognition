"""
Módulo para preprocesar imágenes antes de la detección de layout.
"""

import cv2
import numpy as np
from PIL import Image
import os


class ImagePreprocessor:
    """Clase para preprocesar imágenes antes de la detección"""

    """Las funciones preprocesar_pizarra_para_layout y funcion_contraste son para etapa de detección"""

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def preprocesar_pizarra_para_layout(self, image_path, output_path=None): 
        """ 
        Preprocesa una imagen de pizarra de forma SUAVE para detección de layout.
        Usa solo escala de grises y CLAHE para corregir iluminación.
        NO binariza.
        
        Returns:
            str: Ruta de la imagen procesada guardada en disco.
        """
        # 1. Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # 2. Convertir a Escala de Grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Mejora de Contraste Suave con CLAHE (Adaptive Histogram Equalization)
        # Esto nivela el brillo sin ser destructivo.
        # clipLimit=2.0 es conservador. tileGridSize=(8,8) es estándar.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # IMPORTANTE: Los modelos de detección (LayoutParser/Detectron2) a menudo esperan
        # una imagen de 3 canales (RGB/BGR), aunque internamente usen grises.
        # Convertimos el resultado gris de vuelta a BGR para evitar errores de forma.
        final_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # NOTA: Hemos eliminado la binarización (threshold) y la inversión de colores.
        # El modelo de layout debería ser capaz de manejar la imagen en escala de grises optimizada.

        # 4. Guardar imagen
        if output_path:
            self.ensure_dir(output_path)
            cv2.imwrite(output_path, final_img)
            return output_path
        else:
            temp_path = "temp_processed.jpg"
            cv2.imwrite(temp_path, final_img)
            return temp_path

            
    @staticmethod
    def funcion_contraste(image_path):
        """
        Aplica mejora de contraste usando espacio de color YCrCb
        Esto es realce de imagen, mantiene los colores y hace el texto resaltante

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
