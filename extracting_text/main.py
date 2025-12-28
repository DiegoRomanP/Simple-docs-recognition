from ContentExtractor import ContentExtractor
from PIL import Image
import json

"""
funcion que sirve como referencia para mejorar el código
"""
def main(ruta_imagen, json_layout,extractor, output_md="nota.md"):
    # Cargar imagen y datos
    img = Image.open(ruta_imagen).convert("RGB")
    with open(json_layout, 'r') as f:
        bloques = json.load(f)
    

    # IMPORTANTE: Ordenar bloques por posición vertical (Y)
    # Para que el documento se lea de arriba a abajo
    bloques.sort(key=lambda x: x['bbox'][1])

    markdown_content = ""

    print("Iniciando transcripción...")
    for bloque in bloques:
        label = bloque['label']
        bbox = bloque['bbox']
        
        # Llamar a tu extractor modular
        texto_transcrito = extractor.procesar_region(img, bbox, label)
        
        markdown_content += texto_transcrito + "\n"
        print(f"Detectado [{label}]: {texto_transcrito[:30]}...")

    # Guardar archivo final
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"¡Nota guardada en {output_md}!")


if __name__ == "__main__":
    extractor = ContentExtractor()
    ruta_imagen = "/home/diego/Documentos/proyecto_pam/resultado_detectado.jpg"
    json_layout = "/home/diego/Documentos/proyecto_pam/coordenadas.json"
    main(ruta_imagen, json_layout, extractor)