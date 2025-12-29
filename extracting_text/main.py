import os
import glob
from PIL import Image
from ContentExtractor import ContentExtractor

# Rutas relativas según tu estructura
INPUT_DIR = os.path.join(os.path.dirname(__file__), "../images_processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../markdowns")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def obtener_info_del_archivo(filename):
    """
    Analiza el nombre del archivo generado por BoundingBoxProcessor.
    Formato esperado: {base_name}_crop_{index}_{label}.jpg
    Ejemplo: imagen1_crop_1_Title.jpg
    """
    try:
        # Quitamos la extensión
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        # Buscamos dónde está la palabra 'crop'
        if 'crop' in parts:
            crop_idx = parts.index('crop')
            # El índice numérico debería ser el siguiente elemento después de 'crop'
            sort_index = int(parts[crop_idx + 1])
            # El label es lo que sigue (pueden ser varias palabras unidas por _)
            label = "_".join(parts[crop_idx + 2:])
            # El nombre base es todo lo anterior a 'crop'
            base_name = "_".join(parts[:crop_idx])
            
            return {
                "base_name": base_name,
                "sort_index": sort_index,
                "label": label,
                "full_path": os.path.join(INPUT_DIR, filename),
                "filename": filename
            }
    except Exception as e:
        print(f"Saltando archivo con formato desconocido: {filename} ({e})")
        return None

def main():
    print("="*60)
    print("GENERADOR DE MARKDOWN DESDE RECORTES")
    print("="*60)
    
    ensure_dir(OUTPUT_DIR)
    
    # 1. Buscar todos los archivos que contengan "_crop"
    patron = os.path.join(INPUT_DIR, "*_crop*")
    archivos = glob.glob(patron)
    
    if not archivos:
        print(f"❌ No se encontraron archivos de recorte en: {INPUT_DIR}")
        return

    print(f"🔍 Se encontraron {len(archivos)} recortes. Analizando...")

    # 2. Agrupar archivos por "documento original" (base_name)
    # Esto sirve por si procesaste "imagen1.jpg" e "imagen2.jpg", para que no se mezclen en un solo MD.
    documentos = {}
    
    for ruta_archivo in archivos:
        filename = os.path.basename(ruta_archivo)
        info = obtener_info_del_archivo(filename)
        
        if info:
            base_name = info['base_name']
            if base_name not in documentos:
                documentos[base_name] = []
            documentos[base_name].append(info)

    # 3. Inicializar Extractor (Carga modelos en RAM)
    extractor = ContentExtractor(use_gpu=False) # Pon True si tienes CUDA configurado

    # 4. Procesar cada documento agrupado
    for doc_name, recortes in documentos.items():
        print(f"\n📄 Procesando documento: {doc_name} ({len(recortes)} fragmentos)")
        
        # ORDENAR por el índice numérico (para mantener el orden de lectura)
        recortes.sort(key=lambda x: x['sort_index'])
        
        markdown_content = f"# Transcripción de: {doc_name}\n\n"
        
        for recorte in recortes:
            label = recorte['label']
            path = recorte['full_path']
            
            print(f"  -> Procesando [{recorte['sort_index']}]: {label} ...", end=" ", flush=True)
            
            try:
                # Abrir imagen
                img_crop = Image.open(path).convert("RGB")
                
                # Usar el nuevo método del extractor
                texto = extractor.procesar_imagen_ya_recortada(img_crop, label)
                
                markdown_content += texto + "\n"
                print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")
                markdown_content += f"\n[Error procesando imagen: {recorte['filename']}]\n"

        # 5. Guardar Markdown
        output_filename = os.path.join(OUTPUT_DIR, f"{doc_name}.md")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        print(f"✨ Guardado en: {output_filename}")

    print("\n" + "="*60)
    print("¡Proceso completado!")

if __name__ == "__main__":
    main()