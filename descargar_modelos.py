import os
import zipfile
import gdown

# --- CONFIGURACIÓN ---
# PEGA AQUÍ EL ID DE TU ARCHIVO EN GOOGLE DRIVE
DRIVE_FILE_ID = "1DD-D1SEPH2tG8j_QmwB3N_kqhDbwNetn"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ZIP_NAME = "models.zip"


def setup():
    print("=" * 60)
    print("📥 DESCARGANDO MODELOS DESDE GOOGLE DRIVE")
    print("=" * 60)

    # 1. Descargar el ZIP usando gdown (Maneja la seguridad de Google Drive automáticamente)
    output_zip = os.path.join(BASE_DIR, ZIP_NAME)

    # URL de descarga directa
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

    print(f"Descargando {ZIP_NAME}...")
    try:
        gdown.download(url, output_zip, quiet=False)
    except Exception as e:
        print(f"❌ Error en la descarga: {e}")
        return

    # 2. Descomprimir
    print("\n📦 Descomprimiendo modelos...")

    # Crear carpeta models si no existe (o limpiarla)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    try:
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(
                BASE_DIR
            )  # Se asume que el zip contiene la carpeta 'models' dentro
        print(f"✅ Modelos extraídos en: {MODELS_DIR}")
    except zipfile.BadZipFile:
        print("❌ Error: El archivo descargado no es un ZIP válido.")
        return

    # 3. Limpieza
    if os.path.exists(output_zip):
        os.remove(output_zip)
        print("🧹 Archivo temporal eliminado.")

    print("\n✨ ¡Configuración de modelos completada!")


if __name__ == "__main__":
    setup()
