import sounddevice as sd
import soundfile as sf
import os
import time

# --- CONFIGURACIÓN ---
FS = 16000  # 16kHz como el compañero
DURATION = 1.0  # 1 segundo como el compañero
BASE_DIR = os.path.join("Datos", "Banco_audios")
COMANDOS = ["SEGMENTAR", "COMPRIMIR", "CIFRAR"]
MUESTRAS_POR_PALABRA = 5  # Mínimo 5 para mejor entrenamiento

def asegurar_directorios():
    """Crea las carpetas si no existen"""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    
    for cmd in COMANDOS:
        path = os.path.join(BASE_DIR, cmd)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"📁 Carpeta creada: {path}")

def grabar_audio(filename):
    print(f"🔴 Grabando (Habla ahora)...")
    
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    
    print("✅ Grabación finalizada.")
    
    sf.write(filename, recording.flatten(), FS)
    print(f"💾 Guardado en: {filename}")

def main():
    print("="*50)
    print("🎙️  GRABADORA DE BANCO DE VOZ")
    print("="*50)
    print(f"Configuración: {DURATION} segundos, {FS} Hz")
    print(f"Se grabarán {MUESTRAS_POR_PALABRA} muestras por cada comando.")
    
    asegurar_directorios()
    
    for cmd in COMANDOS:
        print("\n" + "-"*40)
        print(f"📢 PALABRA A GRABAR: '{cmd}'")
        print("-"*40)
        
        carpeta_cmd = os.path.join(BASE_DIR, cmd)
        files = [f for f in os.listdir(carpeta_cmd) if f.endswith(".wav")]
        
        if files:
            print(f"⚠️  Nota: Ya existen {len(files)} audios en esta carpeta.")
            opcion = input("¿Quieres borrarlos y empezar de cero? (s/n): ")
            if opcion.lower() == 's':
                for f in files:
                    os.remove(os.path.join(carpeta_cmd, f))
                print("🗑️  Audios viejos eliminados.")

        for i in range(1, MUESTRAS_POR_PALABRA + 1):
            nombre_archivo = os.path.join(carpeta_cmd, f"{cmd.lower()}_{i}.wav")
            
            print(f"\nPreparando muestra {i}/{MUESTRAS_POR_PALABRA}...")
            input(f"👉 Presiona ENTER, respira y di '{cmd}'...")
            
            print("3...", end=" ", flush=True)
            time.sleep(0.5)
            print("2...", end=" ", flush=True)
            time.sleep(0.5)
            print("1...", end=" ", flush=True)
            time.sleep(0.5)
            
            grabar_audio(nombre_archivo)
            time.sleep(0.5)
            
    print("\n" + "="*50)
    print("🎉 ¡PROCESO TERMINADO!")
    print("Ahora ejecuta 'python entrenar_modelo.py' para generar el modelo")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")