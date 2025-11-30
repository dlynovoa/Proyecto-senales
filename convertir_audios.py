import soundfile as sf
import numpy as np
from scipy.signal import resample
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Parámetros objetivo
TARGET_FS = 16000
TARGET_DURATION = 1.0

def detectar_voz_vad(audio, fs, umbral_db=-30, margen_ms=100):
    """
    Detección de actividad de voz (VAD) avanzada
    Retorna inicio y fin de la porción con voz
    """
    # Calcular energía por ventana
    ventana_muestras = int(0.025 * fs)  # 25ms
    hop = int(0.010 * fs)  # 10ms
    
    energia = []
    posiciones = []
    
    for i in range(0, len(audio) - ventana_muestras, hop):
        chunk = audio[i:i+ventana_muestras]
        E = np.sum(chunk ** 2)
        energia.append(E)
        posiciones.append(i)
    
    energia = np.array(energia)
    
    # Evitar log(0)
    energia = np.maximum(energia, 1e-10)
    
    # Convertir a dB
    energia_db = 10 * np.log10(energia)
    
    # Umbral adaptativo: máximo - umbral_db
    umbral = np.max(energia_db) + umbral_db
    
    # Detectar frames con voz
    voz_frames = energia_db > umbral
    
    if not np.any(voz_frames):
        # Si no detecta nada, usar todo el audio
        return 0, len(audio)
    
    # Encontrar primer y último frame con voz
    indices_voz = np.where(voz_frames)[0]
    primer_frame = indices_voz[0]
    ultimo_frame = indices_voz[-1]
    
    # Convertir a muestras
    margen_muestras = int(margen_ms * fs / 1000)
    
    inicio = max(0, posiciones[primer_frame] - margen_muestras)
    fin = min(len(audio), posiciones[ultimo_frame] + ventana_muestras + margen_muestras)
    
    return inicio, fin


def normalizar_rms(audio, rms_objetivo=0.1):
    """Normaliza el audio a un RMS específico"""
    rms_actual = np.sqrt(np.mean(audio ** 2))
    
    if rms_actual > 1e-6:
        factor = rms_objetivo / rms_actual
        audio = audio * factor
    
    return audio


def convertir_audio_inteligente(ruta_entrada, ruta_salida, visualizar=False):
    """
    Convierte audio a 16kHz, 1 segundo usando VAD inteligente
    """
    try:
        # 1. Leer audio original
        audio, fs_original = sf.read(ruta_entrada)
        
        # Convertir a mono si es estéreo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        duracion_original = len(audio) / fs_original
        
        print(f"  📄 Original: {fs_original} Hz, {duracion_original:.2f}s, {len(audio)} muestras")
        
        # 2. Remuestrear a 16kHz si es necesario
        if fs_original != TARGET_FS:
            nuevas_muestras = int(len(audio) * TARGET_FS / fs_original)
            audio = resample(audio, nuevas_muestras)
            print(f"  🔄 Remuestreado: {TARGET_FS} Hz, {len(audio)} muestras")
        
        # 3. Detectar porción con voz usando VAD
        inicio, fin = detectar_voz_vad(audio, TARGET_FS)
        audio_voz = audio[inicio:fin]
        
        duracion_voz = len(audio_voz) / TARGET_FS
        print(f"  🎤 Voz detectada: {duracion_voz:.2f}s ({inicio}-{fin} muestras)")
        
        # 4. Ajustar a exactamente 1 segundo
        muestras_objetivo = int(TARGET_DURATION * TARGET_FS)
        
        if len(audio_voz) > muestras_objetivo:
            # Si es más largo, tomar ventana con máxima energía
            mejor_energia = -1
            mejor_inicio = 0
            paso = TARGET_FS // 10  # Paso de 100ms
            
            for i in range(0, len(audio_voz) - muestras_objetivo, paso):
                ventana = audio_voz[i:i + muestras_objetivo]
                energia = np.sum(ventana ** 2)
                
                if energia > mejor_energia:
                    mejor_energia = energia
                    mejor_inicio = i
            
            audio_final = audio_voz[mejor_inicio:mejor_inicio + muestras_objetivo]
            print(f"  ✂️  Recortado: Ventana de máxima energía (inicio={mejor_inicio})")
            
        elif len(audio_voz) < muestras_objetivo:
            # Si es más corto, centrar y aplicar padding
            padding_total = muestras_objetivo - len(audio_voz)
            pad_antes = padding_total // 2
            pad_despues = padding_total - pad_antes
            
            audio_final = np.pad(audio_voz, (pad_antes, pad_despues), mode='constant')
            print(f"  ➕ Padding: {pad_antes} antes, {pad_despues} después")
        else:
            audio_final = audio_voz
            print(f"  ✓ Duración perfecta")
        
        # 5. Normalizar RMS
        audio_final = normalizar_rms(audio_final, rms_objetivo=0.1)
        print(f"  🔊 Normalizado: RMS = 0.1")
        
        # 6. Visualizar si se solicita
        if visualizar:
            plt.figure(figsize=(12, 6))
            
            # Audio original
            plt.subplot(2, 1, 1)
            tiempo_orig = np.arange(len(audio)) / TARGET_FS
            plt.plot(tiempo_orig, audio, alpha=0.7)
            plt.axvline(inicio / TARGET_FS, color='green', linestyle='--', label='Inicio voz')
            plt.axvline(fin / TARGET_FS, color='red', linestyle='--', label='Fin voz')
            plt.title(f'Audio Original Remuestreado ({len(audio)} muestras)')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Audio final
            plt.subplot(2, 1, 2)
            tiempo_final = np.arange(len(audio_final)) / TARGET_FS
            plt.plot(tiempo_final, audio_final, color='orange')
            plt.title(f'Audio Final ({len(audio_final)} muestras = {TARGET_DURATION}s @ {TARGET_FS}Hz)')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar visualización (convertir Path a str)
            nombre_viz = str(ruta_salida).replace('.wav', '_viz.png')
            plt.savefig(nombre_viz, dpi=100)
            plt.close()
            print(f"  📊 Visualización: {Path(nombre_viz).name}")
        
        # 7. Guardar
        sf.write(ruta_salida, audio_final, TARGET_FS)
        print(f"  💾 Guardado: {ruta_salida}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("  🔄 CONVERSOR INTELIGENTE DE AUDIOS CON VAD")
    print("  Detecta voz automáticamente y ajusta a 16kHz, 1 segundo")
    print("="*70)
    
    base_dir = Path("Datos/Banco_audios")
    
    if not base_dir.exists():
        print(f"\n❌ No existe la carpeta {base_dir}")
        return
    
    comandos = ["SEGMENTAR", "COMPRIMIR", "CIFRAR"]
    total_convertidos = 0
    
    # Preguntar si quiere visualizaciones
    respuesta = input("\n¿Generar visualizaciones de cada audio? (s/n): ")
    visualizar = respuesta.lower() == 's'
    
    for cmd in comandos:
        carpeta = base_dir / cmd
        
        if not carpeta.exists():
            print(f"\n⚠️  Carpeta no encontrada: {carpeta}")
            continue
        
        archivos = list(carpeta.glob("*.wav"))
        
        if not archivos:
            print(f"\n⚠️  No hay archivos WAV en {cmd}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📁 Procesando {cmd}: {len(archivos)} archivos")
        print(f"{'='*70}")
        
        for i, archivo in enumerate(archivos, 1):
            print(f"\n[{i}/{len(archivos)}] {archivo.name}")
            print(f"  {'-'*66}")
            
            # Crear backup
            backup_dir = carpeta / "backup_originales"
            backup_dir.mkdir(exist_ok=True)
            
            ruta_backup = backup_dir / archivo.name
            
            # Mover original a backup (solo si no existe)
            if not ruta_backup.exists():
                import shutil
                shutil.copy2(archivo, ruta_backup)
                print(f"  💾 Backup: backup_originales/{archivo.name}")
            
            # Convertir
            if convertir_audio_inteligente(archivo, archivo, visualizar=visualizar):
                total_convertidos += 1
                print(f"  ✅ Conversión exitosa")
    
    print(f"\n{'='*70}")
    print(f"  ✅ CONVERSIÓN COMPLETADA")
    print(f"  Total archivos convertidos: {total_convertidos}")
    print(f"{'='*70}")
    print(f"\n📌 Los archivos originales están en:")
    print(f"   Datos/Banco_audios/*/backup_originales/")
    
    if visualizar:
        print(f"\n📊 Las visualizaciones están junto a cada audio (*_viz.png)")
    
    print(f"\n🎯 Ahora ejecuta:")
    print(f"   python entrenar_modelo.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()