import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, resample
import shutil

TARGET_FS = 16000

def filtro_pasabajos(audio, fs, fc=3500, orden=4):
    """Filtro pasabajos para eliminar ruido de alta frecuencia"""
    b, a = butter(orden, fc / (fs / 2), btype='low')
    return filtfilt(b, a, audio)


def filtro_pasaaltos(audio, fs, fc=80, orden=2):
    """Filtro pasaaltos para eliminar ruido de baja frecuencia (rumble)"""
    b, a = butter(orden, fc / (fs / 2), btype='high')
    return filtfilt(b, a, audio)


def reduccion_ruido_espectral(audio, umbral_percentil=15):
    """
    Reducción de ruido por sustracción espectral simple
    Asume que el ruido está en las frecuencias más débiles
    """
    # FFT
    espectro = np.fft.rfft(audio)
    magnitud = np.abs(espectro)
    fase = np.angle(espectro)
    
    # Estimar umbral de ruido (percentil bajo)
    umbral_ruido = np.percentile(magnitud, umbral_percentil)
    
    # Atenuar componentes debajo del umbral (reducción suave)
    factor = np.where(magnitud < umbral_ruido, 
                      magnitud / (umbral_ruido + 1e-6), 
                      1.0)
    
    magnitud_limpia = magnitud * factor
    
    # Reconstruir señal
    espectro_limpio = magnitud_limpia * np.exp(1j * fase)
    audio_limpio = np.fft.irfft(espectro_limpio, n=len(audio))
    
    return audio_limpio


def gate_ruido(audio, umbral_db=-40, ventana_ms=20, fs=16000):
    """
    Noise gate: silencia partes donde la energía está por debajo del umbral
    """
    ventana_muestras = int(ventana_ms * fs / 1000)
    hop = ventana_muestras // 2
    
    # Calcular energía por ventanas
    energia = []
    for i in range(0, len(audio) - ventana_muestras, hop):
        chunk = audio[i:i+ventana_muestras]
        E = np.sum(chunk ** 2)
        energia.append(E)
    
    energia = np.array(energia)
    energia = np.maximum(energia, 1e-10)
    energia_db = 10 * np.log10(energia)
    
    # Crear máscara (1 = mantener, valor < 1 = atenuar)
    mascara = np.where(energia_db > umbral_db, 1.0, 0.05)  # Atenuar a 5% en lugar de silenciar
    
    # Interpolar máscara para que tenga el tamaño del audio
    mascara_interp = np.interp(
        np.arange(len(audio)),
        np.arange(len(mascara)) * hop + ventana_muestras // 2,
        mascara
    )
    
    # Aplicar máscara con suavizado
    audio_gated = audio * mascara_interp
    
    return audio_gated


def normalizar_audio(audio, rms_objetivo=0.08):
    """Normaliza el audio a un RMS específico"""
    rms_actual = np.sqrt(np.mean(audio ** 2))
    
    if rms_actual > 1e-6:
        factor = rms_objetivo / rms_actual
        # Limitar el factor para evitar amplificación excesiva
        factor = min(factor, 5.0)
        audio = audio * factor
    
    # Limitar amplitud para evitar clipping
    audio = np.clip(audio, -0.95, 0.95)
    
    return audio


def limpiar_ruido_completo(audio, fs):
    """
    Pipeline completo de limpieza de ruido
    """
    print(f"    🧹 Aplicando limpieza de ruido...")
    
    # 1. Filtro pasaaltos (eliminar rumble)
    audio = filtro_pasaaltos(audio, fs, fc=80)
    print(f"       ✓ Filtro pasaaltos (elimina rumble)")
    
    # 2. Filtro pasabajos (eliminar ruido agudo)
    audio = filtro_pasabajos(audio, fs, fc=3500)
    print(f"       ✓ Filtro pasabajos (elimina agudos)")
    
    # 3. Reducción espectral de ruido
    audio = reduccion_ruido_espectral(audio, umbral_percentil=20)
    print(f"       ✓ Reducción espectral")
    
    # 4. Noise gate
    audio = gate_ruido(audio, umbral_db=-35, fs=fs)
    print(f"       ✓ Noise gate")
    
    # 5. Normalizar
    audio = normalizar_audio(audio, rms_objetivo=0.08)
    print(f"       ✓ Normalización RMS")
    
    return audio


def procesar_archivo_problematico(ruta_entrada, comando_destino):
    """
    Procesa un archivo de _PROBLEMATICOS, lo limpia y lo mueve a su carpeta
    """
    try:
        # Leer audio
        audio, fs_original = sf.read(ruta_entrada)
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        print(f"    📄 Original: {fs_original} Hz, {len(audio)/fs_original:.2f}s")
        
        # Remuestrear si es necesario
        if fs_original != TARGET_FS:
            nuevas_muestras = int(len(audio) * TARGET_FS / fs_original)
            audio = resample(audio, nuevas_muestras)
            print(f"    🔄 Remuestreado a {TARGET_FS} Hz")
        
        # Limpiar ruido
        audio_limpio = limpiar_ruido_completo(audio, TARGET_FS)
        
        # Ajustar a 1 segundo (tomar ventana de máxima energía)
        muestras_objetivo = TARGET_FS
        
        if len(audio_limpio) > muestras_objetivo:
            mejor_energia = -1
            mejor_inicio = 0
            paso = TARGET_FS // 10
            
            for i in range(0, len(audio_limpio) - muestras_objetivo, paso):
                ventana = audio_limpio[i:i + muestras_objetivo]
                energia = np.sum(ventana ** 2)
                
                if energia > mejor_energia:
                    mejor_energia = energia
                    mejor_inicio = i
            
            audio_final = audio_limpio[mejor_inicio:mejor_inicio + muestras_objetivo]
            print(f"    ✂️  Recortado a 1s (ventana de máxima energía)")
        
        elif len(audio_limpio) < muestras_objetivo:
            padding = muestras_objetivo - len(audio_limpio)
            pad_antes = padding // 2
            pad_despues = padding - pad_antes
            audio_final = np.pad(audio_limpio, (pad_antes, pad_despues))
            print(f"    ➕ Padding a 1s")
        else:
            audio_final = audio_limpio
        
        # Guardar en carpeta de destino
        carpeta_destino = Path("Datos/Banco_audios") / comando_destino
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        
        nombre_limpio = ruta_entrada.name.replace(f"{comando_destino}_", "")
        ruta_salida = carpeta_destino / nombre_limpio
        
        # Si ya existe, agregar número
        contador = 1
        while ruta_salida.exists():
            nombre_base = ruta_salida.stem
            ruta_salida = carpeta_destino / f"{nombre_base}_{contador}.wav"
            contador += 1
        
        sf.write(ruta_salida, audio_final, TARGET_FS)
        print(f"    💾 Guardado: {comando_destino}/{ruta_salida.name}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def main():
    print("="*70)
    print("  🧼 LIMPIEZA DE AUDIOS PROBLEMÁTICOS")
    print("  Remueve ruido y reubica archivos a sus carpetas")
    print("="*70)
    
    problema_dir = Path("Datos/Banco_audios/_PROBLEMATICOS")
    
    if not problema_dir.exists():
        print(f"\n⚠️  No existe la carpeta {problema_dir}")
        print(f"   Ejecuta primero: python limpiar_banco_audios.py")
        return
    
    archivos = list(problema_dir.glob("*.wav"))
    
    if not archivos:
        print(f"\n✅ No hay archivos en _PROBLEMATICOS")
        return
    
    print(f"\n📁 Archivos encontrados: {len(archivos)}")
    print(f"\nEstos archivos serán:")
    print(f"  1. Limpiados de ruido (filtros + reducción espectral)")
    print(f"  2. Ajustados a 16kHz, 1 segundo")
    print(f"  3. Movidos a sus carpetas correspondientes")
    
    # Agrupar por comando
    por_comando = {}
    for archivo in archivos:
        # Extraer comando del nombre (formato: COMANDO_archivo.wav)
        nombre = archivo.name
        
        if nombre.startswith("SEGMENTAR_"):
            comando = "SEGMENTAR"
        elif nombre.startswith("COMPRIMIR_"):
            comando = "COMPRIMIR"
        elif nombre.startswith("CIFRAR_"):
            comando = "CIFRAR"
        else:
            # Preguntar manualmente
            print(f"\n⚠️  No se pudo detectar comando de: {nombre}")
            comando = input(f"   ¿A qué carpeta pertenece? (SEGMENTAR/COMPRIMIR/CIFRAR): ").upper()
            
            if comando not in ["SEGMENTAR", "COMPRIMIR", "CIFRAR"]:
                print(f"   ⚠️  Saltando archivo...")
                continue
        
        if comando not in por_comando:
            por_comando[comando] = []
        por_comando[comando].append(archivo)
    
    # Mostrar resumen
    print(f"\n📊 Resumen:")
    for cmd, files in por_comando.items():
        print(f"  {cmd}: {len(files)} archivos")
    
    respuesta = input(f"\n¿Proceder con la limpieza? (s/n): ")
    
    if respuesta.lower() != 's':
        print("\n❌ Cancelado")
        return
    
    # Procesar archivos
    procesados = 0
    errores = 0
    
    for comando, files in por_comando.items():
        print(f"\n{'='*70}")
        print(f"🧹 Procesando {comando}: {len(files)} archivos")
        print(f"{'='*70}")
        
        for i, archivo in enumerate(files, 1):
            print(f"\n  [{i}/{len(files)}] {archivo.name}")
            print(f"  {'-'*66}")
            
            if procesar_archivo_problematico(archivo, comando):
                # Eliminar de _PROBLEMATICOS
                archivo.unlink()
                print(f"    🗑️  Eliminado de _PROBLEMATICOS")
                procesados += 1
            else:
                errores += 1
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"  ✅ LIMPIEZA COMPLETADA")
    print(f"{'='*70}")
    print(f"  Procesados: {procesados}")
    print(f"  Errores: {errores}")
    
    if procesados > 0:
        print(f"\n  🎯 Siguiente paso:")
        print(f"     python diagnosticar_audios.py")
        print(f"     python entrenar_modelo.py")
    
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