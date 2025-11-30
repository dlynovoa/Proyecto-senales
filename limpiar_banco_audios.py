import soundfile as sf
import numpy as np
from pathlib import Path
import os
import shutil

def tiene_voz(audio, fs, umbral_rms=0.01):
    """
    Detecta si el audio tiene voz (no está vacío)
    Retorna True si tiene suficiente energía
    """
    rms = np.sqrt(np.mean(audio ** 2))
    return rms > umbral_rms


def calcular_similitud_comandos(audio, fs):
    """
    Calcula qué tan probable es que sea cada comando basado en características
    Retorna diccionario con scores para cada comando
    """
    # Características simples basadas en duración y energía
    duracion_voz = detectar_duracion_voz(audio, fs)
    energia_total = np.sum(audio ** 2)
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Patrones aproximados (ajusta según tus necesidades)
    scores = {
        'SEGMENTAR': 0,
        'COMPRIMIR': 0,
        'CIFRAR': 0
    }
    
    # SEGMENTAR: 3 sílabas, duración media-larga
    if 0.6 <= duracion_voz <= 1.2:
        scores['SEGMENTAR'] += 30
    if rms > 0.05:
        scores['SEGMENTAR'] += 20
    
    # COMPRIMIR: 3 sílabas, duración media-larga
    if 0.6 <= duracion_voz <= 1.2:
        scores['COMPRIMIR'] += 30
    if rms > 0.05:
        scores['COMPRIMIR'] += 20
    
    # CIFRAR: 2 sílabas, duración corta-media
    if 0.4 <= duracion_voz <= 0.9:
        scores['CIFRAR'] += 30
    if rms > 0.04:
        scores['CIFRAR'] += 20
    
    return scores


def detectar_duracion_voz(audio, fs, umbral_db=-30):
    """Detecta la duración de la porción con voz"""
    ventana = int(0.025 * fs)
    hop = int(0.010 * fs)
    
    energia = []
    for i in range(0, len(audio) - ventana, hop):
        chunk = audio[i:i+ventana]
        E = np.sum(chunk ** 2)
        energia.append(E)
    
    energia = np.array(energia)
    energia = np.maximum(energia, 1e-10)
    energia_db = 10 * np.log10(energia)
    
    umbral = np.max(energia_db) + umbral_db
    voz_frames = energia_db > umbral
    
    if not np.any(voz_frames):
        return 0
    
    duracion = np.sum(voz_frames) * hop / fs
    return duracion


def validar_audio(ruta, comando_esperado):
    """
    Valida un archivo de audio
    Retorna: ('ok', mensaje) o ('error', razon) o ('warning', razon)
    """
    try:
        # Leer audio
        audio, fs = sf.read(ruta)
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # 1. Verificar que no esté vacío
        if len(audio) == 0:
            return ('error', 'Audio vacío (0 muestras)')
        
        # 2. Verificar que tenga voz
        if not tiene_voz(audio, fs):
            return ('error', 'Sin voz detectada (muy silencioso)')
        
        # 3. Verificar duración mínima
        duracion = len(audio) / fs
        if duracion < 0.3:
            return ('error', f'Muy corto ({duracion:.2f}s < 0.3s)')
        
        # 4. Verificar parámetros técnicos
        if fs not in [16000, 44100, 48000]:
            return ('warning', f'Sample rate inusual: {fs} Hz')
        
        # 5. Calcular RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.005:
            return ('error', f'Volumen muy bajo (RMS={rms:.4f})')
        
        if rms > 0.5:
            return ('warning', f'Volumen muy alto (RMS={rms:.4f}, posible clipping)')
        
        # 6. Verificar similitud con comando esperado
        scores = calcular_similitud_comandos(audio, fs)
        mejor_comando = max(scores, key=scores.get)
        
        # Si el mejor no coincide, advertir
        if mejor_comando != comando_esperado:
            return ('warning', f'Parece más "{mejor_comando}" (score={scores[mejor_comando]})')
        
        return ('ok', f'RMS={rms:.3f}, Dur={duracion:.2f}s')
        
    except Exception as e:
        return ('error', f'Error al leer: {str(e)}')


def limpiar_banco_audios():
    """
    Limpia el banco de audios eliminando archivos problemáticos
    """
    print("="*70)
    print("  🧹 LIMPIEZA INTELIGENTE DEL BANCO DE AUDIOS")
    print("="*70)
    
    base_dir = Path("Datos/Banco_audios")
    
    if not base_dir.exists():
        print(f"\n❌ No existe {base_dir}")
        return
    
    comandos = ["SEGMENTAR", "COMPRIMIR", "CIFRAR"]
    
    # Crear carpeta de archivos problemáticos
    problema_dir = base_dir / "_PROBLEMATICOS"
    problema_dir.mkdir(exist_ok=True)
    
    estadisticas = {
        'ok': 0,
        'error': 0,
        'warning': 0,
        'eliminados': 0
    }
    
    archivos_problematicos = []
    
    for cmd in comandos:
        carpeta = base_dir / cmd
        
        if not carpeta.exists():
            print(f"\n⚠️  Carpeta no encontrada: {carpeta}")
            continue
        
        archivos = [f for f in carpeta.glob("*.wav") if not f.name.startswith('_')]
        
        if not archivos:
            print(f"\n⚠️  No hay archivos en {cmd}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📁 Validando {cmd}: {len(archivos)} archivos")
        print(f"{'='*70}")
        
        for i, archivo in enumerate(archivos, 1):
            estado, mensaje = validar_audio(archivo, cmd)
            
            if estado == 'ok':
                print(f"  ✅ [{i:2d}/{len(archivos)}] {archivo.name}: {mensaje}")
                estadisticas['ok'] += 1
            
            elif estado == 'warning':
                print(f"  ⚠️  [{i:2d}/{len(archivos)}] {archivo.name}: {mensaje}")
                estadisticas['warning'] += 1
                archivos_problematicos.append({
                    'archivo': archivo,
                    'comando': cmd,
                    'razon': mensaje,
                    'tipo': 'warning'
                })
            
            elif estado == 'error':
                print(f"  ❌ [{i:2d}/{len(archivos)}] {archivo.name}: {mensaje}")
                estadisticas['error'] += 1
                archivos_problematicos.append({
                    'archivo': archivo,
                    'comando': cmd,
                    'razon': mensaje,
                    'tipo': 'error'
                })
    
    # Mostrar resumen
    print(f"\n{'='*70}")
    print(f"  📊 RESUMEN DE VALIDACIÓN")
    print(f"{'='*70}")
    print(f"  ✅ Archivos OK: {estadisticas['ok']}")
    print(f"  ⚠️  Advertencias: {estadisticas['warning']}")
    print(f"  ❌ Errores: {estadisticas['error']}")
    print(f"  Total validados: {estadisticas['ok'] + estadisticas['warning'] + estadisticas['error']}")
    
    # Procesar archivos problemáticos
    if archivos_problematicos:
        print(f"\n{'='*70}")
        print(f"  🗑️  ARCHIVOS PROBLEMÁTICOS ENCONTRADOS: {len(archivos_problematicos)}")
        print(f"{'='*70}")
        
        # Mostrar detalles
        for item in archivos_problematicos:
            simbolo = "⚠️ " if item['tipo'] == 'warning' else "❌"
            print(f"  {simbolo} {item['comando']}/{item['archivo'].name}")
            print(f"      Razón: {item['razon']}")
        
        # Preguntar qué hacer
        print(f"\n{'='*70}")
        print(f"  ¿Qué deseas hacer con los archivos problemáticos?")
        print(f"{'='*70}")
        print(f"  1. Eliminar solo los que tienen ERROR (❌)")
        print(f"  2. Eliminar ERROR y WARNING (❌ + ⚠️)")
        print(f"  3. Mover a carpeta '_PROBLEMATICOS' (sin eliminar)")
        print(f"  4. No hacer nada (solo ver reporte)")
        print(f"{'='*70}")
        
        opcion = input("\nElige opción (1-4): ").strip()
        
        if opcion == '1':
            # Eliminar solo errores
            for item in archivos_problematicos:
                if item['tipo'] == 'error':
                    item['archivo'].unlink()
                    print(f"  🗑️  Eliminado: {item['comando']}/{item['archivo'].name}")
                    estadisticas['eliminados'] += 1
        
        elif opcion == '2':
            # Eliminar errores y warnings
            for item in archivos_problematicos:
                item['archivo'].unlink()
                print(f"  🗑️  Eliminado: {item['comando']}/{item['archivo'].name}")
                estadisticas['eliminados'] += 1
        
        elif opcion == '3':
            # Mover a carpeta de problemáticos
            for item in archivos_problematicos:
                destino = problema_dir / f"{item['comando']}_{item['archivo'].name}"
                shutil.move(str(item['archivo']), str(destino))
                print(f"  📦 Movido: {item['archivo'].name} → _PROBLEMATICOS/")
                estadisticas['eliminados'] += 1
        
        elif opcion == '4':
            print("\n  ℹ️  No se realizaron cambios")
        
        else:
            print("\n  ⚠️  Opción inválida. No se realizaron cambios.")
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"  ✅ LIMPIEZA COMPLETADA")
    print(f"{'='*70}")
    print(f"  Archivos procesados: {estadisticas['eliminados']}")
    print(f"  Archivos restantes OK: {estadisticas['ok']}")
    
    if estadisticas['eliminados'] > 0:
        print(f"\n  🎯 Siguiente paso:")
        print(f"     python entrenar_modelo.py")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        limpiar_banco_audios()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()