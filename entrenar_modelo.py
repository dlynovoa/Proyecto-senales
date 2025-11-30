from Logica.Reconocimiento_voz import VoiceEngine
import json

def main():
    print("="*60)
    print("  🎓 ENTRENAMIENTO DEL MODELO DE VOZ")
    print("="*60)
    
    # Crear instancia del motor
    engine = VoiceEngine()
    
    # Entrenar desde las carpetas de audio
    print("\n📚 Entrenando desde carpetas de audio...")
    engine.entrenar_desde_carpetas()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    
    print("\n📊 Resumen del modelo:")
    for cmd, datos in engine.templates.items():
        print(f"  • {cmd}: {datos['count']} ejemplos")
    
    print(f"\n💾 Modelo guardado en: {engine.json_path}")
    
    # Previsualizar JSON
    print("\n📄 Estructura del modelo:")
    with open(engine.json_path, 'r') as f:
        data = json.load(f)
        
        print(f"\n📐 Configuración:")
        config = data.get("config", {})
        print(f"  - Sample Rate: {config.get('fs')} Hz")
        print(f"  - FFT Size: {config.get('N')} puntos")
        print(f"  - Subbandas: {config.get('K')}")
        print(f"  - Ventana: {config.get('window')}")
        
        print(f"\n🎯 Comandos entrenados:")
        for cmd, datos in data.get("commands", {}).items():
            print(f"\n  {cmd}:")
            print(f"    - Muestras: {datos.get('count')}")
            print(f"    - Vector medio: {len(datos.get('mean', []))} features")
            print(f"    - Primeras 4 energías: {datos.get('mean', [])[:4]}")
    
    print("\n" + "="*60)
    print("  🎯 LISTO PARA USAR")
    print("  Ejecuta: streamlit run main_streamlit.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()