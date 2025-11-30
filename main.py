import sys
import os

# Importar módulos
import Logica.Reconocimiento_voz as Reconocimiento_voz
import Logica.Segmentacion_imagen as Segmentacion_imagen
import Logica.Compresion_imagen as Compresion_imagen
import Logica.Cifrado_imagen as Cifrado_imagen

def mostrar_menu():
    """Muestra el menú principal"""
    print("\n" + "="*60)
    print("  🎤 SISTEMA DE PROCESAMIENTO CON VOZ")
    print("="*60)
    print("\n1. 🎤 Sistema con Reconocimiento de Voz (Completo)")
    print("2. 🎯 Entrenar Reconocimiento de Voz")
    print("3. 🔍 Probar Segmentación (sin voz)")
    print("4. 🗜️  Probar Compresión DCT (sin voz)")
    print("5. 🔐 Probar Cifrado FrDCT+DOST (sin voz)")
    print("6. 🚪 Salir")
    print("="*60)
    
    return input("\nSelecciona opción (1-6): ")


def sistema_completo_con_voz():
    """Modo completo: reconoce comando y ejecuta la operación"""
    print("\n" + "="*60)
    print("  🎤 MODO COMPLETO CON VOZ")
    print("="*60)
    
    # Obtener imagen
    ruta_imagen = input("\nRuta de la imagen: ")
    if not os.path.exists(ruta_imagen):
        print("❌ Imagen no encontrada")
        return
    
    # Inicializar reconocedor
    print("\n🎤 Inicializando sistema de reconocimiento...")
    reconocedor = Reconocimiento_voz.VoiceEngine()
    
    # Entrenar rápido (o cargar modelo)
    print("\n⚠️  ENTRENAMIENTO RÁPIDO")
    print("Vamos a grabar 2 muestras de cada comando:\n")
    
    nombres = {1: "SEGMENTAR", 2: "COMPRIMIR", 3: "CIFRAR"}
    
    for palabra_id in [1, 2, 3]:
        nombre = nombres[palabra_id]
        print(f"\n--- {nombre} ---")
        for i in range(2):
            input(f"Presiona ENTER y di '{nombre}' ({i+1}/2)...")
            audio = reconocedor.grabar_audio()
            reconocedor.agregar_muestra(palabra_id, audio)
    
    if not reconocedor.entrenar():
        print("❌ Error en entrenamiento")
        return
    
    # Loop de reconocimiento
    print("\n" + "="*60)
    print("  🎤 LISTO - Di tu comando")
    print("="*60)
    print("\nPresiona Ctrl+C para salir\n")
    
    try:
        while True:
            input("🎤 Presiona ENTER para grabar comando...")
            audio = reconocedor.grabar_audio()
            palabra_id, distancia, nombre = reconocedor.reconocer(audio)
            
            print(f"\n✨ Comando reconocido: {nombre}\n")
            
            # Ejecutar comando correspondiente
            if nombre == "SEGMENTAR":
                print("🔍 Ejecutando SEGMENTACIÓN...")
                Segmentacion_imagen.segmentar_imagen(ruta_imagen)
            
            elif nombre == "COMPRIMIR":
                print("🗜️  Ejecutando COMPRESIÓN...")
                Compresion_imagen.comprimir_imagen(ruta_imagen)
            
            elif nombre == "CIFRAR":
                print("🔐 Ejecutando CIFRADO...")
                Cifrado_imagen.cifrar_imagen(ruta_imagen)
            
            print("\n" + "-"*60)
    
    except KeyboardInterrupt:
        print("\n\n👋 Saliendo...")


def entrenar_reconocimiento():
    """Modo de entrenamiento del reconocedor"""
    print("\n" + "="*60)
    print("  🎓 ENTRENAMIENTO DE RECONOCIMIENTO DE VOZ")
    print("="*60)
    
    reconocedor = Reconocimiento_voz.VoiceEngine()
    reconocedor.modo_entrenamiento_interactivo()
    
    # Probar reconocimiento
    reconocedor.modo_reconocimiento_interactivo()


def probar_segmentacion():
    """Prueba directa de segmentación"""
    ruta = input("\nRuta de la imagen: ")
    if os.path.exists(ruta):
        Segmentacion_imagen.segmentar_imagen(ruta)
    else:
        print("❌ Imagen no encontrada")


def probar_compresion():
    """Prueba directa de compresión"""
    ruta = input("\nRuta de la imagen: ")
    if os.path.exists(ruta):
        Compresion_imagen.comprimir_imagen(ruta)
    else:
        print("❌ Imagen no encontrada")


def probar_cifrado():
    """Prueba directa de cifrado"""
    ruta = input("\nRuta de la imagen: ")
    if os.path.exists(ruta):
        Cifrado_imagen.cifrar_imagen(ruta)
    else:
        print("❌ Imagen no encontrada")


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("  🎤 PROYECTO FINAL - PROCESAMIENTO DE SEÑALES")
    print("="*60)
    print("\nMódulos:")
    print("  • Reconocimiento de Voz (FFT + Energía por Bandas)")
    print("  • Segmentación de Imágenes (Canny + Contornos)")
    print("  • Compresión DCT (3 niveles)")
    print("  • Cifrado FrDCT + DOST (Algoritmo Académico)")
    
    while True:
        opcion = mostrar_menu()
        
        if opcion == "1":
            sistema_completo_con_voz()
        
        elif opcion == "2":
            entrenar_reconocimiento()
        
        elif opcion == "3":
            probar_segmentacion()
        
        elif opcion == "4":
            probar_compresion()
        
        elif opcion == "5":
            probar_cifrado()
        
        elif opcion == "6":
            print("\n👋 ¡Hasta luego!\n")
            break
        
        else:
            print("\n❌ Opción inválida")
        
        input("\nPresiona ENTER para continuar...")


if __name__ == "__main__":
    main()