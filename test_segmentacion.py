"""
Test rápido para probar la segmentación de imágenes con IA
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Logica.Segmentacion_imagen import segmentar_con_ia, generar_visualizacion_ia

def test_segmentacion_rapido():
    """Test rápido del sistema de segmentación con IA"""
    
    print("="*60)
    print("TEST DE SEGMENTACIÓN DE IMÁGENES")
    print("="*60)
    
    # Cargar imagen de prueba
    ruta_imagen = "Recursos/Prueba1.jpg"
    print(f"\n1. Cargando imagen: {ruta_imagen}")
    
    try:
        imagen = Image.open(ruta_imagen)
        print(f"   ✓ Imagen cargada: {imagen.size} - Modo: {imagen.mode}")
    except Exception as e:
        print(f"   ✗ Error al cargar imagen: {e}")
        return
    
    print(f"\n2. Método de segmentación:")
    print(f"   - Modelo: U²-Net (Deep Learning)")
    print(f"   - Técnica: Eliminación automática de fondo")
    print(f"   - Librería: rembg")
    
    # Segmentar imagen
    print(f"\n3. Segmentando imagen...")
    try:
        resultado = segmentar_con_ia(imagen)
        print(f"   ✓ Segmentación completada")
    except Exception as e:
        print(f"   ✗ Error durante segmentación: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calcular estadísticas de segmentación
    print(f"\n4. Calculando estadísticas...")
    
    mascara = resultado['mascara']
    total_pixels = mascara.shape[0] * mascara.shape[1]
    pixels_objeto = np.sum(mascara > 0)
    pixels_fondo = total_pixels - pixels_objeto
    porcentaje_objeto = (pixels_objeto / total_pixels) * 100
    porcentaje_fondo = 100 - porcentaje_objeto
    
    print(f"   - Total de píxeles: {total_pixels:,}")
    print(f"   - Píxeles del objeto: {pixels_objeto:,} ({porcentaje_objeto:.2f}%)")
    print(f"   - Píxeles del fondo: {pixels_fondo:,} ({porcentaje_fondo:.2f}%)")
    
    # Evaluación
    print(f"\n5. Evaluación:")
    if porcentaje_objeto > 5 and porcentaje_objeto < 95:
        print(f"   ✓ Segmentación exitosa (objeto detectado: {porcentaje_objeto:.1f}%)")
    elif porcentaje_objeto >= 95:
        print(f"   ⚠ Advertencia: Casi no se detectó fondo ({porcentaje_fondo:.1f}%)")
    else:
        print(f"   ⚠ Advertencia: Objeto muy pequeño ({porcentaje_objeto:.1f}%)")
    
    # Visualización
    print(f"\n6. Generando visualización...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Test de Segmentación con IA (U²-Net)', fontsize=16, fontweight='bold')
        
        # Fila 1: Original y Máscara
        axes[0, 0].imshow(resultado['original'])
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(resultado['mascara'], cmap='gray')
        axes[0, 1].set_title(f'Máscara Binaria\n{porcentaje_objeto:.1f}% objeto')
        axes[0, 1].axis('off')
        
        # Fila 2: Objeto segmentado y comparación
        axes[1, 0].imshow(resultado['segmentada'])
        axes[1, 0].set_title('Objeto Segmentado\n(Sin fondo)')
        axes[1, 0].axis('off')
        
        # Mostrar overlay
        overlay = resultado['original'].copy()
        overlay_array = np.array(overlay)
        mascara_3d = np.stack([resultado['mascara']] * 3, axis=-1) > 0
        overlay_array[~mascara_3d] = overlay_array[~mascara_3d] // 2  # Atenuar fondo
        axes[1, 1].imshow(overlay_array)
        axes[1, 1].set_title('Overlay\n(Fondo atenuado)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        output_file = 'test_segmentacion_resultado.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualización guardada: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"   ⚠ No se pudo generar visualización: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    test_segmentacion_rapido()
