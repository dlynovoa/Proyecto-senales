"""
Test rápido para probar la compresión de imágenes con DCT
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Logica.Compresion_imagen import comprimir_imagen_dct

def test_compresion_rapido():
    """Test rápido del sistema de compresión DCT"""
    
    print("="*60)
    print("TEST DE COMPRESIÓN DE IMÁGENES")
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
    
    # Definir niveles de compresión
    porcentajes = [30.0, 50.0, 95.0]
    
    print(f"\n2. Parámetros de compresión:")
    print(f"   - Algoritmo: DCT 2D por bloques")
    print(f"   - Tamaño de bloque: 8x8 píxeles")
    print(f"   - Canales: RGB (3 canales)")
    print(f"   - Niveles a probar: {porcentajes}")
    
    # Comprimir en los 3 niveles
    print(f"\n3. Comprimiendo imagen en {len(porcentajes)} niveles...")
    resultados = []
    
    for pct in porcentajes:
        print(f"\n   Compresión {pct}%...")
        try:
            resultado = comprimir_imagen_dct(imagen, pct)
            resultados.append(resultado)
            metricas = resultado['metricas']
            print(f"   ✓ Nivel {pct}% completado")
            print(f"      - PSNR: {metricas['psnr']:.2f} dB")
            print(f"      - MSE: {metricas['mse']:.2f}")
            print(f"      - Coefs eliminados: {metricas['coefs_eliminados']:,}")
        except Exception as e:
            print(f"   ✗ Error en compresión {pct}%: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Evaluación de calidad
    print(f"\n4. Evaluación de calidad:")
    for i, (pct, resultado) in enumerate(zip(porcentajes, resultados)):
        psnr = resultado['metricas']['psnr']
        if psnr > 35:
            calidad = "Excelente"
        elif psnr > 30:
            calidad = "Muy buena"
        elif psnr > 25:
            calidad = "Buena"
        elif psnr > 20:
            calidad = "Aceptable"
        else:
            calidad = "Baja"
        print(f"   {pct}%: {calidad} (PSNR: {psnr:.2f} dB)")
    
    # Visualización
    print(f"\n5. Generando visualización...")
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Test de Compresión DCT - Comparación de Niveles', fontsize=16, fontweight='bold')
        
        # Fila 1: Imágenes
        # Original
        axes[0, 0].imshow(imagen)
        axes[0, 0].set_title('Original\nSin compresión')
        axes[0, 0].axis('off')
        
        # 3 niveles de compresión
        for i, (resultado, pct) in enumerate(zip(resultados, porcentajes)):
            axes[0, i+1].imshow(resultado['comprimida'])
            axes[0, i+1].set_title(f'Compresión {pct}%\nPSNR: {resultado["metricas"]["psnr"]:.2f} dB')
            axes[0, i+1].axis('off')
        
        # Fila 2: Mapas de diferencia
        img_original_array = np.array(imagen)
        
        axes[1, 0].imshow(img_original_array)
        axes[1, 0].set_title('Original')
        axes[1, 0].axis('off')
        
        for i, (resultado, pct) in enumerate(zip(resultados, porcentajes)):
            # Calcular diferencia
            diferencia = np.abs(img_original_array.astype(float) - resultado['comprimida'].astype(float))
            diferencia_visual = diferencia.mean(axis=2)  # Promedio de canales RGB
            
            im = axes[1, i+1].imshow(diferencia_visual, cmap='hot', vmin=0, vmax=50)
            axes[1, i+1].set_title(f'Diferencia {pct}%\nMSE: {resultado["metricas"]["mse"]:.2f}')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        output_file = 'test_compresion_resultado.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualización guardada: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"   ⚠ No se pudo generar visualización: {e}")
        import traceback
        traceback.print_exc()
    
    # Tabla de métricas
    print(f"\n6. Tabla comparativa de métricas:")
    print(f"   {'Nivel':<12} {'PSNR (dB)':<12} {'MSE':<12} {'Coefs. Elim.':<15} {'Tasa Comp.':<12}")
    print(f"   {'-'*70}")
    for pct, resultado in zip(porcentajes, resultados):
        metricas = resultado['metricas']
        print(f"   {pct}%{'':<9} {metricas['psnr']:<12.2f} {metricas['mse']:<12.2f} "
              f"{metricas['coefs_eliminados']:<15,} {metricas['tasa_compresion']:<12.2f}%")
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    test_compresion_rapido()
