"""
Test rápido para probar el cifrado de imágenes
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Logica.Cifrado_imagen import cifrar_imagen_arnold_frdct, descifrar_imagen_arnold_frdct

def test_cifrado_rapido():
    """Test rápido del sistema de cifrado Arnold + FrDCT"""
    
    print("="*60)
    print("TEST DE CIFRADO DE IMÁGENES")
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
    
    # Parámetros de cifrado
    a = 2
    k = 5
    alpha = 0.5
    
    print(f"\n2. Parámetros de cifrado:")
    print(f"   - a (Arnold): {a}")
    print(f"   - k (iteraciones Arnold): {k}")
    print(f"   - α (FrDCT): {alpha}")
    print(f"\n   Secuencia Cifrado: FrDCT → DOST → Arnold")
    print(f"   Secuencia Descifrado: Arnold⁻¹ → DOST⁻¹ → FrDCT⁻¹")
    
    # Cifrar imagen
    print(f"\n3. Cifrando imagen...")
    try:
        resultado = cifrar_imagen_arnold_frdct(imagen, a, k, alpha)
        print(f"   ✓ Cifrado completado")
        if resultado['coef_eliminados'] > 0:
            print(f"   - Coeficientes eliminados: {resultado['coef_eliminados']:.2f}%")
    except Exception as e:
        print(f"   ✗ Error durante cifrado: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Descifrar imagen
    print(f"\n4. Descifrando imagen...")
    try:
        img_descifrada = descifrar_imagen_arnold_frdct(
            resultado['matriz_frdct'],
            a, k, alpha
        )
        print(f"   ✓ Descifrado completado")
    except Exception as e:
        print(f"   ✗ Error durante descifrado: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calcular métricas de calidad
    print(f"\n5. Calculando métricas de calidad...")
    original_array = resultado['original']
    descifrada_array = np.array(img_descifrada)
    
    mse = np.mean((original_array.astype(float) - descifrada_array.astype(float)) ** 2)
    
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    print(f"   - MSE: {mse:.4f}")
    print(f"   - PSNR: {psnr:.2f} dB")
    
    # Evaluación
    print(f"\n6. Evaluación:")
    if psnr > 30:
        print(f"   ✓ Excelente calidad de recuperación (PSNR > 30 dB)")
    elif psnr > 20:
        print(f"   ✓ Buena calidad de recuperación (PSNR > 20 dB)")
    else:
        print(f"   ⚠ Calidad de recuperación baja (PSNR < 20 dB)")
    
    # Visualización
    print(f"\n7. Generando visualización...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Test de Cifrado FrDCT → DOST → Arnold', fontsize=16, fontweight='bold')
        
        # Fila 1: Proceso de cifrado
        axes[0, 0].imshow(resultado['original'])
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Visualizar FrDCT (primer canal como ejemplo)
        frdct_visual = np.abs(resultado['frdct'][0])
        axes[0, 1].imshow(frdct_visual, cmap='viridis')
        axes[0, 1].set_title(f'FrDCT (α={alpha})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(resultado['cifrada_visual'])
        axes[0, 2].set_title('Cifrada (Arnold)')
        axes[0, 2].axis('off')
        
        # Fila 2: Comparación original vs descifrada
        axes[1, 0].imshow(resultado['original'])
        axes[1, 0].set_title('Original')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_descifrada)
        axes[1, 1].set_title('Descifrada')
        axes[1, 1].axis('off')
        
        # Diferencia
        if len(original_array.shape) == 3 and len(descifrada_array.shape) == 3:
            diferencia = np.abs(original_array.astype(float) - descifrada_array.astype(float))
            # Mostrar diferencia como promedio de canales
            diferencia_visual = diferencia.mean(axis=2)
        else:
            diferencia = np.abs(original_array.astype(float) - descifrada_array.astype(float))
            diferencia_visual = diferencia
        axes[1, 2].imshow(diferencia_visual.astype(np.uint8), cmap='hot')
        axes[1, 2].set_title(f'Diferencia\nMSE: {mse:.2f}')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización
        output_file = 'test_cifrado_resultado.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualización guardada: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"   ⚠ No se pudo generar visualización: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    test_cifrado_rapido()
