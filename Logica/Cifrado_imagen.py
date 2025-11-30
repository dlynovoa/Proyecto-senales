import numpy as np
import cv2
from scipy.fftpack import dct, idct
from PIL import Image

def transformacion_arnold(imagen, a, k, inversa=False):
    """
    Transformación de Arnold (Cat Map)
    
    Args:
        imagen: numpy array
        a: parámetro de Arnold
        k: número de iteraciones
        inversa: True para descifrado
    """
    n, m = imagen.shape
    resultado = imagen.copy()
    
    es_cuadrada = (n == m)
    
    if inversa:
        for _ in range(k):
            temp = np.zeros_like(resultado)
            if es_cuadrada:
                for x in range(n):
                    for y in range(m):
                        x_new = ((a + 1) * x - y) % n
                        y_new = (-a * x + y) % m
                        temp[x_new, y_new] = resultado[x, y]
            else:
                for x in range(n):
                    for y in range(m):
                        x_new = (y - a * x) % n
                        y_new = x % m
                        temp[x_new, y_new] = resultado[x, y]
            resultado = temp
    else:
        for _ in range(k):
            temp = np.zeros_like(resultado)
            if es_cuadrada:
                for x in range(n):
                    for y in range(m):
                        x_new = (x + y) % n
                        y_new = (a * x + (a + 1) * y) % m
                        temp[x_new, y_new] = resultado[x, y]
            else:
                for x in range(n):
                    for y in range(m):
                        x_new = y % n
                        y_new = (x + a * y) % m
                        temp[x_new, y_new] = resultado[x, y]
            resultado = temp
    
    return resultado


def frdct_2d(imagen, alpha):
    """
    Fractional DCT 2D
    
    Args:
        imagen: numpy array
        alpha: orden fraccional (0.0 - 2.0)
    """
    # DCT estándar
    dct_result = dct(dct(imagen.T, norm='ortho').T, norm='ortho')
    
    # Modulación fraccional
    if abs(alpha) > 1e-6:
        N, M = imagen.shape
        u_vals = np.arange(N).reshape(-1, 1)
        v_vals = np.arange(M).reshape(1, -1)
        
        phase_u = alpha * u_vals / (2 * N)
        phase_v = alpha * v_vals / (2 * M)
        modulation = np.exp(-1j * np.pi * (phase_u + phase_v))
        
        dct_result = np.real(dct_result * modulation)
    
    return dct_result


def frdct_inversa_2d(matriz, alpha):
    """Inversa de FrDCT 2D"""
    matriz_proc = matriz.copy()
    
    # Demodulación
    if abs(alpha) > 1e-6:
        N, M = matriz.shape
        u_vals = np.arange(N).reshape(-1, 1)
        v_vals = np.arange(M).reshape(1, -1)
        
        phase_u = alpha * u_vals / (2 * N)
        phase_v = alpha * v_vals / (2 * M)
        modulation_inv = np.exp(1j * np.pi * (phase_u + phase_v))
        
        matriz_proc = np.real(matriz_proc * modulation_inv)
    
    # IDCT estándar
    resultado = idct(idct(matriz_proc.T, norm='ortho').T, norm='ortho')
    
    return resultado


def comprimir_dct(imagen, porcentaje_eliminacion):
    """Compresión DCT rápida usando OpenCV"""
    imagen_float = imagen.astype(np.float32)
    dct_coef = cv2.dct(imagen_float)
    
    # Umbralización
    coef_flat = dct_coef.flatten()
    umbral_comp = np.percentile(np.abs(coef_flat), porcentaje_eliminacion)
    dct_comprimida = dct_coef.copy()
    dct_comprimida[np.abs(dct_comprimida) < umbral_comp] = 0
    
    # Reconstruir
    imagen_comprimida = cv2.idct(dct_comprimida)
    imagen_comprimida = np.clip(imagen_comprimida, 0, 255).astype(np.uint8)
    
    coef_eliminados = np.sum(dct_comprimida == 0) / dct_comprimida.size * 100
    
    return imagen_comprimida, dct_comprimida, coef_eliminados


def cifrar_imagen_arnold_frdct(imagen_pil, a=2, k=5, alpha=0.5, porcentaje_compresion=2.0):
    """
    Cifrado completo: Arnold + Compresión + FrDCT
    
    Args:
        imagen_pil: PIL Image o numpy array
        a: parámetro Arnold
        k: iteraciones Arnold
        alpha: orden fraccional FrDCT
        porcentaje_compresion: % de coefs DCT a eliminar
    
    Returns:
        dict con resultados del cifrado
    """
    # Convertir a numpy
    if isinstance(imagen_pil, Image.Image):
        imagen_original = np.array(imagen_pil.convert('L'))
    else:
        imagen_original = imagen_pil
    
    if len(imagen_original.shape) == 3:
        imagen_original = cv2.cvtColor(imagen_original, cv2.COLOR_RGB2GRAY)
    
    print(f"\n=== PROCESO DE CIFRADO ===")
    print(f"Parámetros: a={a}, k={k}, α={alpha}")
    print(f"Compresión: {porcentaje_compresion}% coeficientes eliminados")
    
    # PASO 1: Arnold Transform
    print("\nPASO 1: Transformación de Arnold...")
    imagen_arnold = transformacion_arnold(imagen_original, a, k, inversa=False)
    print(f"✓ Arnold completado ({k} iteraciones)")
    
    # PASO 2: Compresión DCT
    print("\nPASO 2: Compresión DCT...")
    imagen_comprimida, matriz_dct_comprimida, coef_eliminados = comprimir_dct(
        imagen_arnold, porcentaje_compresion
    )
    print(f"✓ Compresión: {coef_eliminados:.2f}% coeficientes eliminados")
    
    # PASO 3: FrDCT
    print("\nPASO 3: Aplicando FrDCT...")
    imagen_norm = imagen_comprimida.astype(np.float64) / 255.0
    matriz_frdct = frdct_2d(imagen_norm, alpha)
    print(f"✓ FrDCT completado")
    
    # Visualización (imagen cifrada como ruido)
    imagen_cifrada = np.abs(matriz_frdct)
    imagen_cifrada = (imagen_cifrada - imagen_cifrada.min())
    imagen_cifrada = (imagen_cifrada / imagen_cifrada.max() * 255).astype(np.uint8)
    
    print("\n✓ CIFRADO COMPLETADO")
    
    return {
        'original': imagen_original,
        'arnold': imagen_arnold,
        'comprimida': imagen_comprimida,
        'matriz_dct': matriz_dct_comprimida,
        'matriz_frdct': matriz_frdct,
        'cifrada_visual': imagen_cifrada,
        'coef_eliminados': coef_eliminados,
        'parametros': {'a': a, 'k': k, 'alpha': alpha}
    }


def descifrar_imagen_arnold_frdct(matriz_frdct, a, k, alpha):
    """
    Descifrado completo: FrDCT⁻¹ + Arnold⁻¹
    
    Args:
        matriz_frdct: matriz cifrada FrDCT
        a, k, alpha: parámetros de cifrado
    
    Returns:
        imagen descifrada (numpy array)
    """
    print(f"\n=== PROCESO DE DESCIFRADO ===")
    print(f"Parámetros: a={a}, k={k}, α={alpha}")
    
    # PASO 1: FrDCT inversa
    print("\nPASO 1: FrDCT inversa...")
    imagen_desc_norm = frdct_inversa_2d(matriz_frdct, alpha)
    
    imagen_desc_arnold = np.abs(imagen_desc_norm)
    imagen_desc_arnold = (imagen_desc_arnold - imagen_desc_arnold.min())
    imagen_desc_arnold = (imagen_desc_arnold / imagen_desc_arnold.max() * 255).astype(np.uint8)
    print(f"✓ FrDCT inversa completado")
    
    # PASO 2: Arnold inverso
    print("\nPASO 2: Arnold inverso...")
    imagen_descifrada = transformacion_arnold(imagen_desc_arnold, a, k, inversa=True)
    print(f"✓ Arnold inverso completado ({k} iteraciones)")
    
    print("\n✓ DESCIFRADO COMPLETADO")
    
    return imagen_descifrada


# Funciones wrapper para Streamlit
def cifrar_frdct(imagen_pil, a=2, k=5, alpha=0.5):
    """Wrapper simple para Streamlit"""
    resultado = cifrar_imagen_arnold_frdct(imagen_pil, a, k, alpha)
    return Image.fromarray(resultado['cifrada_visual']), resultado['matriz_frdct']


def descifrar_frdct(matriz_frdct, a=2, k=5, alpha=0.5):
    """Wrapper simple para Streamlit"""
    imagen_desc = descifrar_imagen_arnold_frdct(matriz_frdct, a, k, alpha)
    return Image.fromarray(imagen_desc)


if __name__ == "__main__":
    import sys
    
    ruta = sys.argv[1] if len(sys.argv) > 1 else "Recursos/Prueba1.jpg"
    
    img = Image.open(ruta)
    print(f"Cifrando: {ruta}")
    
    # Cifrar
    resultado = cifrar_imagen_arnold_frdct(img, a=2, k=5, alpha=0.5)
    
    # Descifrar
    img_desc = descifrar_imagen_arnold_frdct(
        resultado['matriz_frdct'],
        a=2, k=5, alpha=0.5
    )
    
    # Calcular MSE
    mse = np.mean((resultado['original'].astype(float) - img_desc.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    print(f"\n📊 MÉTRICAS:")
    print(f"  MSE: {mse:.2f}")
    print(f"  PSNR: {psnr:.2f} dB")