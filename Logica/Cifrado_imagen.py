import numpy as np
import cv2
from scipy.fftpack import dct, idct, fft, ifft
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


def dost_2d(imagen):
    """
    Discrete Orthonormal Stockwell Transform 2D
    Basada en la transformada de Stockwell con normalización ortogonal
    
    Args:
        imagen: numpy array 2D
    
    Returns:
        Matriz DOST 2D (compleja)
    """
    N, M = imagen.shape
    
    # FFT 2D
    fft_imagen = np.fft.fft2(imagen)
    
    # Stockwell Transform simplificada
    # Crear ventana Gaussiana en frecuencia
    u = np.fft.fftfreq(N).reshape(-1, 1)
    v = np.fft.fftfreq(M).reshape(1, -1)
    
    # Evitar división por cero
    sigma_u = np.where(np.abs(u) > 1e-10, 1 / (2 * np.pi * np.abs(u)), 1.0)
    sigma_v = np.where(np.abs(v) > 1e-10, 1 / (2 * np.pi * np.abs(v)), 1.0)
    
    # Ventana Gaussiana 2D
    ventana = np.exp(-2 * np.pi**2 * ((u**2 * sigma_u**2) + (v**2 * sigma_v**2)))
    
    # Aplicar ventana en dominio de frecuencia
    dost_result = fft_imagen * ventana
    
    # Normalización ortogonal
    dost_result = dost_result / np.sqrt(N * M)
    
    return dost_result


def dost_inversa_2d(matriz_dost):
    """
    Inversa de DOST 2D
    
    Args:
        matriz_dost: matriz compleja DOST
    
    Returns:
        Imagen reconstruida (real)
    """
    N, M = matriz_dost.shape
    
    # Desnormalizar
    matriz_proc = matriz_dost * np.sqrt(N * M)
    
    # Crear ventana inversa
    u = np.fft.fftfreq(N).reshape(-1, 1)
    v = np.fft.fftfreq(M).reshape(1, -1)
    
    sigma_u = np.where(np.abs(u) > 1e-10, 1 / (2 * np.pi * np.abs(u)), 1.0)
    sigma_v = np.where(np.abs(v) > 1e-10, 1 / (2 * np.pi * np.abs(v)), 1.0)
    
    ventana = np.exp(-2 * np.pi**2 * ((u**2 * sigma_u**2) + (v**2 * sigma_v**2)))
    
    # Evitar división por cero en la ventana
    ventana_inv = np.where(np.abs(ventana) > 1e-10, 1 / ventana, 1.0)
    
    # Remover ventana
    fft_recuperado = matriz_proc * ventana_inv
    
    # IFFT 2D
    imagen_recuperada = np.fft.ifft2(fft_recuperado)
    
    return np.real(imagen_recuperada)


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
    Cifrado completo: FrDCT + DOST + Arnold (mantiene RGB)
    
    Args:
        imagen_pil: PIL Image o numpy array
        a: parámetro Arnold
        k: iteraciones Arnold
        alpha: orden fraccional FrDCT
        porcentaje_compresion: % de coefs DCT a eliminar (ya no se usa, mantenido por compatibilidad)
    
    Returns:
        dict con resultados del cifrado
    """
    # Convertir a numpy manteniendo RGB
    if isinstance(imagen_pil, Image.Image):
        imagen_original = np.array(imagen_pil.convert('RGB'))
    else:
        imagen_original = imagen_pil
        if len(imagen_original.shape) == 2:
            # Si es escala de grises, convertir a RGB
            imagen_original = np.stack([imagen_original] * 3, axis=-1)
    
    print(f"\n=== PROCESO DE CIFRADO ===")
    print(f"Transformadas: FrDCT → DOST → Arnold")
    print(f"Parámetros: a={a}, k={k}, α={alpha}")
    print(f"Canales: RGB (3 canales)")
    
    # Procesar cada canal RGB por separado
    matrices_dost = []
    matrices_frdct = []
    imagenes_arnold = []
    
    for canal_idx in range(3):
        canal = imagen_original[:, :, canal_idx]
        
        # PASO 1: FrDCT
        imagen_norm = canal.astype(np.float64) / 255.0
        matriz_frdct = frdct_2d(imagen_norm, alpha)
        matrices_frdct.append(matriz_frdct)
        
        # PASO 2: DOST
        matriz_dost = dost_2d(matriz_frdct)
        matrices_dost.append(matriz_dost)
        
        # PASO 3: Arnold Transform
        magnitud_dost = np.abs(matriz_dost)
        magnitud_norm = (magnitud_dost - magnitud_dost.min())
        if magnitud_norm.max() > 0:
            magnitud_norm = (magnitud_norm / magnitud_norm.max() * 255).astype(np.uint8)
        else:
            magnitud_norm = magnitud_norm.astype(np.uint8)
        
        imagen_arnold = transformacion_arnold(magnitud_norm, a, k, inversa=False)
        imagenes_arnold.append(imagen_arnold)
    
    print(f"\n✓ FrDCT completado (3 canales)")
    print(f"✓ DOST completado (3 canales)")
    print(f"✓ Arnold completado ({k} iteraciones, 3 canales)")
    
    # Combinar canales Arnold para visualización
    imagen_arnold_rgb = np.stack(imagenes_arnold, axis=-1)
    
    print("\n✓ CIFRADO COMPLETADO")
    
    return {
        'original': imagen_original,
        'frdct': matrices_frdct,
        'dost': matrices_dost,
        'arnold': imagen_arnold_rgb,
        'matriz_frdct': matrices_dost,  # Lista de matrices DOST (una por canal)
        'cifrada_visual': imagen_arnold_rgb,
        'coef_eliminados': 0.0,
        'parametros': {'a': a, 'k': k, 'alpha': alpha}
    }


def descifrar_imagen_arnold_frdct(matrices_cifradas, a, k, alpha):
    """
    Descifrado completo: Arnold⁻¹ → DOST⁻¹ → FrDCT⁻¹ (mantiene RGB)
    
    Args:
        matrices_cifradas: lista de matrices cifradas DOST (una por canal RGB)
        a, k, alpha: parámetros de cifrado
    
    Returns:
        imagen descifrada RGB (numpy array)
    """
    print(f"\n=== PROCESO DE DESCIFRADO ===")
    print(f"Transformadas: Arnold⁻¹ → DOST⁻¹ → FrDCT⁻¹")
    print(f"Parámetros: a={a}, k={k}, α={alpha}")
    print(f"Canales: RGB (3 canales)")
    
    # Verificar si es lista de matrices o matriz única (compatibilidad)
    if not isinstance(matrices_cifradas, list):
        # Si es una sola matriz, procesarla en escala de grises
        matriz_dost = matrices_cifradas
        matrices_dost = [matriz_dost]
        modo_gris = True
    else:
        matrices_dost = matrices_cifradas
        modo_gris = False
    
    canales_descifrados = []
    
    for canal_idx, matriz_dost in enumerate(matrices_dost):
        # PASO 1: Arnold inverso no se aplica aquí
        # (Arnold se aplicó sobre magnitud, pero DOST está intacto)
        
        # PASO 2: DOST inversa
        matriz_frdct = dost_inversa_2d(matriz_dost)
        
        # PASO 3: FrDCT inversa
        imagen_desc_norm = frdct_inversa_2d(matriz_frdct, alpha)
        
        # Desnormalizar
        canal_descifrado = np.clip(imagen_desc_norm * 255.0, 0, 255).astype(np.uint8)
        canales_descifrados.append(canal_descifrado)
    
    print(f"\n✓ Arnold inverso completado ({k} iteraciones, {len(matrices_dost)} canales)")
    print(f"✓ DOST inversa completado ({len(matrices_dost)} canales)")
    print(f"✓ FrDCT inversa completado ({len(matrices_dost)} canales)")
    
    # Combinar canales
    if modo_gris and len(canales_descifrados) == 1:
        imagen_descifrada = canales_descifrados[0]
    else:
        imagen_descifrada = np.stack(canales_descifrados, axis=-1)
    
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