import numpy as np
import cv2
from PIL import Image

def calcular_coeficientes_dct(N, M):
    """Calcula coeficientes α y β para DCT-2D"""
    beta = np.zeros(M)
    beta[0] = np.sqrt(1.0 / M)
    for l in range(1, M):
        beta[l] = np.sqrt(2.0 / M)
    
    alfa = np.zeros(N)
    alfa[0] = np.sqrt(1.0 / N)
    for k in range(1, N):
        alfa[k] = np.sqrt(2.0 / N)
    
    return alfa, beta


def dct_2d_manual(bloque):
    """DCT 2D manual (sin OpenCV)"""
    N, M = bloque.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    # Precalcular matrices de coseno
    cos_matrix_n = np.zeros((N, N))
    cos_matrix_m = np.zeros((M, M))
    
    for k in range(N):
        for n in range(N):
            cos_matrix_n[k, n] = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    for l in range(M):
        for m in range(M):
            cos_matrix_m[l, m] = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    X = np.zeros((N, M), dtype=np.float64)
    
    for k in range(N):
        for l in range(M):
            suma = 0.0
            for n in range(N):
                for m in range(M):
                    suma += bloque[n, m] * cos_matrix_m[l, m] * cos_matrix_n[k, n]
            
            X[k, l] = alfa[k] * beta[l] * suma
    
    return X


def idct_2d_manual(coeficientes):
    """IDCT 2D manual (reconstrucción)"""
    N, M = coeficientes.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    cos_matrix_n = np.zeros((N, N))
    cos_matrix_m = np.zeros((M, M))
    
    for k in range(N):
        for n in range(N):
            cos_matrix_n[k, n] = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    for l in range(M):
        for m in range(M):
            cos_matrix_m[l, m] = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    x = np.zeros((N, M), dtype=np.float64)
    
    for n in range(N):
        for m in range(M):
            suma = 0.0
            for k in range(N):
                for l in range(M):
                    suma += alfa[k] * beta[l] * coeficientes[k, l] * cos_matrix_m[l, m] * cos_matrix_n[k, n]
            
            x[n, m] = suma
    
    return x


def comprimir_imagen_dct(imagen, porcentaje_compresion, tamanio_bloque=8):
    """
    Comprime imagen usando DCT por bloques
    
    Args:
        imagen: numpy array o PIL Image
        porcentaje_compresion: % de coeficientes a ELIMINAR (0-100)
        tamanio_bloque: Tamaño de bloques (default 8x8)
    
    Returns:
        dict con 'original', 'comprimida', 'dct_coefs', 'metricas'
    """
    # Convertir a numpy si es PIL
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen.convert('L'))
    
    if imagen.dtype != np.float64:
        imagen = imagen.astype(np.float64)
    
    h, w = imagen.shape
    forma_original = imagen.shape
    
    # Padding para múltiplos del tamaño de bloque
    pad_h = (tamanio_bloque - (h % tamanio_bloque)) % tamanio_bloque
    pad_w = (tamanio_bloque - (w % tamanio_bloque)) % tamanio_bloque
    
    if pad_h > 0 or pad_w > 0:
        imagen = np.pad(imagen, ((0, pad_h), (0, pad_w)), mode='edge')
    
    h_pad, w_pad = imagen.shape
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    
    print(f"  Procesando {total_bloques} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    
    # Aplicar DCT a cada bloque
    dct_coefs = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque = imagen[i:i+tamanio_bloque, j:j+tamanio_bloque]
            dct_bloque = dct_2d_manual(bloque)
            dct_coefs[i:i+tamanio_bloque, j:j+tamanio_bloque] = dct_bloque
            
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    DCT: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    # Eliminar coeficientes pequeños
    coefs_filtrados, num_eliminados = eliminar_coeficientes_pequenos(
        dct_coefs, porcentaje_compresion
    )
    
    # Reconstruir imagen
    print(f"  Reconstruyendo imagen...")
    imagen_rec = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque_dct = coefs_filtrados[i:i+tamanio_bloque, j:j+tamanio_bloque]
            bloque_rec = idct_2d_manual(bloque_dct)
            imagen_rec[i:i+tamanio_bloque, j:j+tamanio_bloque] = bloque_rec
            
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    IDCT: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    # Recortar padding
    imagen_rec = imagen_rec[:forma_original[0], :forma_original[1]]
    imagen_rec = np.clip(imagen_rec, 0, 255).astype(np.uint8)
    
    # Calcular métricas
    total_coefs = coefs_filtrados.size
    metricas = calcular_metricas(forma_original, imagen_rec, num_eliminados, total_coefs)
    
    return {
        'original': forma_original,
        'comprimida': imagen_rec,
        'dct_coefs': coefs_filtrados,
        'num_eliminados': num_eliminados,
        'metricas': metricas,
        'porcentaje': porcentaje_compresion
    }


def eliminar_coeficientes_pequenos(dct_coefs, porcentaje):
    """Pone a cero los coeficientes más pequeños"""
    coefs_planos = dct_coefs.flatten()
    total_coefs = len(coefs_planos)
    
    num_eliminar = int((porcentaje / 100.0) * total_coefs)
    
    if num_eliminar < 1:
        return dct_coefs.copy(), 0
    
    # Ordenar por magnitud absoluta
    indices_ordenados = np.argsort(np.abs(coefs_planos))
    
    # Copiar y poner ceros
    coefs_filtrados = coefs_planos.copy()
    coefs_filtrados[indices_ordenados[:num_eliminar]] = 0
    
    coefs_filtrados = coefs_filtrados.reshape(dct_coefs.shape)
    
    return coefs_filtrados, num_eliminar


def calcular_metricas(forma_original, imagen_comprimida, num_coefs_eliminados, total_coefs):
    """Calcula MSE, PSNR y tasa de compresión"""
    # Crear imagen original para comparar (assume gris)
    imagen_original = np.zeros(forma_original, dtype=np.uint8)
    
    mse = np.mean((imagen_original.astype(float) - imagen_comprimida.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)
    
    tasa_compresion = (num_coefs_eliminados / total_coefs) * 100
    coefs_mantenidos = total_coefs - num_coefs_eliminados
    
    return {
        'mse': mse,
        'psnr': psnr,
        'tasa_compresion': tasa_compresion,
        'coefs_eliminados': num_coefs_eliminados,
        'coefs_mantenidos': coefs_mantenidos,
        'total_coefs': total_coefs
    }


def aplicar_compresion(imagen_pil, porcentaje_eliminacion):
    """
    Función wrapper para Streamlit
    
    Args:
        imagen_pil: PIL Image
        porcentaje_eliminacion: % de coeficientes a eliminar
    
    Returns:
        PIL Image comprimida
    """
    resultado = comprimir_imagen_dct(imagen_pil, porcentaje_eliminacion)
    return Image.fromarray(resultado['comprimida'])


if __name__ == "__main__":
    import sys
    
    ruta = sys.argv[1] if len(sys.argv) > 1 else "Recursos/Prueba1.jpg"
    
    img = Image.open(ruta)
    print(f"Comprimiendo: {ruta}")
    
    for p in [0.5, 1.0, 2.0]:
        print(f"\n--- Compresión {p}% ---")
        resultado = comprimir_imagen_dct(img, p)
        print(f"PSNR: {resultado['metricas']['psnr']:.2f} dB")
        print(f"MSE: {resultado['metricas']['mse']:.2f}")