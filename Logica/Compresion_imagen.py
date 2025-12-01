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
    """DCT 2D optimizada con operaciones vectorizadas de NumPy"""
    N, M = bloque.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    # Precalcular matrices de coseno usando broadcasting
    k = np.arange(N)[:, np.newaxis]
    n = np.arange(N)
    cos_matrix_n = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    l = np.arange(M)[:, np.newaxis]
    m = np.arange(M)
    cos_matrix_m = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    # Calcular DCT usando multiplicación de matrices
    # X[k,l] = alfa[k] * beta[l] * sum(bloque[n,m] * cos_n[k,n] * cos_m[l,m])
    # Reorganizamos: X = alfa * (cos_n @ bloque @ cos_m.T) * beta
    X = cos_matrix_n @ bloque @ cos_matrix_m.T
    X = X * alfa[:, np.newaxis] * beta
    
    return X


def idct_2d_manual(coeficientes):
    """IDCT 2D optimizada con operaciones vectorizadas de NumPy"""
    N, M = coeficientes.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    # Precalcular matrices de coseno usando broadcasting
    k = np.arange(N)[:, np.newaxis]
    n = np.arange(N)
    cos_matrix_n = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    l = np.arange(M)[:, np.newaxis]
    m = np.arange(M)
    cos_matrix_m = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    # Calcular IDCT usando multiplicación de matrices
    # x[n,m] = sum(alfa[k] * beta[l] * coef[k,l] * cos_n[k,n] * cos_m[l,m])
    # Reorganizamos: x = cos_n.T @ (alfa * coef * beta) @ cos_m
    coef_escalados = coeficientes * alfa[:, np.newaxis] * beta
    x = cos_matrix_n.T @ coef_escalados @ cos_matrix_m
    
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
    # Guardar imagen original para métricas
    if isinstance(imagen, Image.Image):
        imagen_original_uint8 = np.array(imagen.convert('RGB'))
        imagen = imagen_original_uint8.astype(np.float64)
    else:
        imagen_original_uint8 = imagen.astype(np.uint8)
        if imagen.dtype != np.float64:
            imagen = imagen.astype(np.float64)
    
    # Determinar si es escala de grises o RGB
    es_color = len(imagen.shape) == 3
    
    if es_color:
        # Imagen RGB
        h, w, c = imagen.shape
        forma_original = (h, w, c)
    else:
        # Imagen escala de grises
        h, w = imagen.shape
        c = 1
        forma_original = (h, w)
    
    # Padding para múltiplos del tamaño de bloque
    pad_h = (tamanio_bloque - (h % tamanio_bloque)) % tamanio_bloque
    pad_w = (tamanio_bloque - (w % tamanio_bloque)) % tamanio_bloque
    
    if pad_h > 0 or pad_w > 0:
        if es_color:
            imagen = np.pad(imagen, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        else:
            imagen = np.pad(imagen, ((0, pad_h), (0, pad_w)), mode='edge')
    
    if es_color:
        h_pad, w_pad, _ = imagen.shape
    else:
        h_pad, w_pad = imagen.shape
        
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    
    if es_color:
        total_bloques_procesados = total_bloques * 3
        print(f"  Procesando imagen RGB: {total_bloques} bloques x 3 canales = {total_bloques_procesados} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    else:
        print(f"  Procesando {total_bloques} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    
    # Aplicar DCT a cada bloque (por canal si es RGB)
    dct_coefs = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    if es_color:
        # Procesar cada canal RGB por separado
        for canal in range(3):
            for i in range(0, h_pad, tamanio_bloque):
                for j in range(0, w_pad, tamanio_bloque):
                    bloque = imagen[i:i+tamanio_bloque, j:j+tamanio_bloque, canal]
                    dct_bloque = dct_2d_manual(bloque)
                    dct_coefs[i:i+tamanio_bloque, j:j+tamanio_bloque, canal] = dct_bloque
                    
                    bloque_actual += 1
                    if bloque_actual % 200 == 0 or bloque_actual == total_bloques_procesados:
                        porcentaje_progreso = (bloque_actual / total_bloques_procesados) * 100
                        print(f"    DCT: {bloque_actual}/{total_bloques_procesados} bloques ({porcentaje_progreso:.1f}%)")
    else:
        # Procesar escala de grises
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
    
    if es_color:
        # Reconstruir cada canal RGB
        for canal in range(3):
            for i in range(0, h_pad, tamanio_bloque):
                for j in range(0, w_pad, tamanio_bloque):
                    bloque_dct = coefs_filtrados[i:i+tamanio_bloque, j:j+tamanio_bloque, canal]
                    bloque_rec = idct_2d_manual(bloque_dct)
                    imagen_rec[i:i+tamanio_bloque, j:j+tamanio_bloque, canal] = bloque_rec
                    
                    bloque_actual += 1
                    if bloque_actual % 200 == 0 or bloque_actual == total_bloques_procesados:
                        porcentaje_progreso = (bloque_actual / total_bloques_procesados) * 100
                        print(f"    IDCT: {bloque_actual}/{total_bloques_procesados} bloques ({porcentaje_progreso:.1f}%)")
        
        # Recortar padding
        imagen_rec = imagen_rec[:forma_original[0], :forma_original[1], :]
    else:
        # Reconstruir escala de grises
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
    
    # Calcular métricas usando imagen original guardada
    total_coefs = coefs_filtrados.size
    metricas = calcular_metricas(forma_original, imagen_rec, num_eliminados, total_coefs, imagen_original_uint8)
    
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


def calcular_metricas(forma_original, imagen_comprimida, num_coefs_eliminados, total_coefs, imagen_original=None):
    """Calcula MSE, PSNR y tasa de compresión"""
    # Si no se proporciona imagen original, crear una de referencia
    if imagen_original is None:
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