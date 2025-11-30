import numpy as np
import cv2
from PIL import Image
from rembg import remove

def segmentar_con_ia(imagen_pil_o_ruta):
    """
    Segmenta imagen usando IA (rembg) para quitar fondo
    
    Args:
        imagen_pil_o_ruta: PIL Image o ruta de archivo
    
    Returns:
        dict con 'original', 'mascara', 'segmentada'
    """
    # Cargar imagen
    if isinstance(imagen_pil_o_ruta, str):
        img_pil = Image.open(imagen_pil_o_ruta)
    else:
        img_pil = imagen_pil_o_ruta
    
    print("  Procesando con IA (rembg)...")
    
    # Quitar fondo con IA
    salida_pil = remove(img_pil)
    
    # Convertir a numpy
    original_np = np.array(img_pil.convert("RGB"))
    salida_np = np.array(salida_pil)
    
    # Extraer máscara del canal Alpha
    if salida_np.shape[2] == 4:
        mascara = salida_np[:, :, 3]
    else:
        mascara = np.zeros(salida_np.shape[:2], dtype=np.uint8)
    
    # Crear imagen segmentada (fondo negro)
    fondo_negro = np.zeros_like(original_np)
    mascara_3ch = cv2.merge([mascara, mascara, mascara])
    segmentada = np.where(mascara_3ch > 0, original_np, fondo_negro)
    
    print("  ✓ Segmentación IA completada")
    
    return {
        'original': original_np,
        'mascara': mascara,
        'segmentada': segmentada
    }


def generar_visualizacion_ia(resultado):
    """Genera imágenes PIL para mostrar en Streamlit"""
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Original
    axes[0].imshow(resultado['original'])
    axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Máscara
    axes[1].imshow(resultado['mascara'], cmap='gray')
    axes[1].set_title('Máscara (IA - Alpha Channel)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Segmentada
    axes[2].imshow(resultado['segmentada'])
    axes[2].set_title('Objeto Recortado', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convertir a PIL
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


if __name__ == "__main__":
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "Recursos/Prueba1.jpg"
    
    print(f"Segmentando con IA: {ruta}")
    resultado = segmentar_con_ia(ruta)
    
    img_viz = generar_visualizacion_ia(resultado)
    img_viz.show()