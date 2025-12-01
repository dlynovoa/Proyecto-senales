import streamlit as st
import time
import numpy as np
import sounddevice as sd
from PIL import Image
import os
import sys

# Importar lógica del proyecto
from Logica.Reconocimiento_voz import VoiceEngine
from Logica.Segmentacion_imagen import segmentar_con_ia, generar_visualizacion_ia
from Logica.Compresion_imagen import comprimir_imagen_dct, aplicar_compresion
from Logica.Cifrado_imagen import cifrar_imagen_arnold_frdct, descifrar_imagen_arnold_frdct

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Proyecto Final Señales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .stButton>button {
        height: 3em;
        font-weight: bold;
        font-size: 18px;
        border-radius: 8px;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # --- ENCABEZADO ---
    st.title("Sistema de Procesamiento de Imágenes por Voz")
    st.caption("Proyecto Final - Procesamiento Digital de Señales")
    st.markdown("---")

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.header("Configuración")
        
        # 1. Cargar Imagen
        st.subheader("1. Imagen Base")
        
        folder_path = "Recursos"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        archivos = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        image = None
        opcion_carga = st.radio("Fuente:", ["📁 Carpeta Recursos", "⬆️ Subir Archivo"])
        
        if opcion_carga == "📁 Carpeta Recursos":
            if archivos:
                img_selec = st.selectbox("Selecciona:", archivos)
                image = Image.open(os.path.join(folder_path, img_selec))
            else:
                st.warning("Carpeta vacía. Sube imágenes a /Recursos")
        else:
            uploaded = st.file_uploader("Arrastra aquí", type=['jpg', 'png', 'jpeg'])
            if uploaded:
                image = Image.open(uploaded)

        st.markdown("---")
        
        # 2. Motor de Voz
        st.subheader("2. Motor de Voz")
        
        if 'engine' not in st.session_state:
            with st.spinner("🔄 Inicializando motor de voz..."):
                try:
                    st.session_state.engine = VoiceEngine()
                    st.success(" Motor Listo")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.stop()
        else:
            st.success("Motor Activo")
        
        st.markdown("---")
        
        # 3. Parámetros de operaciones
        st.subheader("3. Parámetros")
        
        with st.expander("🗜️ Compresión DCT"):
            pct_comp1 = st.slider("% Compresión 1", 0.5, 100.0, 1.0, 0.5)
            pct_comp2 = st.slider("% Compresión 2", 0.5, 100.0, 2.5, 0.5)
            pct_comp3 = st.slider("% Compresión 3", 0.5, 100.0, 5.0, 0.5)
        
        with st.expander("🔐 Cifrado FrDCT+DOST+Arnold"):
            arnold_a = st.number_input("Parámetro a (Arnold)", 1, 10, 2)
            arnold_k = st.number_input("Iteraciones k (Arnold)", 1, 20, 5)
            frdct_alpha = st.slider("α (FrDCT)", 0.0, 2.0, 0.5, 0.1)

    # --- ÁREA PRINCIPAL ---
    
    if image is None:
        st.info("👈 Por favor carga una imagen en el menú lateral para comenzar.")
        return

    # Mostrar imagen base
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.image(image, caption="Imagen Original", use_container_width=True)
    
    with c2:
        st.markdown("### 🎧 Panel de Control")
        st.write("Presiona el botón y **di un comando claro**:")
        st.code("SEGMENTAR  |  COMPRIMIR  |  CIFRAR", language="text")
        
        st.markdown("#### 🔴 Captura de Audio (2 segundos)")
        
        # BOTÓN DE GRABACIÓN
        if st.button("🎤 GRABAR COMANDO", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            with status_placeholder.container():
                st.info("🎙️ Escuchando...")
            
            fs = 16000  # 16kHz como el compañero
            duration = 1  # 1 segundo como el compañero
            
            try:
                # Grabar audio
                rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                
                # Barra de progreso animada
                for i in range(100):
                    time.sleep(duration / 100)
                    progress_bar.progress(i + 1)
                
                sd.wait()
                
                # Procesar audio
                with status_placeholder.container():
                    st.info("🧠 Analizando audio...")
                
                audio = rec.flatten()
                
                # Reconocer comando
                comando = st.session_state.engine.predecir(audio)
                
                progress_bar.empty()
                
                if comando:
                    with status_placeholder.container():
                        st.success(f"✅ Comando Detectado: **{comando}**")
                    
                    time.sleep(1)
                    status_placeholder.empty()
                    
                    # --- EJECUTAR OPERACIONES ---
                    st.divider()
                    st.subheader(f"📊 Resultado: {comando}")
                    
                    if comando == "SEGMENTAR":
                        ejecutar_segmentacion(image)
                    
                    elif comando == "COMPRIMIR":
                        ejecutar_compresion(image, [pct_comp1, pct_comp2, pct_comp3])
                    
                    elif comando == "CIFRAR":
                        ejecutar_cifrado(image, arnold_a, arnold_k, frdct_alpha)
                    
                    # Esperar 10 segundos antes de reiniciar
                    time.sleep(10)
                    st.rerun()
                
                else:
                    with status_placeholder.container():
                        st.error("❌ No se reconoció el comando. Intenta de nuevo.")
                
            except Exception as e:
                st.error(f"❌ Error de micrófono: {e}")


def ejecutar_segmentacion(image):
    with st.spinner(f"🔍 Segmentando (removiendo fondo)..."):
        try:
            resultado = segmentar_con_ia(image)
            
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            
            col1.image(resultado['original'], caption="Original", use_container_width=True)
            col2.image(resultado['mascara'], caption="Máscara", use_container_width=True)
            col3.image(resultado['segmentada'], caption="Objeto Recortado", use_container_width=True)
            
            # Generar visualización completa
            img_viz = generar_visualizacion_ia(resultado)
            
            st.markdown("### 📊 Resultado de Segmentación ")
            st.image(img_viz, use_container_width=True)
            
            # Información
            with st.expander("ℹ️ Información del Algoritmo"):
                st.write("**Modelo:** U²-Net (Deep Learning)")
                st.write("**Tarea:** Eliminación de fondo automática")
                st.write("**Salida:** Máscara binaria + imagen segmentada")
                
                # Estadísticas de la máscara
                total_pixels = resultado['mascara'].shape[0] * resultado['mascara'].shape[1]
                pixels_objeto = np.sum(resultado['mascara'] > 0)
                porcentaje_objeto = (pixels_objeto / total_pixels) * 100
                
                st.write(f"**Total de píxeles:** {total_pixels:,}")
                st.write(f"**Píxeles del objeto:** {pixels_objeto:,} ({porcentaje_objeto:.1f}%)")
                st.write(f"**Píxeles del fondo:** {total_pixels - pixels_objeto:,} ({100-porcentaje_objeto:.1f}%)")
            
            st.success("✅ Segmentación completada exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error en segmentación: {e}")
            import traceback
            st.code(traceback.format_exc())


def ejecutar_compresion(image, porcentajes):
    """Ejecuta compresión DCT con 3 niveles diferentes"""
    with st.spinner(f"🗜️ Comprimiendo imagen en 3 niveles..."):
        try:
            # Comprimir en los 3 porcentajes
            resultados = []
            for pct in porcentajes:
                resultado = comprimir_imagen_dct(image, pct)
                resultados.append(resultado)
            
            # Mostrar original + 3 comprimidas
            st.markdown("### 🖼️ Comparación de Niveles de Compresión")
            cols = st.columns(4)
            
            # Original
            cols[0].image(image, caption="Original", use_container_width=True)
            cols[0].markdown("**Sin compresión**")
            
            # 3 Comprimidas
            for i, (resultado, pct) in enumerate(zip(resultados, porcentajes)):
                cols[i+1].image(resultado['comprimida'], caption=f"Compresión {i+1}", use_container_width=True)
                cols[i+1].markdown(f"**{pct}% eliminado**")
                cols[i+1].metric("PSNR", f"{resultado['metricas']['psnr']:.2f} dB")
            
            # Métricas detalladas
            st.markdown("### 📊 Métricas Comparativas")
            
            # Tabla de métricas
            import pandas as pd
            data = []
            for i, (resultado, pct) in enumerate(zip(resultados, porcentajes)):
                metricas = resultado['metricas']
                data.append({
                    'Nivel': f'Compresión {i+1}',
                    '% Eliminado': f"{pct}%",
                    'PSNR (dB)': f"{metricas['psnr']:.2f}",
                    'MSE': f"{metricas['mse']:.2f}",
                    'Coefs. Eliminados': f"{metricas['coefs_eliminados']:,}",
                    'Coefs. Mantenidos': f"{metricas['coefs_mantenidos']:,}"
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Información adicional
            with st.expander("ℹ️ Detalles del Algoritmo"):
                st.write(f"**Total de coeficientes por canal:** {resultados[0]['metricas']['total_coefs']:,}")
                st.write(f"**Canales procesados:** RGB (3 canales)")
                
                st.markdown("**Algoritmo:**")
                st.write("1. DCT 2D por bloques de 8×8 en cada canal RGB")
                st.write("2. Eliminación de coeficientes pequeños según porcentaje")
                st.write("3. IDCT 2D para reconstrucción de cada canal")
                st.write("4. Combinación de canales RGB")
            
            st.success("✅ Compresión completada exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error en compresión: {e}")
            import traceback
            st.code(traceback.format_exc())


def ejecutar_cifrado(image, a, k, alpha):
    """Ejecuta cifrado FrDCT + DOST + Arnold"""
    with st.spinner(f"🔐 Cifrando (a={a}, k={k}, α={alpha})..."):
        try:
            # Cifrar
            resultado = cifrar_imagen_arnold_frdct(image, a, k, alpha)
            
            # Mostrar proceso
            st.markdown("### 🔒 Proceso de Cifrado")
            
            cols = st.columns(3)
            cols[0].image(resultado['original'], caption="Original", use_container_width=True)
            # Mostrar FrDCT (magnitud del primer canal)
            frdct_visual = np.abs(resultado['frdct'][0])
            frdct_visual = (frdct_visual - frdct_visual.min())
            if frdct_visual.max() > 0:
                frdct_visual = (frdct_visual / frdct_visual.max() * 255).astype(np.uint8)
            cols[1].image(frdct_visual, caption=f"FrDCT (α={alpha})", use_container_width=True, clamp=True)
            cols[2].image(resultado['cifrada_visual'], caption="Cifrada (Arnold)", use_container_width=True)
            
            st.markdown("---")
            
            # Descifrar
            with st.spinner("🔓 Descifrando..."):
                img_descifrada = descifrar_imagen_arnold_frdct(
                    resultado['matriz_frdct'],
                    a, k, alpha
                )
            
            # Mostrar descifrado
            st.markdown("### 🔓 Verificación de Descifrado")
            
            col1, col2, col3 = st.columns(3)
            col1.image(resultado['original'], caption="Original", use_container_width=True)
            col2.image(resultado['cifrada_visual'], caption="Cifrada", use_container_width=True)
            col3.image(img_descifrada, caption="Descifrada", use_container_width=True)
            
            # Calcular métricas
            mse = np.mean((resultado['original'].astype(float) - img_descifrada.astype(float)) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
            
            st.markdown("### 📊 Métricas de Cifrado/Descifrado")
            
            cols = st.columns(2)
            cols[0].metric("MSE", f"{mse:.2f}")
            cols[1].metric("PSNR", f"{psnr:.2f} dB")
            
            # Información
            with st.expander("ℹ️ Clave de Cifrado"):
                st.write(f"**a (Arnold):** {a}")
                st.write(f"**k (iteraciones):** {k}")
                st.write(f"**α (FrDCT):** {alpha}")
                
                st.markdown("**Proceso de Cifrado:**")
                st.write("1. FrDCT (DCT fraccional)")
                st.write("2. DOST (Discrete Orthonormal Stockwell Transform)")
                st.write("3. Transformación de Arnold (scrambling espacial)")
                
                st.markdown("**Proceso de Descifrado:**")
                st.write("1. Arnold⁻¹ (descrambling)")
                st.write("2. DOST⁻¹ (inversa)")
                st.write("3. FrDCT⁻¹ (inversa)")
            
            st.success("✅ Cifrado/Descifrado completado exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error en cifrado: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()