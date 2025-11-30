import numpy as np
import os
import json
import soundfile as sf
import sounddevice as sd

class VoiceEngine:
    def __init__(self, data_path="Datos"):
        self.json_path = os.path.join(data_path, "Vectores_promedio.json")
        self.bank_path = os.path.join(data_path, "Banco_audios")
        
        # Configuración de Audio (del compañero)
        self.SAMPLE_RATE = 16000  # 16kHz como tu compañero
        self.DURATION = 1.0       # 1 segundo como tu compañero
        self.CHANNELS = 1
        
        # Configuración FFT (del compañero)
        self.N_FFT = 4096
        self.K_SUBBANDAS = 16
        self.VENTANA = "hamming"
        
        # Parámetros de filtrado (del compañero)
        self.FRECUENCIA_CORTE_PB = 3500
        self.ORDEN_FILTRO = 4
        self.PREENFASIS_ALPHA = 0.97
        
        self.templates = {}
        self.muestras_temp = {1: [], 2: [], 3: []}
        
        self.inicializar()

    def inicializar(self):
        """Carga el modelo si existe, sino entrena automáticamente"""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    self.templates = data.get("commands", {})
                    print(f"✅ Modelo cargado: {len(self.templates)} comandos")
            except Exception as e:
                print(f"⚠ Error cargando modelo: {e}")
        else:
            if os.path.exists(self.bank_path):
                print("📚 No existe modelo. Entrenando desde carpetas...")
                self.entrenar_desde_carpetas()
    
    # ==================== PROCESAMIENTO DE AUDIO (DEL COMPAÑERO) ====================
    
    def aplicar_preenfasis(self, senal, coef=None):
        """Pre-énfasis para realzar altas frecuencias"""
        if coef is None:
            coef = self.PREENFASIS_ALPHA
        return np.append(senal[0], senal[1:] - coef * senal[:-1])
    
    def filtrar_pasabajos(self, senal, fs):
        """Filtro pasabajos Butterworth"""
        from scipy.signal import butter, filtfilt
        
        b, a = butter(self.ORDEN_FILTRO, self.FRECUENCIA_CORTE_PB / (fs / 2), btype='low')
        return filtfilt(b, a, senal)
    
    def eliminar_silencio(self, senal, fs):
        """Eliminación de silencio por energía (VAD)"""
        energia = senal ** 2
        ventana_muestras = int(0.025 * fs)
        
        # Energía por ventana
        energia_ventana = np.convolve(energia, np.ones(ventana_muestras)/ventana_muestras, mode='same')
        energia_ventana = np.maximum(energia_ventana, 1e-10)
        energia_db = 10 * np.log10(energia_ventana)
        
        # Umbral dinámico
        umbral = np.max(energia_db) - 30  # 30 dB debajo del máximo
        
        mascara_voz = energia_db > umbral
        
        if not np.any(mascara_voz):
            return senal
        
        indices_voz = np.where(mascara_voz)[0]
        margen_muestras = int(0.1 * fs)  # 100ms margen
        
        inicio = max(0, indices_voz[0] - margen_muestras)
        fin = min(len(senal), indices_voz[-1] + margen_muestras)
        
        return senal[inicio:fin]
    
    def extraer_ventana_maxima_energia(self, senal, N):
        """Extrae ventana de N muestras con máxima energía"""
        if len(senal) <= N:
            ventana = np.pad(senal, (0, N - len(senal)))
        else:
            mejor_energia = -1
            mejor_inicio = 0
            paso = N // 4
            
            for i in range(0, len(senal) - N, paso):
                ventana_temp = senal[i:i + N]
                energia = np.sum(ventana_temp ** 2)
                
                if energia > mejor_energia:
                    mejor_energia = energia
                    mejor_inicio = i
            
            ventana = senal[mejor_inicio:mejor_inicio + N]
        
        # Normalizar RMS
        rms = np.sqrt(np.mean(ventana ** 2))
        if rms > 1e-6:
            ventana = ventana * (0.1 / rms)
        
        return ventana
    
    def calcular_vector_energias_temporal(self, senal):
        """
        Calcula vector de energías por subbandas (MÉTODO DEL COMPAÑERO)
        Retorna vector de K energías
        """
        from scipy.signal import get_window
        
        # Centrar señal
        x = senal - np.mean(senal)
        
        # Pre-énfasis
        x = self.aplicar_preenfasis(x)
        
        # Ajustar a N_FFT
        if len(x) < self.N_FFT:
            xN = np.pad(x, (0, self.N_FFT - len(x)), mode='constant')
        elif len(x) > self.N_FFT:
            start = (len(x) - self.N_FFT) // 2
            xN = x[start:start + self.N_FFT]
        else:
            xN = x
        
        # Ventaneo
        if self.VENTANA.lower() == "none" or self.VENTANA == "rect":
            w = np.ones(self.N_FFT)
        else:
            w = get_window(self.VENTANA, self.N_FFT, fftbins=True)
        
        xN_windowed = xN * w
        
        # FFT
        X = np.fft.fft(xN_windowed, n=self.N_FFT)
        
        # Solo frecuencias positivas
        N_half = self.N_FFT // 2
        X_positivas = X[:N_half]
        
        # Dividir en K subbandas
        puntos_por_subbanda = N_half // self.K_SUBBANDAS
        
        energias = np.zeros(self.K_SUBBANDAS, dtype=np.float32)
        
        for i in range(self.K_SUBBANDAS):
            inicio = i * puntos_por_subbanda
            
            if i == self.K_SUBBANDAS - 1:
                fin = N_half
            else:
                fin = (i + 1) * puntos_por_subbanda
            
            Xi = X_positivas[inicio:fin]
            
            # Energía normalizada
            Ei = (1.0 / self.N_FFT) * np.sum(np.abs(Xi) ** 2)
            energias[i] = Ei
        
        return energias

    def entrenar_desde_carpetas(self):
        """Lee WAV desde carpetas y genera templates (MÉTODO DEL COMPAÑERO)"""
        comandos = ["SEGMENTAR", "COMPRIMIR", "CIFRAR"]
        nuevo_modelo = {"commands": {}}
        
        for cmd in comandos:
            ruta = os.path.join(self.bank_path, cmd)
            if not os.path.exists(ruta):
                print(f"⚠ Carpeta no encontrada: {ruta}")
                continue
            
            vectores_energia = []
            archivos = [f for f in os.listdir(ruta) if f.endswith(".wav")]
            
            print(f"📂 Procesando {cmd}: {len(archivos)} archivos")
            
            for f in archivos:
                try:
                    # Cargar audio
                    a, fs_original = sf.read(os.path.join(ruta, f))
                    
                    # Convertir a mono si es estéreo
                    if len(a.shape) > 1:
                        a = a[:,0]
                    
                    # Remuestrear si es necesario
                    if fs_original != self.SAMPLE_RATE:
                        from scipy.signal import resample
                        duracion = len(a) / fs_original
                        nuevo_num_muestras = int(duracion * self.SAMPLE_RATE)
                        a = resample(a, nuevo_num_muestras)
                    
                    # Procesar
                    senal = self.filtrar_pasabajos(a, self.SAMPLE_RATE)
                    senal = self.eliminar_silencio(senal, self.SAMPLE_RATE)
                    senal = self.aplicar_preenfasis(senal)
                    senal = self.extraer_ventana_maxima_energia(senal, self.N_FFT)
                    
                    # Extraer vector de energías
                    vector = self.calcular_vector_energias_temporal(senal)
                    vectores_energia.append(vector)
                    
                except Exception as e:
                    print(f"  ✗ Error en {f}: {e}")
            
            if len(vectores_energia) == 0:
                print(f"⚠ No se procesó ningún archivo para {cmd}")
                continue
            
            # Calcular estadísticas (media y desviación)
            matriz = np.vstack(vectores_energia)
            medias = np.mean(matriz, axis=0)
            desviaciones = np.std(matriz, axis=0, ddof=0)
            
            nuevo_modelo["commands"][cmd] = {
                "mean": medias.tolist(),
                "std": desviaciones.tolist(),
                "count": len(vectores_energia)
            }
            
            print(f"  ✓ {len(vectores_energia)} ejemplos procesados")
            print(f"  Media: {medias}")
        
        # Guardar modelo
        nuevo_modelo["config"] = {
            "fs": self.SAMPLE_RATE,
            "N": self.N_FFT,
            "K": self.K_SUBBANDAS,
            "window": self.VENTANA
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(nuevo_modelo, f, indent=2)
        
        self.templates = nuevo_modelo["commands"]
        print(f"✅ Modelo guardado en {self.json_path}")

    def predecir(self, audio):
        """Reconoce comando usando distancia Euclidiana (MÉTODO DEL COMPAÑERO)"""
        if not self.templates:
            print("⚠ No hay modelo entrenado")
            return None
        
        # Procesar audio de entrada
        try:
            senal = self.filtrar_pasabajos(audio, self.SAMPLE_RATE)
            senal = self.eliminar_silencio(senal, self.SAMPLE_RATE)
            senal = self.aplicar_preenfasis(senal)
            senal = self.extraer_ventana_maxima_energia(senal, self.N_FFT)
            
            # Extraer vector de energías
            E = self.calcular_vector_energias_temporal(senal)
        except Exception as e:
            print(f"⚠ Error procesando audio: {e}")
            return None
        
        # Normalizar
        norma_E = np.linalg.norm(E)
        if norma_E > 1e-10:
            E_norm = E / norma_E
        else:
            E_norm = E
        
        print(f"\n{'='*60}")
        print(f"RECONOCIMIENTO DE COMANDO")
        print(f"{'='*60}")
        print(f"Vector entrada: {E}")
        print(f"Energía total: {np.sum(E):.6f}")
        print(f"Vector normalizado: {E_norm}")
        print(f"{'-'*60}")
        
        # Comparar con templates
        distancias = {}
        
        for nombre_comando, datos_comando in self.templates.items():
            umbral_vector = np.array(datos_comando.get("mean", []), dtype=float)
            
            if len(umbral_vector) == 0:
                print(f"⚠ {nombre_comando}: sin vector de umbrales")
                continue
            
            # Normalizar umbral
            norma_umbral = np.linalg.norm(umbral_vector)
            if norma_umbral > 1e-10:
                umbral_norm = umbral_vector / norma_umbral
            else:
                umbral_norm = umbral_vector
            
            # Distancia Euclidiana
            distancia = np.linalg.norm(E_norm - umbral_norm)
            distancias[nombre_comando] = distancia
            
            print(f"{nombre_comando}:")
            print(f"  Umbral normalizado: {umbral_norm}")
            print(f"  Distancia: {distancia:.6f}")
        
        if len(distancias) == 0:
            print(f"✗ No hay comandos para comparar")
            return None
        
        # Seleccionar mejor comando
        mejor_comando = min(distancias.items(), key=lambda x: x[1])
        comando_ganador = mejor_comando[0]
        distancia_minima = mejor_comando[1]
        
        print(f"{'-'*60}")
        print(f"RANKING DE COMANDOS (por cercanía):")
        sorted_distancias = sorted(distancias.items(), key=lambda x: x[1])
        for i, (cmd, dist) in enumerate(sorted_distancias, 1):
            marca = "★" if cmd == comando_ganador else " "
            print(f"  {marca} {i}° {cmd}: {dist:.6f}")
        
        print(f"{'='*60}")
        print(f"✓ COMANDO RECONOCIDO: {comando_ganador}")
        print(f"  Distancia: {distancia_minima:.6f}")
        print(f"{'='*60}\n")
        
        return comando_ganador

    def grabar_audio(self):
        """Graba audio desde micrófono"""
        print(f"🎤 Grabando {self.DURATION}s...")
        rec = sd.rec(
            int(self.DURATION * self.SAMPLE_RATE), 
            samplerate=self.SAMPLE_RATE, 
            channels=1, 
            dtype='float32'
        )
        sd.wait()
        return rec.flatten()


if __name__ == "__main__":
    # Prueba rápida
    engine = VoiceEngine()
    print("\n--- Modo Prueba: Habla un comando ---")
    audio = engine.grabar_audio()
    cmd = engine.predecir(audio)
    print(f"\n🎯 Resultado: {cmd}")