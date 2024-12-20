import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.fftpack as fft

# Função para carregar o arquivo WAV
def carregar_audio(arquivo):
    with wave.open(arquivo, 'rb') as wav_file:
        # Obtendo as informações do arquivo de áudio
        n_channels = wav_file.getnchannels()  # Número de canais (1: mono, 2: estéreo)
        sample_rate = wav_file.getframerate()  # Taxa de amostragem (Hz)
        n_frames = wav_file.getnframes()  # Número de frames (amostras totais)
        
        # Lendo os frames de áudio
        frames = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        # Se o áudio for estéreo, pega apenas o primeiro canal
        if n_channels == 2:
            audio_data = audio_data[::2]
        
        return audio_data, sample_rate

# Função para calcular a FFT e plotar
def plot_fft(audio_data, sample_rate, fmin=0, fmax=None):
    # Calculando a FFT
    fft_result = fft.fft(audio_data)
    
    # Calculando as frequências correspondentes
    n = len(audio_data)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    
    # Pegando apenas a parte positiva das frequências
    pos_freqs = freqs[:n//2]
    pos_fft = np.abs(fft_result[:n//2])
    
    # Plotando a FFT
    plt.figure(figsize=(10, 6))
    plt.plot(pos_freqs, pos_fft)
    
    # Ajustando os limites do eixo X
    if fmax is None:
        fmax = np.max(pos_freqs)
    plt.xlim(fmin, fmax)  # Ajustando intervalo do eixo X em Hz
    
    # Títulos e rótulos
    plt.title("Espectro de Frequência do Áudio")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Caminho do arquivo WAV
arquivo_audio = 'vazado_3.wav'

# Carregando o áudio
audio_data, sample_rate = carregar_audio(arquivo_audio)

# Plotando a FFT e limitando o eixo X entre 0 e 2000 Hz
plot_fft(audio_data, sample_rate, fmin=0, fmax=5100)