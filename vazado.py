import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch
import sounddevice as sd 

# Função para criar e aplicar o filtro FIR passa-faixa com transições suaves
def apply_fir_bandpass_filter(audio, lowcut, highcut, sr, numtaps):
    nyquist = 0.5 * sr  # Frequência de Nyquist
    low = lowcut / nyquist  
    high = highcut / nyquist  

    # Filtro FIR com uma janela Blackman-Harris 
    fir_coefficients = firwin(numtaps, [low, high], pass_zero=False, window='blackmanharris')

    # Aplicar o filtro passa-faixa
    filtered_audio = lfilter(fir_coefficients, 1.0, audio)

    return filtered_audio

# Função para salvar o áudio filtrado em um arquivo
def save_audio(filtered_audio, sr, filename):
    sf.write(filename, filtered_audio, sr)

# Função para carregar e processar áudio
def load_and_process_audio(audio_path, lowcut, highcut, sr, numtaps):
    audio, sr = sf.read(audio_path)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    filtered_audio = apply_fir_bandpass_filter(audio, lowcut, highcut, sr, numtaps)
    filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))  # Normalização
    return audio, filtered_audio, sr

# Definição de limites
audio_paths = {
    "vazado_01": {"path": "vazado_1.wav", "lowcut": 700, "highcut": 1500},
    "vazado_03": {"path": "vazado_3.wav", "lowcut": 600, "highcut": 2200}
}
numtaps = 201  # Número de coeficientes

# Função para plotar os gráficos
def plot_graphs(audio, filtered_audio, sr):
    time = np.linspace(0, len(audio) / sr, num=len(audio))
    
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, audio, color='blue')
    plt.title("Waveform Original", fontsize=20)
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_audio, color='green')
    plt.title("Waveform Filtrado", fontsize=20)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.specgram(audio, NFFT=2048, Fs=sr, noverlap=1024, cmap='viridis')
    plt.title("Espectrograma Original", fontsize=20)
    plt.subplot(2, 1, 2)
    plt.specgram(filtered_audio, NFFT=2048, Fs=sr, noverlap=1024, cmap='viridis')
    plt.title("Espectrograma Filtrado", fontsize=20)
    plt.tight_layout()
    plt.show()

    # Gráficos de Densidade Espectral de Potência
    f_audio, Pxx_audio = welch(audio, sr, nperseg=1024)
    f_filtered, Pxx_filtered = welch(filtered_audio, sr, nperseg=1024)
    
    plt.figure(figsize=(14, 6))
    plt.semilogy(f_audio, Pxx_audio, label='Original')
    plt.semilogy(f_filtered, Pxx_filtered, label='Filtrado')
    plt.title("Densidade Espectral de Potência", fontsize=20)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Densidade de Potência")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Função para reproduzir áudio
def play_audio(audio, sr, is_filtered=False):
    sd.play(audio, sr)
    sd.wait()
    
    if is_filtered:
        display_audio_info(audio, sr)

# Função para exibir informações sobre o áudio filtrado
def display_audio_info(audio, sr):
    duration = len(audio) / sr
    print(f"\nDuração do áudio: {duration:.2f} segundos")
    f_audio, Pxx_audio = welch(audio, sr, nperseg=1024)
    max_freq = f_audio[np.argmax(Pxx_audio)]  # Frequência com maior potência
    print(f"Frequência dominante (máxima potência): {max_freq:.2f} Hz")
    print(f"Desvio padrão do áudio: {np.std(audio):.4f}")
    print(f"Taxa de amostragem do áudio: {sr} Hz")

# Execução principal
if __name__ == "__main__":
    while True:
        print("\nEscolha o áudio para processar:")
        for idx, key in enumerate(audio_paths.keys()):
            print(f"{idx + 1} - {key}")
        print("0 - Sair")
        
        choice = input("Digite sua escolha: ")
        if choice == "0":
            print("Encerrando o programa.")
            break
        
        if choice not in map(str, range(1, len(audio_paths) + 1)):
            print("Opção inválida. Tente novamente.")
            continue
        
        selected_audio = list(audio_paths.keys())[int(choice) - 1]
        audio, filtered_audio, sr = load_and_process_audio(
            audio_paths[selected_audio]["path"],
            audio_paths[selected_audio]["lowcut"],
            audio_paths[selected_audio]["highcut"],
            None,
            numtaps
        )

        while True:
            print("\nEscolha uma ação:")
            print("1 - Reproduzir Áudio Original")
            print("2 - Reproduzir Áudio Filtrado")
            print("3 - Exibir Gráficos")
            print("0 - Voltar ao menu de seleção de áudio")
            
            action = input("Digite sua escolha: ")
            if action == "0":
                break
            elif action == "1":
                play_audio(audio, sr)
            elif action == "2":
                play_audio(filtered_audio, sr, is_filtered=True)
            elif action == "3":
                plot_graphs(audio, filtered_audio, sr)
            else:
                print("Opção inválida. Tente novamente.")
        
        save_audio(filtered_audio, sr, f"{selected_audio}_filtrado.wav")
        print(f"Áudio filtrado salvo como {selected_audio}_filtrado.wav")