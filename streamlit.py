import random

import torchaudio
import torch
import wave
from io import BytesIO
import librosa
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

torch.classes.__path__ = []

import streamlit as st
from model.classifier import Classifier


def load_audio_from_wave_object(wave_obj):
    """Loads audio data from a wave.Wave_read object into a Torch tensor.

    Args:
        wave_obj: A wave.Wave_read object.

    Returns:
        A tuple containing:
            - A Torch tensor containing the audio data.
            - The sample rate of the audio.
    """

    num_frames = wave_obj.getnframes()
    num_channels = wave_obj.getnchannels()
    sample_rate = wave_obj.getframerate()
    sample_width = wave_obj.getsampwidth()

    # Read all frames from the wave object
    str_data = wave_obj.readframes(num_frames)

    # Convert the byte string to a NumPy array.  Important: Use the correct dtype
    # based on the sample width.
    if sample_width == 1:
        dtype = torch.int8  # 8-bit PCM
    elif sample_width == 2:
        dtype = torch.int16  # 16-bit PCM (most common)
    elif sample_width == 4:
        dtype = torch.int32  # 32-bit PCM
    elif sample_width == 3:  # 24 bit PCM
        dtype = torch.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio_data = torch.frombuffer(str_data, dtype=dtype)

    # Reshape the 1D array to (channels, frames) if stereo or multichannel
    if num_channels > 1:
        audio_data = audio_data.reshape(num_frames, num_channels).t().contiguous()  # Transpose for (C, T)

    # Normalize if needed (optional, but often recommended)
    # Be careful with int types; divide by the maximum possible value.
    if dtype == torch.int8:
        audio_data = audio_data.float() / (2 ** 7 - 1)
    elif dtype == torch.int16:
        audio_data = audio_data.float() / (2 ** 15 - 1)
    elif dtype == torch.int32:
        audio_data = audio_data.float() / (2 ** 31 - 1)

    return audio_data.unsqueeze(0), sample_rate


@st.cache_resource
def get_audio_processor():
    return AudioProcessor()


class AudioProcessor:
    def __init__(self):
        self.classifier = Classifier()
        self.id_map = {0: "Вячеслав Чертан", 1: "Миша Рыбалков", 2: "Олюбочка Соболь", 3: "Путис", 4: "Дима Авдеев",
                       5: "Азер"}
        self.array=None
        self.sampling_rate=None

    def get_user(self, audio_data):
        self.array, self.sampling_rate = load_audio_from_wave_object(wave.open(BytesIO(audio_data)))
        id, probabilities = self.classifier.get_id_probabilities(self.array, self.sampling_rate)
        return id, self.id_map.get(id, None), probabilities

    def get_fig(self, audio_data):
        fig = self.get_figure((self.array, self.sampling_rate))
        return fig

    def get_id_map(self):
        return self.id_map

    def wrap_probabilities(self, probabilities):
        speaker_probabilities = sorted([(speaker, float(probabilities[id])) for id, speaker in self.id_map.items()], key=lambda x: -x[1])
        return "##### " + "\n##### ".join(
            [f"{speaker}: {"{:.4f}".format(prob)}" for speaker, prob in speaker_probabilities])

    # files - list of pairs of (torchaudio.load(***), "filename")
    def get_figure(self, file, fmax=16000, graph_size=3):
        base_scale = 100

        array, sampling_rate = file
        length_seconds = array.shape[1]//sampling_rate

        fig = make_subplots(rows=3, cols=1, )

        x = np.linspace(0, length_seconds, length_seconds * sampling_rate);
        y = array[0, :sampling_rate * length_seconds]

        fig.add_trace(
            go.Scatter(x=x, y=y, name="", ),
            row=1, col=1
        )
        fig.update_yaxes(range=[-1, 1], title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Time, seconds", row=1, col=1)

        spectrogram = librosa.stft(y.numpy())
        spectrogram_dB = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

        time = librosa.times_like(spectrogram_dB, sr=sampling_rate)
        freq = librosa.fft_frequencies(sr=sampling_rate)

        fig.add_trace(go.Heatmap(
            x=time,
            y=freq,
            z=spectrogram_dB.tolist(),
            colorscale='Viridis',
            showscale=False
        ), row=2, col=1)
        fig.update_yaxes(title_text="Frequency, Hz", row=2, col=1)
        fig.update_xaxes(title_text="Time, seconds", row=2, col=1)

        mel_spectrogram = librosa.feature.melspectrogram(y=y.numpy(), sr=sampling_rate, n_mels=128, fmax=fmax)

        mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)

        time = librosa.times_like(mel_spectrogram_dB, sr=sampling_rate)
        freq = librosa.mel_frequencies(n_mels=128, fmax=fmax)

        fig.add_trace(go.Heatmap(
            x=time,
            y=freq,
            z=mel_spectrogram_dB.tolist(),
            colorscale='Electric',
            showscale=True,
            colorbar_title="dB"
        ), row=3, col=1).update_coloraxes(showscale=False)
        fig.update_yaxes(title_text="Frequency, Hz", row=3, col=1)
        fig.update_xaxes(title_text="Time, seconds", row=3, col=1)

        fig.update_layout(height=base_scale * 3 * graph_size, width=graph_size * base_scale * 5, showlegend=False)
        return fig


@st.cache_resource
def get_phrase_generator():
    return PhraseGenerator()


class PhraseGenerator:
    def __init__(self, phrase_source="phrases.txt"):
        self.phrases = open(phrase_source, "r", encoding="utf8").readlines()[::2]

    def get_phrase(self):
        return random.choice(self.phrases)


def main():
    st.title("Распознавание спикера по голосу")
    need_debug = st.checkbox("Показать дополнительные данные")

    audio_processor = get_audio_processor()
    phrase_generator = get_phrase_generator()

    st.markdown(f"#### Скажите что-нибудь, например: \n##### {phrase_generator.get_phrase()}")

    audio_data = st.audio_input("Запишите звук для снятия биометрических данных")
    if audio_data:
        id, name, probabilities = audio_processor.get_user(audio_data.getvalue())
        # st.markdown(f"#### id: {id}")
        if need_debug:
            st.markdown(f"####  Распределение предсказаний:\n\n{audio_processor.wrap_probabilities(probabilities)}")
            st.plotly_chart(audio_processor.get_fig(audio_data))

        elif name is not None:
            st.markdown(f"#### Имя: {name}")


if __name__ == "__main__":
    main()
