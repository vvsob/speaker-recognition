import torchaudio
import torch
import wave
from io import BytesIO

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
        self.id_map = {"0": "Вячеслав Чертан", "1": "Миша Рыбалков", "2": "Олюбочка Соболь", "3": "Путис", "4": "Дима Авдеев", "5": "Азер"} # temporary mapping

    def get_user(self, audio_data):
        array, sampling_rate = load_audio_from_wave_object(wave.open(BytesIO(audio_data)))
        id = self.classifier.get_id(array, sampling_rate)
        return id, self.id_map.get(id, None)


def main():
    st.title("Audio Recorder and Processor")

    audio_processor = get_audio_processor()

    st.write("Click the button to start recording:")
    audio_data = st.audio_input("Audio recorder")
    if audio_data:
        id, name = audio_processor.get_user(audio_data.getvalue())
        if name is not None:
            st.write(f"Name: {name}")
        st.write(f"id: {id}")

if __name__ == "__main__":
    main()