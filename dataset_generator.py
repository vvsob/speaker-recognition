import streamlit as st
# import torchaudio
# from datsets import load_dataset

@st.cache_resource
def get_datset_generator(filename="ds.hf"):
    return DatasetGenerator(filename)

class DatasetGenerator:
    def __init__(self, filename):
        self.buffer = list()

    def save_recording(self, audio_data, speaker_id):
        print(f"Saving recording for speaker {speaker_id}")

def main():
    st.title("Audio Dataset Generator")

    dataset_generator = get_datset_generator()
    speaker_id = st.text_input("Speaker id")
    st.write("Click the button to start recording:")
    audio_data = st.audio_input("Audio recorder")
    if audio_data and speaker_id:
        st.audio(audio_data)
        if st.button("Save recording"):
            dataset_generator.save_recording(audio_data, speaker_id)

if __name__ == "__main__":
    main()