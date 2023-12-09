import streamlit as st
import torchaudio
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import librosa

# Define the model and processor
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

# Streamlit app
def main():
    st.title("Audio to Text Converter")

    # Record audio
    st.subheader("Step 1: Record Audio")
    st.info("Click the button below to start recording audio.")

    # Record audio with torchaudio
    recording = st.button("Record")

    if recording:
        st.warning("Recording... Speak into your microphone.")
        with st.spinner("Recording..."):
            audio_data, sampling_rate = torchaudio.recorder.record(10, 16000, channels=1)
        st.success("Recording complete!")

        # Convert audio to text
        st.subheader("Step 2: Convert Audio to Text")
        st.info("Click the button below to convert the recorded audio to text.")
        if st.button("Convert to Text"):
            st.warning("Converting audio to text... Please wait.")

            # Save the recorded audio to a temporary file
            temp_audio_path = "/tmp/recorded_audio.wav"
            sf.write(temp_audio_path, audio_data[0].numpy(), sampling_rate)

            # Load the audio file
            audio, _ = librosa.load(temp_audio_path, sr=16000)

            # Prepare features for the model
            features = processor(audio, sampling_rate=16000, return_tensors="pt")

            # Generate the transcription
            generated_ids = model.generate(input_features=features.input_features)

            # Decode the generated IDs
            transcription = processor.batch_decode(generated_ids)

            # Display the results
            st.subheader("Audio transcription:")
            st.write(transcription)

if __name__ == "__main__":
    main()
