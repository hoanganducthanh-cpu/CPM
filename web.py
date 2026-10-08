import streamlit as st
import soundfile as sf
from voxcpm import VoxCPM

st.set_page_config(page_title="VoxCPM TTS", layout="centered")
st.title("🎙️ VoxCPM Text-to-Speech")

@st.cache_resource
def load_model():
    # Tự động chọn thiết bị (GPU/CPU)
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="auto")

model = load_model()

text_input = st.text_area("Nhập văn bản cần chuyển thành giọng nói:", "Xin chào, đây là ví dụ về VoxCPM.")

if st.button("🔊 Tạo giọng nói"):
    with st.spinner("Đang xử lý..."):
        wav = model.generate(text=text_input)
        sf.write("output.wav", wav, model.tts_model.sample_rate)
        st.success("Hoàn thành!")
        st.audio("output.wav", format="audio/wav")