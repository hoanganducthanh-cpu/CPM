import os
import re
import sys
import logging
import random
import numpy as np
import streamlit as st
import soundfile as sf
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Copy class VoxCPMDemo từ code gốc của bạn ---
class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id, optimize=self.optimize, device=self.device,
        )
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(f"Loading ASR model: {self.asr_model_id}")
        self.asr_model = AutoModel(
            model=self.asr_model_id, disable_update=True, log_level="DEBUG", device=self.asr_device,
        )
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(input=prompt_wav, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = False,
        denoise: bool = False,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()
        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Vui lòng nhập văn bản cần tổng hợp.")

        control = (control_instruction or "").strip()
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean

        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)

# --- Khởi tạo Model (Cache để không load lại) ---
@st.cache_resource(show_spinner=False)
def load_demo_model():
    return VoxCPMDemo(model_id="openbmb/VoxCPM2", device="auto")

# --- Giao diện Streamlit ---
st.set_page_config(page_title="VoxCPM Studio", layout="wide", page_icon="🎙️")
st.title("🎙️ VoxCPM2 - Text to Speech Studio")
st.markdown("""
**VoxCPM2 cung cấp 3 chế độ:**
* 🎨 **Voice Design**: Tạo giọng mới từ mô tả (Control Instruction).
* 🎛️ **Controllable Cloning**: Clone giọng từ audio mẫu + điều khiển cảm xúc.
* 🎙️ **Ultimate Cloning**: Clone hoàn hảo từng chi tiết (Cần nhập text của audio mẫu).
""")

demo = load_demo_model()

# Sidebar cho các cài đặt
with st.sidebar:
    st.header("🎤 Audio Mẫu (Tùy chọn)")
    ref_audio_file = st.file_uploader("Tải lên file âm thanh mẫu", type=["wav", "mp3", "flac"])
    
    ultimate_cloning = st.checkbox("🎙️ Bật Ultimate Cloning Mode", value=False)
    
    prompt_text = ""
    if ultimate_cloning and ref_audio_file is not None:
        if st.button("📝 Tự động nhận diện văn bản (ASR)"):
            with st.spinner("Đang nhận diện..."):
                # Lưu file tạm để ASR đọc
                with open("temp_ref_asr.wav", "wb") as f:
                    f.write(ref_audio_file.getbuffer())
                asr_result = demo.prompt_wav_recognition("temp_ref_asr.wav")
                st.session_state["prompt_text_val"] = asr_result
                st.success("Đã nhận diện xong!")
                
        prompt_text = st.text_area(
            "Văn bản của Audio mẫu (Transcript)", 
            value=st.session_state.get("prompt_text_val", ""),
            height=100
        )

    st.header("⚙️ Cài đặt nâng cao")
    denoise = st.checkbox("Khử nhiễu Audio mẫu (ZipEnhancer)", value=False)
    do_normalize = st.checkbox("Chuẩn hóa văn bản (wetext)", value=False)
    cfg_value = st.slider("CFG (độ bám sát prompt)", 1.0, 3.0, 2.0, 0.1)
    dit_steps = st.slider("LocDiT Steps (chất lượng)", 1, 50, 10, 1)
    
    use_random_seed = st.checkbox("Dùng Seed ngẫu nhiên", value=True)
    seed_value = st.number_input("Seed", value=random.randint(0, 2**32 - 1), disabled=use_random_seed)

# Main Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Nhập liệu")
    control_instruction = st.text_area(
        "🎛️ Control Instruction (Mô tả giọng nói, cảm xúc...)",
        placeholder="Ví dụ: Giọng nam trầm ấm, chậm rãi. Hoặc 'A young girl with a soft, sweet voice'",
        height=100,
        disabled=ultimate_cloning # Tắt nếu dùng Ultimate Cloning
    )
    
    target_text = st.text_area(
        "✍️ Target Text (Nội dung cần đọc)",
        value="VoxCPM2 là mô hình TTS đa ngôn ngữ sáng tạo đến từ ModelBest, được thiết kế để tạo ra giọng nói có độ chân thực cao.",
        height=150
    )
    
    generate_btn = st.button("🔊 Bắt đầu tạo giọng nói", type="primary", use_container_width=True)

with col2:
    st.subheader("🎧 Kết quả")
    audio_output = st.empty()

if generate_btn:
    if not target_text.strip():
        st.warning("Vui lòng nhập văn bản cần đọc!")
    else:
        with st.spinner("Đang xử lý trên GPU... Quá trình này có thể mất vài giây..."):
            try:
                # Xử lý file audio mẫu
                audio_path = None
                if ref_audio_file is not None:
                    with open("temp_ref_gen.wav", "wb") as f:
                        f.write(ref_audio_file.getbuffer())
                    audio_path = "temp_ref_gen.wav"
                
                # Chuẩn bị seed
                current_seed = random.randint(0, 2**32 - 1) if use_random_seed else int(seed_value)
                
                # Chuẩn bị tham số
                actual_control = "" if ultimate_cloning else control_instruction
                actual_prompt = prompt_text if ultimate_cloning else ""
                
                # Gọi hàm sinh âm thanh
                sr, wav_np, used_seed = demo.generate_tts_audio(
                    text_input=target_text,
                    control_instruction=actual_control,
                    reference_wav_path_input=audio_path,
                    prompt_text=actual_prompt,
                    cfg_value_input=cfg_value,
                    do_normalize=do_normalize,
                    denoise=denoise,
                    inference_timesteps=int(dit_steps),
                    seed=current_seed
                )
                
                # Lưu file kết quả
                output_path = "output_streamlit.wav"
                sf.write(output_path, wav_np, sr)
                
                st.success(f"Tạo thành công! (Seed sử dụng: {used_seed})")
                with open(output_path, "rb") as f:
                    audio_bytes = f.read()
                audio_output.audio(audio_bytes, format="audio/wav")
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")
                st.exception(e)
