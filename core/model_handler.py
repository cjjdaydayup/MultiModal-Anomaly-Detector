import os
import streamlit as st
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


@st.cache_resource
def load_local_model():
    """从本地目录加载大模型，避免每次刷新页面重新加载"""
    local_model_path = "./models/clipseg"

    if not os.path.exists(local_model_path):
        st.error(f"❌ 找不到本地模型：{local_model_path}。请先运行 `python download_model.py`")
        st.stop()

    processor = CLIPSegProcessor.from_pretrained(local_model_path)
    model = CLIPSegForImageSegmentation.from_pretrained(local_model_path)

    return processor, model