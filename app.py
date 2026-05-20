from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from core.image_enhancer import adjust_brightness_contrast, apply_clahe, apply_denoise
from industrial_ad.batch import BatchDetector
from industrial_ad.config import BatchConfig, DetectorConfig
from industrial_ad.detector import AnomalyDetector
from industrial_ad.reports import write_reports
from industrial_ad.storage import DetectionStore


APP_TITLE = "工业异常检测与数据平台"
DEFAULT_MODEL_PATH = "models/clipseg"


def convert_image_to_bytes(img_array) -> bytes:
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()


def convert_pil_to_bytes(img_pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()


@st.cache_resource
def get_detector(model_path: str = DEFAULT_MODEL_PATH, threshold: float = 0.72) -> AnomalyDetector:
    config = DetectorConfig(model_path=model_path, threshold=threshold)
    detector = AnomalyDetector(config=config)
    return detector.load()


@st.cache_resource
def get_batch_runner(model_path: str = DEFAULT_MODEL_PATH, threshold: float = 0.72) -> BatchDetector:
    detector = get_detector(model_path, threshold)
    return BatchDetector(detector, BatchConfig())


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="工")
st.title("工业异常检测与数据平台")
st.caption("支持单图检测、批量检测、图像增强、历史统计和报告导出。")

with st.sidebar:
    st.header("检测配置")
    product_name = st.text_input("产品英文名", value="metal nut")
    model_path = st.text_input("模型目录", value=DEFAULT_MODEL_PATH)
    threshold = st.slider("判定阈值", 0.0, 1.0, 0.72, 0.01)
    st.divider()
    image_source = st.radio("图片来源", ["使用系统预设图片", "手动上传本地图片"])

    selected_image_pil = None
    if image_source == "使用系统预设图片":
        sample_dir = Path("sample_images")
        sample_dir.mkdir(exist_ok=True)
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        sample_files = sorted(
            [item.name for item in sample_dir.iterdir() if item.suffix.lower() in valid_extensions]
        )
        if sample_files:
            selected_file = st.selectbox("选择测试图片", sample_files)
            selected_image_pil = Image.open(sample_dir / selected_file).convert("RGB")
    else:
        uploaded_file = st.file_uploader("支持 JPG, PNG, JPEG, BMP, WEBP", type=["jpg", "png", "jpeg", "bmp", "webp"])
        if uploaded_file is not None:
            selected_image_pil = Image.open(uploaded_file).convert("RGB")

tab1, tab2, tab3 = st.tabs(["核心检测", "图像增强实验室", "生产数据大屏"])

with tab2:
    st.subheader("图像增强实验室")
    if selected_image_pil is None:
        st.info("先在左侧选择或上传一张图片。")
    else:
        col_ctrl, col_img = st.columns([1, 2])
        with col_ctrl:
            use_clahe = st.checkbox("启用 CLAHE", value=False)
            use_denoise = st.checkbox("启用去噪", value=False)
            brightness = st.slider("亮度", -100, 100, 0)
            contrast = st.slider("对比度", -100, 100, 0)

        processed_img = selected_image_pil
        if use_clahe:
            processed_img = apply_clahe(processed_img)
        if use_denoise:
            processed_img = apply_denoise(processed_img)
        if brightness != 0 or contrast != 0:
            processed_img = adjust_brightness_contrast(processed_img, brightness, contrast)

        with col_img:
            st.image(processed_img, caption="预处理结果", width=520)

        st.session_state["final_image"] = processed_img
        st.success("预处理参数已应用，可切回核心检测页继续执行。")

with tab1:
    st.subheader("单图检测")
    if selected_image_pil is None:
        st.info("先选择一张图片。")
    else:
        img_to_detect = st.session_state.get("final_image", selected_image_pil)
        st.image(img_to_detect, width=420, caption="待检测图片")

        if st.button("开始检测", type="primary", use_container_width=True):
            if not product_name.strip():
                st.warning("请先输入产品英文名。")
            else:
                detector = get_detector(model_path, threshold)
                with st.spinner("正在执行检测..."):
                    result = detector.detect_image(img_to_detect, product_name, output_dir="outputs/single")
                DetectionStore().append(result)

                st.header("检测结果")
                if result.is_defective:
                    st.error(f"发现缺陷 | 分数: {result.score:.2f}")
                else:
                    st.success(f"未发现明显缺陷 | 分数: {result.score:.2f}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("#### 原图")
                    st.image(img_to_detect, use_column_width=True)
                    st.download_button("下载原图", convert_pil_to_bytes(img_to_detect), "input.jpg", "image/jpeg")
                with c2:
                    st.markdown("#### 热力图")
                    if result.heatmap_path and Path(result.heatmap_path).exists():
                        st.image(result.heatmap_path, use_column_width=True)
                    st.download_button(
                        "下载热力图",
                        convert_pil_to_bytes(Image.open(result.heatmap_path).convert("RGB"))
                        if result.heatmap_path and Path(result.heatmap_path).exists()
                        else b"",
                        "heatmap.jpg",
                        "image/jpeg",
                    )
                with c3:
                    st.markdown("#### 目标框图")
                    if result.boxed_path and Path(result.boxed_path).exists():
                        st.image(result.boxed_path, use_column_width=True)
                    st.download_button(
                        "下载框图",
                        convert_pil_to_bytes(Image.open(result.boxed_path).convert("RGB"))
                        if result.boxed_path and Path(result.boxed_path).exists()
                        else b"",
                        "boxed.jpg",
                        "image/jpeg",
                    )
                st.json(result.to_dict())

with tab3:
    st.subheader("生产数据大屏")
    store = DetectionStore()
    stats = store.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计检测", stats["total"])
    c2.metric("缺陷数量", stats["defective"])
    c3.metric("良品数量", stats["normal"])
    c4.metric("平均分数", f'{stats["average_score"]:.4f}')
    st.progress(min(max(stats["defect_rate"], 0.0), 1.0))
    history = store.load()
    if history:
        st.dataframe(
            [
                {
                    "time": item.created_at,
                    "product": item.product_name,
                    "status": item.status,
                    "score": item.score,
                    "regions": item.region_count,
                }
                for item in reversed(history[-20:])
            ],
            use_container_width=True,
        )
    else:
        st.info("暂无检测记录。")

    if st.button("导出当前历史报告"):
        if history:
            from industrial_ad.types import BatchItemResult, BatchSummary

            batch_items = [BatchItemResult(image_path=item.source_path or "", ok=True, result=item) for item in history]
            summary = BatchSummary(product_name="history", items=batch_items)
            written = write_reports(summary, "outputs/history", ["json", "csv", "html"])
            st.success(f"已导出 {len(written)} 份报告")
            st.write([str(path) for path in written])
        else:
            st.warning("没有历史可导出。")
