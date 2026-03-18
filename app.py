import os
import io
import streamlit as st
from PIL import Image

# 导入核心算法与工具包
from core.model_handler import load_local_model
from core.inference import detect_universal_anomaly
from core.utils import setup_logger, save_detection_record

# ---> 【新增】导入预处理引擎与数据大屏模块 <---
from core.image_enhancer import adjust_brightness_contrast, apply_clahe, apply_denoise
from core.dashboard import render_dashboard


def convert_image_to_bytes(img_array):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()


def convert_pil_to_bytes(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()


# ==========================================
# 1. 页面设置、日志初始化与模型加载
# ==========================================
st.set_page_config(page_title="工业异常检测综合平台", layout="wide", page_icon="🏭")

logger = setup_logger()
processor, model = load_local_model()

st.title("🏭 工业异常检测与大数据平台 (Zero-Shot)")
st.markdown("集成了 **工业图像增强预处理**、**零样本大模型缺陷检测** 以及 **生产质量分析大屏** 的一站式 SaaS 平台。")

# 使用选项卡 (Tabs) 结构化页面
tab1, tab2, tab3 = st.tabs(["🔍 核心检测引擎", "⚙️ 图像预处理实验室", "📈 生产数据大屏"])

# ==========================================
# Tab 1 & Tab 2 共用的侧边栏图片上传逻辑
# ==========================================
with st.sidebar:
    st.header("⚙️ 质检对象配置")
    product_name = st.text_input("请输入产品英文名称：", value="metal nut",
                                 help="例如: bottle, pill, leather, metal nut...")
    st.markdown("---")

    st.header("🖼️ 数据源获取")
    image_source = st.radio("选择图片获取方式：", ("使用系统预设图片", "手动上传本地图片"))

    selected_image_pil = None
    if image_source == "使用系统预设图片":
        sample_dir = "sample_images"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        valid_extensions = ('.jpg', '.jpeg', '.png')
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(valid_extensions)]

        if len(sample_files) > 0:
            selected_file = st.selectbox("请选择一张测试图片：", sample_files)
            img_path = os.path.join(sample_dir, selected_file)
            selected_image_pil = Image.open(img_path).convert("RGB")
    else:
        uploaded_file = st.file_uploader("支持 JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            selected_image_pil = Image.open(uploaded_file).convert("RGB")

# ==========================================
# Tab 2: 图像预处理实验室 (用户可以在检测前调节图片画质)
# ==========================================
with tab2:
    st.header("⚙️ 工业相机画质增强实验室")
    st.info("在工业现场中，光线暗淡或高噪点会影响 AI 识别。在这里您可以对传入的图像进行专业级预处理。")

    if selected_image_pil is not None:
        col_ctrl, col_img = st.columns([1, 2])

        with col_ctrl:
            st.subheader("调整参数")
            use_clahe = st.checkbox("启用 CLAHE (工业自适应直方图均衡)", value=False)
            use_denoise = st.checkbox("启用非局部均值去噪", value=False)
            brightness = st.slider("亮度调节", -100, 100, 0)
            contrast = st.slider("对比度调节", -100, 100, 0)

        with col_img:
            # 应用图像处理算法
            processed_img = selected_image_pil
            if use_clahe:
                processed_img = apply_clahe(processed_img)
            if use_denoise:
                processed_img = apply_denoise(processed_img)
            if brightness != 0 or contrast != 0:
                processed_img = adjust_brightness_contrast(processed_img, brightness, contrast)

            st.image(processed_img, caption="预处理实时预览结果", width=500)

            # 将处理后的图片保存到 session_state 供 Tab 1 检测使用
            st.session_state['final_image'] = processed_img
            st.success("✅ 预处理参数已应用，请返回【🔍 核心检测引擎】执行检测。")
    else:
        st.warning("请先在左侧侧边栏上传或选择一张图片。")

# ==========================================
# Tab 1: 核心检测引擎 (主界面)
# ==========================================
with tab1:
    if selected_image_pil is not None:
        # 决定使用原图还是预处理后的图
        img_to_detect = st.session_state.get('final_image', selected_image_pil)

        st.subheader("📥 待测图像预览")
        st.image(img_to_detect, width=400, caption="系统将基于此图像进行 AI 检测")

        if st.button("🚀 启动大模型零样本缺陷检测", type="primary", use_container_width=True):
            if not product_name:
                st.warning("请在左侧侧边栏输入产品名称！")
            else:
                logger.info(f"开始对产品 '{product_name}' 执行零样本异常检测...")

                with st.spinner(f"正在对 '{product_name}' 进行数百万参数级特征比对分析..."):
                    # 调用算法
                    is_def, score, heatmap_img, bbox_img = detect_universal_anomaly(
                        img_to_detect, product_name, processor, model
                    )

                save_detection_record(product_name, is_def, score)
                logger.info(f"检测完成！结果: {'发现缺陷' if is_def else '正常'} | 得分: {score:.4f}")

                # --- 结果展示区 ---
                st.header("📊 智能分析结果")
                if is_def:
                    st.error(f"⚠️ **发现产品缺陷！**  |  综合异常得分: `{score:.2f}` (得分越高缺陷越严重)")
                else:
                    st.success(f"✅ **未发现明显异常。**  |  综合异常得分: `{score:.2f}`")

                # 三列展示并提供下载
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.markdown("#### 1. 原始分析图像")
                    st.image(img_to_detect, use_column_width=True)
                    st.download_button("⬇️ 下载此图", convert_pil_to_bytes(img_to_detect), "input.jpg", "image/jpeg",
                                       key="d1")

                with res_col2:
                    st.markdown("#### 2. 缺陷热力分布图")
                    st.image(heatmap_img, use_column_width=True)
                    st.download_button("⬇️ 下载热力图", convert_image_to_bytes(heatmap_img), "heatmap.jpg",
                                       "image/jpeg", key="d2")

                with res_col3:
                    st.markdown("#### 3. 像素级精确定位")
                    st.image(bbox_img, use_column_width=True)
                    st.download_button("⬇️ 下载定位框图", convert_image_to_bytes(bbox_img), "bbox.jpg", "image/jpeg",
                                       key="d3")
    else:
        st.info("👈 请先在左侧侧边栏选择图片数据源。")

# ==========================================
# Tab 3: 生产数据大屏
# ==========================================
with tab3:
    render_dashboard()
