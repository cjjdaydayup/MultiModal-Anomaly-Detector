import os
import io
import streamlit as st
from PIL import Image

# 导入核心算法包
from core.model_handler import load_local_model
from core.inference import detect_universal_anomaly


# ==========================================
# 工具函数：将 Numpy 图像数组转换为可下载的字节流
# ==========================================
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
# 1. 页面设置与模型加载
# ==========================================
st.set_page_config(page_title="工业异常检测平台", layout="wide")
processor, model = load_local_model()

st.title("🏭 多模态工业异常检测平台")
st.markdown("A multi-modal industrial anomaly detection system for cybersecurity applications.")

# ==========================================
# 2. 侧边栏：检测配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 质检配置")
    product_name = st.text_input("1. 请输入产品英文名称：", value="metal nut",
                                 help="例如: bottle, pill, leather, metal nut...")
    st.markdown("---")

    st.header("🖼️ 数据源选择")
    image_source = st.radio("2. 选择图片获取方式：", ("使用系统预设图片", "手动上传本地图片"))

# ==========================================
# 3. 主页面：获取待检测图片并预览
# ==========================================
selected_image_pil = None

st.subheader("📥 步骤 1: 确认待检测图片")

if image_source == "使用系统预设图片":
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # 获取文件夹内所有图片
    valid_extensions = ('.jpg', '.jpeg', '.png')
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(valid_extensions)]

    if len(sample_files) == 0:
        st.warning(f"文件夹 `{sample_dir}` 中没有找到图片。请先下载一些工业测试图放入该文件夹中！")
    else:
        # 下拉菜单选择图片
        selected_file = st.selectbox("请选择一张测试图片：", sample_files)
        img_path = os.path.join(sample_dir, selected_file)
        selected_image_pil = Image.open(img_path).convert("RGB")

        # 预览原图
        st.image(selected_image_pil, caption=f"原图预览：{selected_file}", width=400)

else:
    uploaded_file = st.file_uploader("支持 JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        selected_image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(selected_image_pil, caption="原图预览：已上传的产品图片", width=400)

st.divider()

# ==========================================
# 4. 执行检测与结果展示（含下载功能）
# ==========================================
if selected_image_pil is not None:
    st.subheader("🚀 步骤 2: 执行异常检测")

    if st.button("开始智能检测", type="primary", use_container_width=True):
        if not product_name:
            st.warning("请在左侧侧边栏输入产品名称！")
        else:
            with st.spinner(f"正在对 '{product_name}' 进行深度特征分析，请稍候..."):
                # 调用核心算法
                is_def, score, heatmap_img, bbox_img = detect_universal_anomaly(
                    selected_image_pil, product_name, processor, model
                )

            # --- 结果展示区 ---
            st.header("📊 智能分析结果")
            if is_def:
                st.error(f"⚠️ **发现产品缺陷！**  |  综合异常得分: `{score:.2f}` (得分越高缺陷越严重)")
            else:
                st.success(f"✅ **未发现明显异常。**  |  综合异常得分: `{score:.2f}`")

            # 三列对比展示
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 1. 原始图像")
                st.image(selected_image_pil, use_column_width=True)
                # 原图下载按钮
                st.download_button(
                    label="⬇️ 下载原始图像",
                    data=convert_pil_to_bytes(selected_image_pil),
                    file_name="original_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            with col2:
                st.markdown("#### 2. 缺陷热力图")
                st.image(heatmap_img, use_column_width=True)
                # 热力图下载按钮
                st.download_button(
                    label="⬇️ 下载热力图",
                    data=convert_image_to_bytes(heatmap_img),
                    file_name="heatmap_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            with col3:
                st.markdown("#### 3. 精准定位框")
                st.image(bbox_img, use_column_width=True)
                # 定位框图下载按钮
                st.download_button(
                    label="⬇️ 下载定位框图",
                    data=convert_image_to_bytes(bbox_img),
                    file_name="bbox_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )