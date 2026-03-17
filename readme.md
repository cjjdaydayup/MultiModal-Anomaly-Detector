# 工业异常检测平台 (Industrial Anomaly Detection Platform)

## 📖 项目简介

**工业异常检测平台** 是一个基于多模态模型的缺陷检测系统。它利用先进的多模态模型，通过自然语言提示词（如“划痕”、“裂纹”）即可直接检测工业产品图像中的异常区域，**无需针对特定产品进行任何训练**。系统完全离线运行，保障工业数据安全，并提供了直观的 Web 界面，支持结果可视化与一键下载。

🎯 适用于快速验证、小批量多品种产品质检、以及需要保护数据隐私的工业场景

## ✨ 功能特性

- 🔌 完全离线：模型本地加载，无需联网，保障生产数据不外泄。
- 🖼️ 多种输入：支持从本地文件夹选择预设图片或手动上传图片。
- 📊 结果可视化：
    + 原始图像
    + 缺陷热力图（红色区域表示异常概率高）
    + 缺陷边界框（自动筛选高置信度区域并绘制红框）
- ⬇️ 一键下载：所有结果图均可直接下载，便于存档与报告。
- 🌐 交互式界面：基于 Streamlit 构建，操作简单，响应迅速。

## 🛠️ 技术栈

- 前端/框架：Streamlit
- 深度学习模型：Hugging Face Transformers(注:后期会更新多模态训练模型来提高检测准确率)
- 后端推理：PyTorch, OpenCV, PIL, NumPy等

## 📁 项目结构
```text
industrial-ad-web/
├── app.py                     # 主程序（Web 界面）
├── download_model.py          # 模型下载脚本（首次运行需要）
├── core/                      # 核心算法模块
│   ├── __init__.py
│   ├── model_handler.py       # 本地模型加载（带缓存）
│   └── inference.py           # 缺陷检测推理逻辑
├── models/                    # 存放下载的模型文件（运行 download_model.py 后生成）
│   └── clipseg/               # 模型文件（以后新建文件夹，可以上传更新的训练模型）
├── sample_images/             # 示例图片文件夹（用户可自行添加）
│   ├── bottle_defect.jpg
│   ├── leather_scratch.jpg
│   └── ...
└── README.md                  # 本文件
```

## 🚀 快速开始

```bash
pip install streamlit transformers torch torchvision opencv-python pillow matplotlib numpy
```

**下载模型**（仅首次需要）
```bash
python download_model.py
```

**准备示例图片（可选）**
将您的测试图片放入 sample_images/ 文件夹，支持 .jpg, .jpeg, .png 格式。

**启动Web应用**
```bash
streamlit run app.py
```

## 📝 使用说明
1. 在侧边栏配置产品名称

- 输入产品的英文名称（例如：bottle, metal nut, leather, pill），这将作为文本提示词的一部分。

2. 选择图片来源

- **使用系统预设图片**：从 sample_images/ 下拉列表中选择一张测试图。

- **手动上传本地图片**：点击上传按钮选择您自己的图片。

3. 执行检测

- 确认图片预览无误后，点击 “开始智能检测” 按钮。

4. 查看与下载结果

- 系统将展示三列对比视图：原始图像、缺陷热力图、缺陷边界框。

- 每张结果图下方均有 “下载” 按钮，点击即可保存为 JPEG 文件。

## ⚙️ 自定义提示词
在 core/inference.py 的 detect_universal_anomaly 函数中，可以修改 prompts 列表，添加或删除您关心的缺陷类型。例如：
```python
prompts = [
    f"scratch on {product_name}",      # 划痕
    f"crack on {product_name}",        # 裂纹
    f"stain on {product_name}",        # 污渍
    f"dent on {product_name}",         # 凹痕（自定义）
    f"corrosion on {product_name}"     # 腐蚀（自定义）
]
```

## 📌 注意事项
- 首次运行前必须执行 python download_model.py 下载模型，否则应用会报错。
- 项目完全离线运行，模型文件一旦下载完毕，即可在无网络环境下使用。
- 当前模型文件较大（约 600 MB），请确保网络稳定