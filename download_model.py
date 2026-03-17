# download_model.py
import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def download_and_save_model():
    model_id = "CIDAS/clipseg-rd64-refined"
    save_directory = "./models/clipseg"

    # 如果文件夹不存在则创建
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print(f"开始从 HuggingFace 下载模型: {model_id} ...")
    print("这可能需要几分钟，具体取决于网络速度...")

    # 下载 Processor 和 Model
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)

    # 保存到本地文件夹
    processor.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print(f"✅ 模型已成功下载并保存到本地目录: {save_directory}")
    print("以后拷贝此项目时，只要带上 models 文件夹，即可完全离线运行！")

if __name__ == "__main__":
    download_and_save_model()