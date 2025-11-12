# adapters/qwen_adapter.py
import os
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
from typing import Dict, List

try:
    from adapters.base import BaseAdapter
except ImportError:
    class BaseAdapter:
        def predict(self, item: Dict, task: str) -> Dict:
            raise NotImplementedError


class Adapter(BaseAdapter):
    """
    Adapter 作为 vLLM 服务的客户端。
    它不加载模型，而是通过 HTTP API 与 vLLM 服务通信。
    """

    def __init__(self, base_url: str, model_name: str):
        """
        初始化 OpenAI 客户端，指向 vLLM 服务。
        """
        print(f"[QwenVLLMAdapter] 初始化客户端，连接到 vLLM 服务 at {base_url}")
        # 不需要 API key，因为是本地服务
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model_name = model_name

    def _frames_to_base64_urls(self, frames: List[Image.Image]) -> List[str]:
        """
        将 PIL.Image 帧列表转换为 Base64 编码的数据 URL。
        """
        base64_urls = []
        for frame in frames:
            buffered = BytesIO()
            frame.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_urls.append(f"data:image/jpeg;base64,{img_str}")
        return base64_urls

    def predict(self, item: Dict, task: str) -> Dict:
        """
        将样本打包成 API 请求，发送给 vLLM，并解析返回结果。
        """
        video_path = item.get("video_path")
        label_space = item.get("label_space", [])
        frames = item.get("frames")  # 假设样本迭代器已经加载了帧

        if not frames:
            print(f"警告：样本 {item.get('id')} 中没有找到'frames'。")
            return {"label": "neutral"}

        # 1. 将图像帧转换为 OpenAI API 要求的格式 (base64)
        image_urls = self._frames_to_base64_urls(frames)

        # 2. 构建 prompt 和 message 结构
        options = ", ".join(label_space)
        prompt = f"这是一个单选题。从 [{options}] 中选择一个最能描述视频中人物主要情感的词。你的回答必须只包含这一个词，不要有任何解释、思考过程或多余的文字。"

        content = []
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        try:
            # 3. 发送 API 请求到 vLLM 服务
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=5000,  # 限制生成长度，因为只需要一个词
                temperature=0.0,  # 我们需要确定性的结果
            )

            # 4. 解析 vLLM 返回的结果
            raw_response = response.choices[0].message.content.strip()

            # --- 全新的、更鲁棒的解析逻辑 ---
            final_answer = ""
            # 1. 尝试找到 </think> 标签，并提取它之后的内容
            if '</think>' in raw_response:
                # 使用正则表达式或者简单的分割来提取
                parts = raw_response.split('</think>')
                final_answer = parts[-1].strip().lower().replace('.', '')
            else:
                # 2. 如果没有 </think> 标签，就假定整个回复都是答案
                final_answer = raw_response.lower().replace('.', '')

            # 3. 在提取出的最终答案中进行精确匹配
            final_label = "neutral"  # 默认值
            for label in label_space:
                if label.lower() == final_answer:
                    final_label = label
                    break

            # 4. (备用方案) 如果精确匹配失败，可能是答案包含了额外字符，
            #    则在最终答案中进行模糊查找
            if final_label == "neutral":
                for label in label_space:
                    if label.lower() in final_answer:
                        final_label = label
                        break
            # --- 解析逻辑结束 ---

            print(
                f"视频: {os.path.basename(video_path)}, 模型原始输出: '{raw_response.replace(chr(10), ' ')}', 解析后标签: '{final_label}'")

            return {"label": final_label}

        except Exception as e:
            print(f"错误：调用 vLLM API 失败: {e}")
            # 发生错误时返回一个默认值，以防中断整个评估流程
            return {"label": "neutral"}
