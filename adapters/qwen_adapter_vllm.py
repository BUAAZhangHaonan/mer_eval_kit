# adapters/qwen_adapter.py
import os
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
from typing import Dict, List, Optional
import torch

try:
    from adapters.base import BaseAdapter
except ImportError:
    class BaseAdapter:
        def predict(self, item: Dict, task: str) -> Dict:
            raise NotImplementedError

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class Adapter(BaseAdapter):
    """
    Adapter 作为 vLLM 服务的客户端。
    它不加载模型，而是通过 HTTP API 与 vLLM 服务通信。
    支持多种模态：文本、图像、音频、视频
    """

    def __init__(self, base_url: str, model_name: str, max_frames: int = 8,
                 video_strategy: str = "uniform"):
        """
        初始化 OpenAI 客户端，指向 vLLM 服务。

        Args:
            base_url: vLLM服务地址
            model_name: 模型名称
            max_frames: 视频最大帧数
            video_strategy: 视频帧采样策略 ("uniform", "keyframes")
        """
        print(f"[QwenVLLMAdapter] 初始化客户端，连接到 vLLM 服务 at {base_url}")
        # 不需要 API key，因为是本地服务
        self.client = OpenAI(base_url=base_url, api_key="g203")
        self.model_name = model_name
        self.max_frames = max_frames
        self.video_strategy = video_strategy

    def _frames_to_base64_urls(self, frames: List[Image.Image]) -> List[str]:
        """
        将 PIL.Image 帧列表转换为 Base64 编码的数据 URL。
        """
        base64_urls = []
        for frame in frames:
            buffered = BytesIO()
            frame.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_urls.append(f"data:image/jpeg;base64,{img_str}")
        return base64_urls

    def _load_image_from_path(self, image_path: str) -> Optional[Image.Image]:
        """从路径加载图像"""
        if not os.path.exists(image_path):
            return None
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None

    def _load_frames_from_video(self, video_path: str, num_frames: int = None) -> List[Image.Image]:
        """从视频文件加载帧"""
        if num_frames is None:
            num_frames = self.max_frames

        frames = []

        if not os.path.exists(video_path):
            return frames

        if _HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return frames

            # 均匀采样帧
            if self.video_strategy == "uniform":
                indices = torch.linspace(
                    0, total_frames - 1, num_frames, dtype=torch.long)
            else:  # keyframes or other strategies
                indices = torch.linspace(
                    0, total_frames - 1, num_frames, dtype=torch.long)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))

            cap.release()

        return frames

    def _process_text_input(self, item: Dict, task: str) -> Dict:
        """处理文本输入"""
        text = item.get("text", "")
        if not text:
            return {"error": "No text found"}

        # 构建prompt
        if task in ["mosei_sentiment", "chsims_sentiment"]:
            prompt = f"文本: {text}\n情感极性 (-3到3): "
        else:
            prompt = f"文本: {text}\n情感: "

        messages = [{"role": "user", "content": prompt}]

        try:
            print(f"[DEBUG] 发送请求到 {self.model_name}，消息数量: {len(messages)}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )
            print(f"[DEBUG] 收到响应: {response.choices[0].message.content.strip()!r}")
            result = response.choices[0].message.content.strip()

            # 处理thinking内容，如果有的话
            if result.startswith('<think>'):
                # 尝试找到</think>标签后的内容
                end_think = result.find('</think>')
                if end_think != -1:
                    result = result[end_think + 8:].strip()
                else:
                    # 如果没有结束标签，跳过所有thinking行
                    lines = result.split('\n')
                    content_lines = []
                    in_think = False
                    for line in lines:
                        if line.startswith('<think>'):
                            in_think = True
                            continue
                        elif line.startswith('</think>'):
                            in_think = False
                            continue
                        elif in_think:
                            continue
                        else:
                            content_lines.append(line)
                    result = '\n'.join(content_lines).strip()

            if task in ["mosei_sentiment", "chsims_sentiment"]:
                # 尝试提取数值
                import re
                numbers = re.findall(r'-?\d+\.?\d*', result)
                if numbers:
                    try:
                        polarity = float(numbers[0])
                        polarity = max(-3.0, min(3.0, polarity))
                        return {"polarity": polarity}
                    except ValueError:
                        pass
                return {"polarity": 0.0}
            else:
                return {"label": result}

        except Exception as e:
            print(f"文本处理失败: {e}")
            return {"polarity": 0.0} if task in ["mosei_sentiment", "chsims_sentiment"] else {"label": "neutral"}

    def _process_image_input(self, item: Dict, task: str) -> Dict:
        """处理图像输入"""
        image_path = item.get("image_path")
        label_space = item.get("label_space", [])

        if not image_path:
            return {"error": "No image path found"}

        image = self._load_image_from_path(image_path)
        if not image:
            return {"error": f"Failed to load image: {image_path}"}

        # 转换为base64
        base64_url = self._frames_to_base64_urls([image])[0]

        # 构建prompt
        if task == "image_emotion_class":
            # 确保label_space中的所有元素都是字符串
            label_strings = [str(label) for label in label_space]
            options = ", ".join(label_strings)
            prompt = f"从 [{options}] 中选择一个最能描述图像中人物主要情感的词。只回答这个词，不要其他内容。"
        elif task == "image_va_reg":
            prompt = "请评估图像中人物的效价(valence)和唤醒度(arousal)，格式：valence: [值], arousal: [值]，范围-1到1。只输出格式化的结果。"
        else:
            prompt = "请描述图像中的情感。只回答情感标签。"

        content = [
            {"type": "image_url", "image_url": {"url": base64_url}},
            {"type": "text", "text": prompt}
        ]

        messages = [{"role": "user", "content": content}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=50,
                temperature=0.0
            )

            result = response.choices[0].message.content.strip()

            if task == "image_emotion_class":
                # 解析分类结果
                result = result.lower().replace('.', '').replace(',', '').strip()
                for label in label_space:
                    if str(label).lower() in result:
                        return {"label": str(label)}
                return {"label": "neutral"}
            elif task == "image_va_reg":
                # 解析VA结果
                try:
                    if "valence:" in result.lower() and "arousal:" in result.lower():
                        parts = result.lower().replace('valence:', '').replace('arousal:', '').split(',')
                        valence = float(parts[0].strip())
                        arousal = float(parts[1].strip())
                        return {"valence": max(-1.0, min(1.0, valence)),
                                "arousal": max(-1.0, min(1.0, arousal))}
                except:
                    pass
                return {"valence": 0.0, "arousal": 0.0}
            else:
                return {"result": result}

        except Exception as e:
            print(f"图像处理失败: {e}")
            if task == "image_emotion_class":
                return {"label": "neutral"}
            elif task == "image_va_reg":
                return {"valence": 0.0, "arousal": 0.0}
            else:
                return {"result": ""}

    def _process_video_input(self, item: Dict, task: str) -> Dict:
        """处理视频输入"""
        video_path = item.get("video_path")
        label_space = item.get("label_space", [])
        frames = item.get("frames")  # 如果已经预加载了帧

        if not frames and video_path:
            frames = self._load_frames_from_video(video_path)

        if not frames:
            return {"error": "No video frames found"}

        # 限制帧数
        if len(frames) > self.max_frames:
            indices = torch.linspace(
                0, len(frames) - 1, self.max_frames, dtype=torch.long)
            frames = [frames[i] for i in indices]

        # 转换为base64
        base64_urls = self._frames_to_base64_urls(frames)

        # 构建prompt
        if task == "video_emotion_class":
            # 确保label_space中的所有元素都是字符串
            label_strings = [str(label) for label in label_space]
            options = ", ".join(label_strings)
            prompt = f"从 [{options}] 中选择一个最能描述视频中人物主要情感的词。只回答这个词，不要其他内容。"
        elif task == "video_va_reg":
            prompt = "请评估视频中人物的效价(valence)和唤醒度(arousal)，格式：valence: [值], arousal: [值]，范围-1到1。只输出格式化的结果。"
        else:
            prompt = "请描述视频中的情感。只回答情感标签。"

        content = []
        for url in base64_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.0
            )

            result = response.choices[0].message.content.strip()

            if task == "video_emotion_class":
                # 解析分类结果
                result = result.lower().replace('.', '').replace(',', '').strip()
                for label in label_space:
                    if str(label).lower() in result:
                        return {"label": str(label)}
                return {"label": "neutral"}
            elif task == "video_va_reg":
                # 解析VA结果
                try:
                    if "valence:" in result.lower() and "arousal:" in result.lower():
                        parts = result.lower().replace('valence:', '').replace('arousal:', '').split(',')
                        valence = float(parts[0].strip())
                        arousal = float(parts[1].strip())
                        return {"valence": max(-1.0, min(1.0, valence)),
                                "arousal": max(-1.0, min(1.0, arousal))}
                except:
                    pass
                return {"valence": 0.0, "arousal": 0.0}
            else:
                return {"result": result}

        except Exception as e:
            print(f"视频处理失败: {e}")
            if task == "video_emotion_class":
                return {"label": "neutral"}
            elif task == "video_va_reg":
                return {"valence": 0.0, "arousal": 0.0}
            else:
                return {"result": ""}

    def _process_multimodal_input(self, item: Dict, task: str) -> Dict:
        """处理多模态输入"""
        content = []
        label_space = item.get("label_space", [])

        # 处理文本
        if "text" in item and item["text"]:
            content.append({"type": "text", "text": f"文本内容：{item['text']}"})

        # 处理图像
        if "image_path" in item:
            image = self._load_image_from_path(item["image_path"])
            if image:
                base64_url = self._frames_to_base64_urls([image])[0]
                content.append(
                    {"type": "image_url", "image_url": {"url": base64_url}})

        # 处理视频帧
        if "frames" in item and item["frames"]:
            frames = item["frames"]
            if len(frames) > self.max_frames:
                indices = torch.linspace(
                    0, len(frames) - 1, self.max_frames, dtype=torch.long)
                frames = [frames[i] for i in indices]
            base64_urls = self._frames_to_base64_urls(frames)
            for url in base64_urls:
                content.append(
                    {"type": "image_url", "image_url": {"url": url}})
        elif "video_path" in item:
            frames = self._load_frames_from_video(item["video_path"])
            if frames:
                base64_urls = self._frames_to_base64_urls(frames)
                for url in base64_urls:
                    content.append(
                        {"type": "image_url", "image_url": {"url": url}})

        if not content:
            return {"error": "No valid modalities found"}

        # 构建prompt
        if task in ["meld_dialog_emotion", "emotiontalk_dialog_emotion"]:
            # 确保label_space中的所有元素都是字符串
            label_strings = [str(label) for label in label_space]
            options = ", ".join(label_strings)
            prompt = f"基于以上多模态信息，从 [{options}] 中选择一个最能描述情感状态的词。只回答这个词，不要其他内容。"
        elif task == "mosei_sentiment":
            prompt = "基于以上多模态信息，输出一个-3到3之间的情感极性数值，-3最负面，0中性，3最正面。只输出数值。"
        else:
            prompt = "请分析以上多模态信息中的情感。只回答情感标签。"

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.0
            )

            result = response.choices[0].message.content.strip()

            if task in ["meld_dialog_emotion", "emotiontalk_dialog_emotion"]:
                result = result.lower().replace('.', '').replace(',', '').strip()
                for label in label_space:
                    if str(label).lower() in result:
                        return {"label": str(label)}
                return {"label": "neutral"}
            elif task == "mosei_sentiment":
                try:
                    polarity = float(result)
                    polarity = max(-3.0, min(3.0, polarity))
                    return {"polarity": polarity}
                except ValueError:
                    return {"polarity": 0.0}
            else:
                return {"result": result}

        except Exception as e:
            print(f"多模态处理失败: {e}")
            if task in ["meld_dialog_emotion", "emotiontalk_dialog_emotion"]:
                return {"label": "neutral"}
            elif task == "mosei_sentiment":
                return {"polarity": 0.0}
            else:
                return {"result": ""}

    def predict(self, item: Dict, task: str) -> Dict:
        """
        智能选择处理方式并进行预测。
        根据item中包含的数据类型和任务类型，选择合适的处理方法。
        """
        try:
            # 检查数据类型并选择处理方法
            if "text" in item and item["text"] is not None:
                item["text"] = str(item["text"])
            if "label_space" in item and item["label_space"] is not None:
                item["label_space"] = [str(label) for label in item["label_space"]]
            has_text = "text" in item and item["text"]
            has_image = "image_path" in item and item["image_path"]
            has_video = ("video_path" in item and item["video_path"]) or (
                "frames" in item and item["frames"])

            # 统计模态数量
            # modality_count = sum([has_text, has_image, has_video])
            modality_count = 1

            # 根据模态数量和任务类型选择处理方法
            if modality_count > 1:
                # 多模态处理
                return self._process_multimodal_input(item, task)
            elif has_text:
                # 纯文本处理
                return self._process_text_input(item, task)
            elif has_image:
                # 纯图像处理
                return self._process_image_input(item, task)
            elif has_video:
                # 纯视频处理
                return self._process_video_input(item, task)
            else:
                # 没有有效数据
                print(f"警告：样本 {item.get('id')} 中没有找到有效的数据模态。")

                # 根据任务类型返回默认值
                if task in ["image_emotion_class", "video_emotion_class", "meld_dialog_emotion", "emotiontalk_dialog_emotion"]:
                    return {"label": "neutral"}
                elif task in ["image_va_reg", "video_va_reg"]:
                    return {"valence": 0.0, "arousal": 0.0}
                elif task in ["mosei_sentiment", "chsims_sentiment"]:
                    return {"polarity": 0.0}
                else:
                    return {"result": ""}

        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            # 返回默认值以防中断整个评估流程
            if task in ["image_emotion_class", "video_emotion_class", "meld_dialog_emotion", "emotiontalk_dialog_emotion"]:
                return {"label": "neutral"}
            elif task in ["image_va_reg", "video_va_reg"]:
                return {"valence": 0.0, "arousal": 0.0}
            elif task in ["mosei_sentiment", "chsims_sentiment"]:
                return {"polarity": 0.0}
            else:
                return {"result": ""}
