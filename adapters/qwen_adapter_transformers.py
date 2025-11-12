# adapters/qwen_adapter.py

import os
import cv2
import torch
from PIL import Image
from typing import Dict, List

# 1. Import the EXACT classes from the official documentation
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# A simple BaseAdapter if the project one isn't found
try:
    from adapters.base import BaseAdapter
except ImportError:
    class BaseAdapter:
        def predict(self, item: Dict, task: str) -> Dict:
            raise NotImplementedError

class QwenAdapter(BaseAdapter):
    """
    Adapter for Qwen3-Omni, following the official README documentation.
    """

    def __init__(self, model_path: str):
        """
        Loads the model and the specialized processor.
        """
        print(f"[QwenAdapter] 正在从 {model_path} 加载模型和处理器...")
        
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Load the specific processor for Qwen3-Omni
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

        # 3. Load the specific model class
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",                      # Use "auto" or torch.bfloat16
            device_map="auto",
            attn_implementation="flash_attention_2", # Use Flash Attention for speed/memory
            trust_remote_code=True,
        )

        # 4. CRITICAL: Your model is the "Thinking" version.
        #    According to the docs, it has no audio output ("talker").
        #    Disabling it saves a significant amount of VRAM (~10GB).
        print("[QwenAdapter] 检测到 'Thinking' 模型，正在禁用 Talker 以节省显存...")
        self.model.disable_talker()
        self.model.eval()

        print(f"[QwenAdapter] 模型和处理器加载完成。")

    def _load_video_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """
        Helper function to load and sample frames from a video path.
        (Increased frames to 8 for better context)
        """
        frames = []
        if not os.path.exists(video_path):
            print(f"错误: 视频路径不存在 {video_path}")
            return frames

        if os.path.isdir(video_path):
            frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
            if not frame_files: return []
            indices = torch.linspace(0, len(frame_files) - 1, num_frames, dtype=torch.long)
            for i in indices:
                img = Image.open(frame_files[i]).convert('RGB')
                frames.append(img)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误：无法打开视频文件 {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0: return []
            indices = torch.linspace(0, total_frames - 1, num_frames, dtype=torch.long)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            cap.release()
            
        return frames

    def predict(self, item: Dict, task: str) -> Dict:
        """
        Processes input, runs inference, and parses output according to official docs.
        """
        video_path = item.get("video_path")
        label_space = item.get("label_space", [])

        if not video_path:
            return {"label": "neutral"} # Default on missing video

        images = self._load_video_frames(video_path, num_frames=8)
        if not images:
            return {"label": "neutral"} # Default on frame loading failure

        # 5. Build the conversation structure as shown in the docs
        prompt = f"分析视频中的人物表情，判断其主要情感。请从以下类别中选择一个最合适的：{', '.join(label_space)}."
        conversation = [
            {
                "role": "user",
                "content": [
                    # Add all frames as 'image' type
                    {"type": "image", "image": img} for img in images
                ] + [
                    # Add the text prompt at the end
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 6. Use the processor to prepare all inputs correctly
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # The official processor handles local PIL Images directly when passed to the main call
        inputs = self.processor(
            text=text, 
            images=images, 
            return_tensors="pt"
        ).to(self.model.device)

        # 7. Run generation
        with torch.no_grad():
            # For the Thinking model, the output is only text_ids
            outputs_tuple = self.model.generate(**inputs, max_new_tokens=20)
            text_ids = outputs_tuple[0] # <-- 提取元组的第一个元素
        
        # 8. Decode the output
        # The output includes the input tokens, so we need to slice them off
        response_text = self.processor.batch_decode(text_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        clean_response = response_text.strip().lower()
        
        # 9. Parse the output to find the label
        final_label = "neutral"
        for label in label_space:
            if label.lower() in clean_response:
                final_label = label
                break
        
        print(f"视频: {os.path.basename(video_path)}, 模型输出: '{clean_response}', 解析后标签: '{final_label}'")

        return {"label": final_label}