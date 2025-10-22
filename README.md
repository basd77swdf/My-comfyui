# My-comfyui
import torch
import numpy as np
from PIL import Image
import comfy.utils
import nodes

# 1. 图像加载节点
class LoadImages:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "product_image": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "load"
    CATEGORY = "产品替换"
    
    def load(self, person_image, product_image):
        return (person_image, product_image)

# 2. 手部检测与分割节点
class HandSegmentation:
    def __init__(self):
        self.segmenter = nodes.VideoFrameExtractor()  # 实际应用中替换为手部分割模型
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "segment"
    CATEGORY = "产品替换"
    
    def segment(self, person_image):
        # 简化实现，实际应使用手部检测模型
        mask = torch.zeros_like(person_image[:, :, :, 0], dtype=torch.float32)
        # 假设手部区域在图像下方中央
        h, w = mask.shape[1], mask.shape[2]
        mask[:, h//2:, w//3:w*2//3] = 1.0  # 创建手部区域掩码
        return (person_image, mask)

# 3. 产品姿态调整节点
class ProductTransform:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "product_image": ("IMAGE",),
                "rotation": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    CATEGORY = "产品替换"
    
    def transform(self, product_image, rotation, scale):
        # 简化实现，实际应使用图像变换库
        transformed = comfy.utils.common_upscale(
            product_image.movedim(-1, 1),
            int(product_image.shape[2] * scale),
            int(product_image.shape[1] * scale),
            "bilinear",
            False
        ).movedim(1, -1)
        return (transformed,)

# 4. 图像合成节点
class ImageCompositor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_image": ("IMAGE",),
                "hand_mask": ("MASK",),
                "product_image": ("IMAGE",),
                "x_offset": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compose"
    CATEGORY = "产品替换"
    
    def compose(self, person_image, hand_mask, product_image, x_offset, y_offset):
        # 确保所有图像尺寸匹配
        h, w = person_image.shape[1], person_image.shape[2]
        product = comfy.utils.resize_image(product_image, w, h)
        
        # 创建产品放置掩码
        product_mask = torch.ones_like(product[:, :, :, 0], dtype=torch.float32)
        
        # 应用偏移
        hand_mask_shifted = torch.roll(hand_mask, shifts=(y_offset, x_offset), dims=(1, 2))
        
        # 合成图像：用产品替换手部区域
        result = person_image * (1 - hand_mask_shifted.unsqueeze(-1)) + \
                 product * hand_mask_shifted.unsqueeze(-1)
        
        return (result,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoadImages": LoadImages,
    "HandSegmentation": HandSegmentation,
    "ProductTransform": ProductTransform,
    "ImageCompositor": ImageCompositor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImages": "加载人物与产品图像",
    "HandSegmentation": "手部区域分割",
    "ProductTransform": "产品姿态调整",
    "ImageCompositor": "图像合成"
}
