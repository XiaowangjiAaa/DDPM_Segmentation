# models/fp16_util.py
# 模型精度转换工具：float32 <-> float16

def convert_module_to_f16(l):
    """
    将模块转换为 float16。用于节省显存。
    """
    if hasattr(l, 'to'):
        l.to(dtype='float16')

def convert_module_to_f32(l):
    """
    将模块转换为 float32。用于保持数值稳定。
    """
    if hasattr(l, 'to'):
        l.to(dtype='float32')
