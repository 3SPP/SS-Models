import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DeeplabV3(nn.Layer):
    """
    backbone: resnet50, resnet101
    """
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope=name_scope, dtype=dtype)
        
    def forward(self, x):
        pass


class DeeplabV3P(nn.Layer):
    """
    backbone: resnet50, resnet101
    """
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope=name_scope, dtype=dtype)
        
    def forward(self, x):
        pass