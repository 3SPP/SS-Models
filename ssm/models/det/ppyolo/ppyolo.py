import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class PPYOLO(nn.Layer):
    """
    backbone: resnet50vd_dcn
    """
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope=name_scope, dtype=dtype)
        
    def forward(self, x):
        pass


class PPYOLOV2(nn.Layer):
    """
    backbone: resnet50vd_dcn
    """
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope=name_scope, dtype=dtype)
        
    def forward(self, x):
        pass