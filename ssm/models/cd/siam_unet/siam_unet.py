import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SiamUnet(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope=name_scope, dtype=dtype)
        
    def forward(self, x1, x2):
        pass