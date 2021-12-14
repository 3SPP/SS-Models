import paddle.nn as nn


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input