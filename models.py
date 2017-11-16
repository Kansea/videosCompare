'''
Author: Jiaqing Lin
'''
from __future__ import print_function
import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Spatial(Chain):
    """
    Input frame size is (3 x 224 x224).
    """
    def __init__(self, n_classes):
        super(Spatial, self).__init__(
            conv1=L.Convolution2D(in_channels=3, out_channels=96, ksize=7, stride=2),
            conv2=L.Convolution2D(in_channels=96, out_channels=256, ksize=5, stride=2),
            conv3=L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1),
            conv4=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1),
            conv5=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1),
            fc6=L.Linear(in_size=None, out_size=4096),
            fc7=L.Linear(in_size=4096, out_size=2048),
            fc8=L.Linear(in_size=2048, out_size=n_classes)
        )
        self.train = True

    def __call__(self, x, t=None):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), ksize=2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), ksize=2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), ksize=2, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), ratio=0.9)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.9)
        y = self.fc8(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(y, t)
            self.accuracy = F.accuracy(y, t)
            chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
            return self.loss
        else:
            return h


class Temporal(Chain):
    """
    Input frame size is (20 x 224 x224).
    """
    def __init__(self, n_classes):
        super(Temporal, self).__init__(
            conv1=L.Convolution2D(in_channels=20, out_channels=96, ksize=7, stride=2),
            conv2=L.Convolution2D(in_channels=96, out_channels=256, ksize=5, stride=2),
            conv3=L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1),
            conv4=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1),
            conv5=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1),
            fc6=L.Linear(in_size=None, out_size=4096),
            fc7=L.Linear(in_size=4096, out_size=2048),
            fc8=L.Linear(in_size=2048, out_size=n_classes)
        )
        self.train = True

    def __call__(self, x, t=None):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), ksize=2, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        y = self.fc8(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(y, t)
            self.accuracy = F.accuracy(y, t)
            chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
            return self.loss
        else:
            return y
