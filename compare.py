#!/usr/bin/python3
'''
Author: Jiaqing Lin
'''
import models
import chainer
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from chainer import serializers, cuda
import pdb


def transfrom_image(imgName):
    '''
    This function is used to obtian 10 ConvNet inputs by
    cropping and flipping four corners and the center of the frame.
    '''
    images = [None] * 1
    f = Image.open(imgName)
    image = np.asarray(f, dtype=np.float32)
    if image.ndim == 2:
        # image is greyscale
        image = image[:, :, np.newaxis]
    else:
        image = image.transpose(2, 0, 1)
    crop_size = 244
    channel, height, width = image.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size
    # Crop a region in center with no filp and rotaion.
    image = image[:, top:bottom, left:right]
    image *= (1.0 / 255.0)
    images[0] = image
    # Horizontal filp
    # images[1] = image[:, :, ::-1]
    # Vertical filp
    # images[2] = image[:, ::-1, :]
    # Rotations
    # images[3] = np.rot90(images[0], 1, axes=(-2, -1))
    # images[4] = np.rot90(images[1], 1, axes=(-2, -1))
    # images[5] = np.rot90(images[1], 2, axes=(-2, -1))
    # images[6] = np.rot90(images[1], 3, axes=(-2, -1))
    # images[7] = np.rot90(images[2], 1, axes=(-2, -1))
    # images[8] = np.rot90(images[2], 2, axes=(-2, -1))
    # images[9] = np.rot90(images[2], 3, axes=(-2, -1))
    return np.asarray(images)


def predictor(model, inputs, label):
    xp = cuda.cupy
    x = chainer.Variable(xp.asarray(inputs))
    y = model(x)
    res = y.data
    return np.mean(res)


def main():
    model = models.Spatial(101)
    serializers.load_npz('spatial.model', model)
    chainer.cuda.get_device(0).use()
    model.to_gpu()
    model.train = False

    path = 'frames/'
    classes = [join(path, f) for f in listdir(path)]
    classes.sort()
    with open('compare_result.txt', 'w') as file:
        for i, video in enumerate(classes):
            frames = [join(video, f) for f in listdir(video)]
            frames.sort()
            frames_accuracy = []
            for frame in frames:
                inputs = transfrom_image(frame)
                y = predictor(model, inputs, i)
                frames_accuracy.append(y)
            video_accuracy = sum(frames_accuracy) / len(frames)
            print(video.split('/')[-1] + '.avi ' + str(video_accuracy))
            file.write(video.split('/')[-1] + '.avi ' + str(video_accuracy) + '\n')


if __name__ == '__main__':
    main()
