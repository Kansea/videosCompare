#!/usr/bin/python3
'''
Author: Jiaqing Lin
'''
import six
import cv2 as cv
import models
import chainer
import argparse
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from chainer import serializers, cuda
import pdb

def transfrom_video(videoName):
    '''
    This function is used to obtian frames from a video.
    '''
    xp = cuda.cupy
    crop_size = 224
    cap = cv.VideoCapture(videoName)
    num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    images = xp.zeros((num_frame, 3, crop_size, crop_size), dtype=xp.float32)
    idx = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        # Change opencv iamge to PIL image.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        data = Image.fromarray(frame)
        image = xp.asarray(data, dtype=xp.float32)
        image = image.transpose(2, 0, 1)
        channel, height, width = image.shape
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]
        image *= (1.0 / 255.0)
        images[idx, :, :, :] = image
        idx += 1
    cap.release()
    return images


def predictor(model, inputs):
    xp = cuda.cupy
    batchsize = 128
    data = xp.empty((0, 2048), xp.float32)
    for i in six.moves.range(0, len(inputs), batchsize):
        x = chainer.Variable(xp.asarray(inputs[i:i + batchsize]))
        y = model(x)
        data = xp.vstack((data, y.data))
    array_sum = np.sum(data, axis=0)
    array_sum /= len(inputs)
    return array_sum


def main():
    parser = argparse.ArgumentParser(
            description='\x1b[5;30;47mMake Vector For Videos\x1b[0m',
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('root', help='''root path of video files.''')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='''GPU ID''') 
    args = parser.parse_args()

    chainer.cuda.get_device(args.gpu).use()
    model = models.Spatial(101)
    serializers.load_npz('spatial.model', model)
    model.train = False
    model.to_gpu()
    
    path = args.root
    classes = [join(path, f) for f in listdir(path)]
    classes.sort()
    with open(path.split('/')[-1] + '_vector_list.csv', 'w') as file:
        for i, label in enumerate(classes):
            videos = [join(label, f) for f in listdir(label) if f.endswith('.mp4')]
            videos.sort()
            for video in videos:
                column1 = join(video.split('/')[-2], video.split('/')[-1])
                inputs = transfrom_video(video)
                vec = predictor(model, inputs)
                tmp = cuda.cupy.asnumpy(vec)
                # tmp = tmp.reshape(1, -1)
                file.write('{},{}\n'.format(column1, ' '.join(['{}'.format(f) for f in tmp])))
            print('{} {}'.format(i, label.split('/')[-1]))


if __name__ == '__main__':
    main()
