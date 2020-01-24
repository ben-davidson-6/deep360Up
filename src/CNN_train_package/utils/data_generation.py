from __future__ import division

#File paths
from os.path import *

PROJECT_FILE_PATH = dirname(dirname(dirname((dirname(abspath(__file__))))))+"/"
CURRENT_FILE_PATH = dirname(abspath(__file__))+"/"
HYPER_PARAM = dirname(dirname(abspath(__file__)))+'/'
UTILS_PATH = PROJECT_FILE_PATH+"src/utils/rotation/"


##### path to LUT folder
pathToMatrices = PROJECT_FILE_PATH+"data/matrices221_10000/"


#Import rotation directory
import sys
sys.path.insert(0, UTILS_PATH)
sys.path.insert(0, HYPER_PARAM)
sys.path.insert(0,HYPER_PARAM)
import cv2
import numpy as np
import random

from CNN_train_package.utils.tools import threadsafe_generator
import keras



IMAGE_WIDTH = 442
IMAGE_HEIGHT = 221
PI = 3.141592

# this script is for RotNetDataGenerator (data generator for deep learning)
# this class generates data when you call a generator (def generate)



phiTheta = np.load(pathToMatrices+"phiTheta_10000.npy")
num_points = phiTheta.shape[0]

class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model_for_saving, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.model_for_saving = model_for_saving

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_for_saving.save_weights(filepath, overwrite=True)
                        else:
                            self.model_for_saving.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_for_saving.save_weights(filepath, overwrite=True)
                else:
                    self.model_for_saving.save(filepath, overwrite=True)

class RotNetDataGenerator(object):
    def __init__(self, gt_function,data_augmentation_fn, input_shape=None, color_mode='rgb', batch_size=64, one_hot=True,
                preprocess_func=None, rotate=True, sliced=True, flip=True, crop_center=False, crop_largest_rect=False, contrast_and_brightness=True,
                shuffle=False, noise=False, seed=None):

        self.input_shape = input_shape
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.preprocess_func = preprocess_func
        self.rotate = rotate
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect
        self.shuffle = shuffle
        self.sliced = sliced
        self.flip = flip
        self.contrast_and_brightness = contrast_and_brightness
        self.gt_function = gt_function
        self.data_augmentation_fn = data_augmentation_fn
        self.noise = noise

    @threadsafe_generator
    def generate(self, image_path):
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                np.random.shuffle(image_path)
            # Generate batches
            imax = int(len(image_path)/self.batch_size)
            for i in range(imax):
                start_index = i * self.batch_size
                end_index = (i+1) * self.batch_size
                image_path_temp = image_path[start_index:end_index]
                # Generate data
                X, y = self.__data_generation(image_path_temp)

                yield X,y

    def imageRolling(self,image):
        h, w, c = image.shape
        rolledImage = np.zeros((h, w, c), dtype=np.uint8)
        boundary = np.random.randint(1, w - 1)
        rolledImage[:, 0:w - boundary, :] = image[:, boundary:w, :]
        rolledImage[:, w - boundary:w, :] = image[:, 0:boundary, :]

        return rolledImage

    def make_noise_image(self,):
        return np.random.randint(0, 255, [221, 442, 3], dtype=np.uint8)

    def __data_generation(self, image_path_temp):
        # create array to hold the images
        batch_x = np.zeros((self.batch_size,) + self.input_shape, dtype='float32')

        # create array to hold the labels
        sample_y_len = int(len(self.gt_function( 0, 0)))
        batch_y = np.zeros((self.batch_size,sample_y_len), dtype='float32')

        for index, current_path in enumerate(image_path_temp):

            image = cv2.imread(current_path) if not self.noise else self.make_noise_image()
            # rotationIdx = random.randint(0,numPoints-1)
            rotationIdx = np.random.randint(0,num_points)
            xMapName = pathToMatrices+str("%05d"%rotationIdx)+"_x.npy"
            yMapName = pathToMatrices + str("%05d" % rotationIdx) + "_y.npy"
            xMap = np.load(xMapName)
            yMap = np.load(yMapName)
            phiGT = phiTheta[rotationIdx, 0]
            thetaGT = phiTheta[rotationIdx, 1]

            xMap = xMap*float(image.shape[0]/221)
            yMap = yMap*float(image.shape[0]/221)

            # imageRolling must be before rotation
            if bool(random.getrandbits(1)):
                image = self.imageRolling(image)

            rotatedImage = cv2.remap(image,xMap,yMap,cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)

            # rotatedImage = rotatedImage.astype(np.uint8)
            # rotatedImage = cv2.remap(rotatedImage, top_cubemap_x, top_cubemap_y, cv2.INTER_CUBIC)


            # for_check = bp.rectify_image_given_phi_theta(rotatedImage,phiGT,thetaGT)
            # cv2.imwrite(str(index)+".jpg",for_check)

            # if bool(random.getrandbits(1)):

            # rotatedImage = blur_randomly(rotatedImage)
            # if bool(random.getrandbits(1)):
            # rotatedImage = add_noise(rotatedImage)
            # if bool(random.getrandbits(1)):
            # rotatedImage = add_contrast_brightness(rotatedImage)
            rotatedImage = self.data_augmentation_fn(rotatedImage)
            # cv2.imwrite(str(index) + ".jpg", rotatedImage)

            # cv2.imwrite(str(index)+"_r.jpg",rotatedImage)

            batch_x[index] = rotatedImage
            gt_y = self.gt_function(phiGT,thetaGT)
            # print('gt_y',gt_y,'gt',phiGT,thetaGT)
            batch_y[index,:] = gt_y

        # preprocess input images
        if self.preprocess_func:
            batch_x = self.preprocess_func(batch_x)

        return batch_x, batch_y

