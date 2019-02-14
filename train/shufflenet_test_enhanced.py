#LOAD_MODEL = '../input/mobilenet-test/swa.h5'
#LOAD_MODEL='./sn_512/shufflenet-01-0.0802.model'
LOAD_MODEL=None
initial_lr=0.0005

import os, sys
import numpy as np
import pandas as pd
import skimage.io
import json

with open("PATHS.json", "r") as f:
	paths = json.load(f)

from scipy.misc import imread, imresize
from skimage.transform import resize
from tqdm import tqdm

import keras
from keras.preprocessing.image import ImageDataGenerator
sys.path.append('./keras-shufflenetV2')
from shufflenetv2 import ShuffleNetV2

from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Multiply, Input, Flatten, Dropout, Conv2D
from keras.callbacks import *
from keras.regularizers import l2
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
from imgaug import augmenters as iaa

from itertools import chain
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

USE_EXTERNAL=True
TRAIN_PART_FILE='./sn_224/train_part.csv'
VALID_PART_FILE='./sn_224/valid_part.csv'
SIZE=512

path_to_train = paths['COMP_DATA_DIR']+'data/train/'
data = pd.read_csv(paths['COMP_DATA_DIR']+'train.csv')
data['ext'] = False
print(len(data))

path_to_ext_train = paths['EXT_DATA_DIR']+'512_images/'
if USE_EXTERNAL:
	ext_data = pd.read_csv(paths['COMP_DATA_DIR']+'HPAv18RGBY_WithoutUncertain_wodpl.csv')
	print(len(ext_data))
	ext_data['ext'] = True
	data = data.append(ext_data, ignore_index=True)

train_dataset_info = []
for name, labels, ext in zip(data['Id'], data['Target'].str.split(' '), data['ext']):
    if ext:
	p = path_to_ext_train
    else:
	p = path_to_train
    train_dataset_info.append({
        'path':
        os.path.join(p, name),
        'labels':
        np.array([int(label) for label in labels])
    })
train_dataset_info = np.array(train_dataset_info)
print("1", len(train_dataset_info))

import cv2
import random
import threading


class DataGenerator:
    def __init__(self):
        self.idx_array = None
        self.idx = 0
        self.dataset_len = 0
        self.first = True
        self.threadLock = threading.Lock()
        self.augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(rotate=(-30, 30)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Crop(px=(0, 12)), # crop images from each side by 0 to 16px (randomly chosen)
                iaa.Affine(
                            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                            shear=(-4, 4)
                ),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
            ]),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10)))
        ],
                                          random_order=True)

    def create_train(self, dataset_info, batch_size, shape, augument=True):
        with self.threadLock:
            if self.first:
                self.first = False
                self.idx_array = list(range(0, len(dataset_info)))
                random.shuffle(self.idx_array)
                self.dataset_len = len(self.idx_array)

        assert shape[2] == 3
        while True:
            batch_images1 = np.empty((batch_size, shape[0], shape[1],
                                      shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            with self.threadLock:
                if (self.idx + 1) * batch_size > self.dataset_len:
                    random.shuffle(self.idx_array)
                    self.idx = 0
                batch_indexes = self.idx_array[self.idx * batch_size:
                                               (self.idx + 1) * batch_size]
                self.idx += 1
            for i in range(0, batch_size):
                image1 = self.load_image(
                    dataset_info[batch_indexes[i]]['path'], shape)
                if augument:
                    image1 = self.augment_img.augment_image(image1)
                batch_images1[i] = image1
                batch_labels[i][dataset_info[batch_indexes[i]]['labels']] = 1
            yield batch_images1, batch_labels

    def load_image(self, path, shape):
#	print(path)
        image_red_ch = cv2.imread(path + '_red.png', cv2.IMREAD_GRAYSCALE)
#        image_yellow_ch = cv2.imread(path + '_yellow.png', cv2.IMREAD_GRAYSCALE)
#        image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        image_green_ch = cv2.imread(path + '_green.png', cv2.IMREAD_GRAYSCALE)
#        kernel = np.ones((3,3),np.uint8)
#        image_green_ch = cv2.dilate(image_green_ch,kernel,iterations = 1)

        image_blue_ch = cv2.imread(path + '_blue.png', cv2.IMREAD_GRAYSCALE)

        image1 = np.stack((image_red_ch, image_green_ch, image_blue_ch), -1)
        image1 = cv2.resize(image1, (SIZE, SIZE), interpolation = cv2.INTER_AREA)
        return image1.astype(np.float)

train_datagen = DataGenerator()
valid_datagen = DataGenerator()
data['target_list'] = data['Target'].map(
    lambda x: [int(a) for a in x.split(' ')])
all_labels = list(chain.from_iterable(data['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
data['target_vec'] = data['target_list'].map(
    lambda ck: [i in ck for i in range(max_idx + 1)])
lab_vec = data['target_list'].map(lambda ck: [int(i in ck) for i in range(28)])

if LOAD_MODEL:
	train_df = pd.read_csv(TRAIN_PART_FILE)
	valid_df = pd.read_csv(VALID_PART_FILE)
else:
	import ml_stratifiers
	spl = ml_stratifiers.MultilabelStratifiedShuffleSplit(
	    n_splits=1, test_size=0.1, random_state=42)
	ti, vi = next(spl.split(np.zeros(len(data)), lab_vec.tolist()))
	print(len(vi), len(ti))
	train_df = data.ix[ti]
	valid_df = data.ix[vi]
	print(train_df.shape[0], 'training masks')
	print(valid_df.shape[0], 'validation masks')
	train_df.to_csv('train_part.csv')
	valid_df.to_csv('valid_part.csv')

# ### Create lists for training:

# In[6]:

train_dataset_info = []
for name, labels, ext in zip(train_df['Id'], train_df['Target'].str.split(' '), train_df['ext']):
    if ext:
	p = path_to_ext_train
    else:
	p = path_to_train
    train_dataset_info.append({
        'path':
        os.path.join(p, name),
        'labels':
        np.array([int(label) for label in labels])
    })
train_dataset_info = np.array(train_dataset_info)
valid_dataset_info = []
for name, labels, ext in zip(valid_df['Id'], valid_df['Target'].str.split(' '), valid_df['ext']):
    if ext:
	p = path_to_ext_train
    else:
	p = path_to_train
    valid_dataset_info.append({
        'path':
        os.path.join(p, name),
        'labels':
        np.array([int(label) for label in labels])
    })
valid_dataset_info = np.array(valid_dataset_info)
print(train_dataset_info.shape, valid_dataset_info.shape)


def create_model(input_shape, n_out):
    inp_mask = Input(shape=input_shape)
    pretrain_model_mask = ShuffleNetV2(
        input_shape=(SIZE, SIZE, 3),  #SWITCH
        include_top=False,
#        weights='imagenet',
#        dropout=0.5,
        pooling='avg')
    pretrain_model_mask.name = 'shufflenet'

    x = pretrain_model_mask(inp_mask)
    out = Dense(n_out, activation='sigmoid')(x)
    model = Model(inputs=inp_mask, outputs=[out])

    return model


import tensorflow as tf



# ### Compile model
# Compile our model, note, we will use binary_crossentropy as loss (we have sigmoid output layer and multilabel task) ans two metrics: accuracy and f1:
keras.backend.clear_session()

model = create_model(input_shape=(SIZE, SIZE, 3), n_out=28)
opt = Adam(lr=initial_lr)

model.compile(
    loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

model.summary()

if LOAD_MODEL != None:
    print("loading weights from {}".format(LOAD_MODEL))
    model.load_weights(LOAD_MODEL)
#    model.save(LOAD_MODEL+".no_f1")
#    exit(0)

#batch_size = 6  #SWITCH
batch_size = 16
checkpointer = ModelCheckpoint(
    paths['MODEL_DIR']+'sndiv-{epoch:02d}-{val_loss:.4f}.model',  #SWITCH
    verbose=2,
    save_best_only=True)

# create train and valid datagens
train_generator = train_datagen.create_train(train_dataset_info, batch_size,
                                             (SIZE, SIZE, 3))
validation_generator = valid_datagen.create_train(
    valid_dataset_info, batch_size, (SIZE, SIZE, 3), False)
class_weights= {0: 1.0, 1: 3.01, 2: 1.95, 3: 2.79, 4: 2.61, 5: 2.31, 6: 3.23, 7: 2.2, 8: 6.17, 9: 6.34, 10: 6.81, 11: 3.15, 12: 3.61, 13: 3.86, 14: 3.17, 15: 7.1, 16: 3.87, 17: 4.8, 18: 3.34, 19: 2.84, 20: 4.99, 21: 1.91, 22: 3.46, 23: 2.15, 24: 4.37, 25: 1.13, 26: 4.35, 27: 7.74}


class moshelr(Callback):
    def __init__(self,
                 lr_sched=[],
                 lr_sched_epoch_len=[]):

        if len(lr_sched) != len(lr_sched_epoch_len):
            print("lr_sched and lr_sched_epoch_len MUST be of same length")
        self.lr_sched = lr_sched
        self.lr_sched_epoch_len = lr_sched_epoch_len
        self.cycle_index = 0
        self.place_in_cycle = 0
        self.total_epoch_len = sum(lr_sched_epoch_len)

        self.history = {}

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        K.set_value(self.model.optimizer.lr, self.lr_sched[ self.cycle_index ])
        print("setting initial lr to {}".format(self.lr_sched[ self.cycle_index ]))


    def on_epoch_end(self, epoch, logs={}):
        self.place_in_cycle += 1
        if self.place_in_cycle >= self.lr_sched_epoch_len[self.cycle_index]:
            self.place_in_cycle = 0
            self.cycle_index+=1
            if self.cycle_index < self.total_epoch_len:
                K.set_value(self.model.optimizer.lr, self.lr_sched[ self.cycle_index ])
                print("setting initial lr to {}".format(self.lr_sched[ self.cycle_index ]))
            else:
                print("ending lr_schedule, lr - {}".format(K.get_value(self.model.optimizer.lr)))

moshelr_callback = moshelr(
    lr_sched=[0.001, 0.0001, 0.00001],
    lr_sched_epoch_len=[3, 13, 5]
)
csv_logger = CSVLogger('shufflenet_log.csv', append=True, separator=',')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4,
                           verbose=1, mode='auto', epsilon=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=6)


# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_df) //
    batch_size,  #because of the way the generator works it will still go over all the data 
    validation_data=validation_generator,
    validation_steps=len(valid_df) // batch_size,
    epochs=125,
    verbose=1,
    class_weight=class_weights,
    use_multiprocessing=False,
    shuffle=False,
    callbacks=[checkpointer,reduce_lr, csv_logger, early_stopping])
