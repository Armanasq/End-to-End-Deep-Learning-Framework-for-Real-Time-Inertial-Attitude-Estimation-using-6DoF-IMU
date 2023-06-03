import tensorflow_addons as tfa
import keras.backend as K
from tensorflow.keras import backend as K

from tcn import TCN
import tensorflow as tf

from keras.layers import *
from tensorflow import keras

from keras.initializers import *
from keras.utils import *
from keras.callbacks import *
from keras.regularizers import *
from keras.applications import *
from keras.losses import *
from keras.models import *
from keras.optimizers import *


import matplotlib.pyplot as plt
from fileinput import filename
from numba import cuda
from sklearn.utils import shuffle
import pandas as pd
import random as rn
import numpy as np
import math
import time
import os
from symbol import import_from
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

# arch_6_model_checkpoint.hdf5


def arch_1_model(window_size):
    Acc = Input(shape=(window_size, 3))
    Gyro = Input(shape=(window_size, 3))
    Fs = Input((1,), name='Fs')

    # normalizing the data
    acc = UnitNormalization(axis=2)(Acc)
    acc = GaussianNoise(0.1)(Acc)
    BiAcc = Bidirectional(CuDNNLSTM(128, return_sequences=True))(acc)
    #BiAcc = Dropout(0.2)(BiAcc)
    BiAcc = tfa.activations.mish(BiAcc)
    #BiAcc = Flatten()(BiAcc)

    gyro = UnitNormalization(axis=2)(Gyro)
    gyro = GaussianNoise(0.1)(Gyro)
    BiGyro = Bidirectional(CuDNNLSTM(128, return_sequences=True))(gyro)
    #BiGyro = Dropout(0.2)(BiGyro)
    BiGyro = tfa.activations.mish(BiGyro)
    #BiGyro = Flatten()(BiGyro)

    conc = concatenate([BiAcc, BiGyro])
    conc = SpatialDropout1D(0.2)(conc)
    conc = Flatten()(conc)
    D1 = Dense(256, activation=tfa.activations.mish)(conc)
    # Fs = 1/Fs
    FsD = Lambda(lambda x: 1 / x)(Fs)
    FsD = Dense(256, activation=tfa.activations.mish)(FsD)

    conc2 = concatenate([D1, FsD])
    quat = Dense(4, activation='linear')(conc2)
    quat = UnitNormalization()(quat)

    model = Model(inputs=[Acc, Gyro, Fs], outputs=quat)
    model.summary()
    return model


# Base Model


def Base_arch_1_model(window_size):
    Acc = Input(shape=(window_size, 3))
    Gyro = Input(shape=(window_size, 3))
    Fs = Input((1,), name='Fs')

    BiAcc = Bidirectional(CuDNNLSTM(128, return_sequences=True))(Acc)
    BiAcc = tfa.activations.mish(BiAcc)
    BiAcc = Flatten()(BiAcc)

    BiGyro = Bidirectional(CuDNNLSTM(128, return_sequences=True))(Gyro)
    BiGyro = tfa.activations.mish(BiGyro)
    BiGyro = Flatten()(BiGyro)

    conc = concatenate([BiAcc, BiGyro])
    D1 = Dense(256, activation=tfa.activations.mish)(conc)

    FsD = Dense(256, activation=tfa.activations.mish)(Fs)

    conc2 = concatenate([D1, FsD])
    quat = Dense(4, activation='linear')(conc2)

    model = Model(inputs=[Acc, Gyro, Fs], outputs=quat)
    model.summary()
    return model
#######################################################################################################################

# model_checkpoint copy


def arch_2_model(window_size):
    acc = Input((window_size, 3), name='acc')
    gyro = Input((window_size, 3), name='gyro')
    Acc = GaussianNoise(0.1)(acc)
    Acc = Lambda(lambda x: k.l2_normalize(x, axis=2), name='GyroNorm')(Acc)

    Gyro = GaussianNoise(0.1)(gyro)
    # normalizing the data
    Gyro = Lambda(lambda x: k.l2_normalize(x, axis=2), name='GyroNorm')(gyro)

    AGconcat = concatenate([Acc, Gyro], axis=2, name='AGconcat')

    fs = Input((1,), name='fs')
    ACNN1 = Conv1D(filters=133,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='ACNN')(Acc)
    ACNN2 = Conv1D(filters=109,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='ACNN1')(ACNN1)
    ACNN3 = MaxPooling1D(pool_size=3,
                         name='MaxPooling1D')(ACNN2)

    GCNN1 = Conv1D(filters=142,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='GCNN')(Gyro)
    GCNN2 = Conv1D(filters=116,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='GCNN1')(GCNN1)
    GCNN3 = MaxPooling1D(pool_size=3,
                         name='GyroMaxPool1D')(GCNN2)

    AG1 = concatenate([ACNN1, GCNN1])
    AG1 = Conv1D(filters=116,
                 kernel_size=11,
                 padding='same',
                 activation=tfa.activations.mish)(AG1)
    AG1 = MaxPooling1D(pool_size=3)(AG1)
    AG1 = Flatten(name='AG1F')(AG1)

    AGconcat = Conv1D(filters=128,
                      kernel_size=11,
                      padding='same',
                      activation=tfa.activations.mish)(AGconcat)

    AG2 = concatenate([ACNN3, GCNN3])
    AG2 = Conv1D(filters=128,
                 kernel_size=11,
                 padding='same',
                 activation=tfa.activations.mish)(AG2)
    AG2 = Flatten(name='AG2F')(AG2)

    AGconLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True,
                                        # return_state=True,
                                        go_backwards=True,
                                        name='BiLSTM1'))(AGconcat)
    AGconLSTM = tfa.activations.mish(AGconLSTM)
    FlattenAG = Flatten(name='FlattenAG')(AGconLSTM)
    AG = concatenate([AG1, AG2, FlattenAG])
    AG = Dense(units=256,
               activation=tfa.activations.mish)(AG)
    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AG)
    x = concatenate([AG, Fdense])
    # x = Dropout(0.15, name='Dropout')(x)
    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    output = Dense(4, activation='linear', name='quat')(x)
    #output = UnitNormalization()(output)

    model = Model(inputs=[acc, gyro, fs], outputs=output)
    model.summary()
    return model

# Base model


def Base_arch_2_model(window_size):
    Acc = Input((window_size, 3), name='acc')
    Gyro = Input((window_size, 3), name='gyro')

    AGconcat = concatenate([Acc, Gyro], axis=2, name='AGconcat')

    fs = Input((1,), name='fs')
    ACNN1 = Conv1D(filters=133,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='ACNN')(Acc)
    ACNN2 = Conv1D(filters=109,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='ACNN1')(ACNN1)
    ACNN3 = MaxPooling1D(pool_size=3,
                         name='MaxPooling1D')(ACNN2)

    GCNN1 = Conv1D(filters=142,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='GCNN')(Gyro)
    GCNN2 = Conv1D(filters=116,
                   kernel_size=11,
                   padding='same',
                   activation=tfa.activations.mish,
                   name='GCNN1')(GCNN1)
    GCNN3 = MaxPooling1D(pool_size=3,
                         name='GyroMaxPool1D')(GCNN2)

    AG1 = concatenate([ACNN1, GCNN1])
    AG1 = Conv1D(filters=116,
                 kernel_size=11,
                 padding='same',
                 activation=tfa.activations.mish)(AG1)
    AG1 = MaxPooling1D(pool_size=3)(AG1)
    AG1 = Flatten(name='AG1F')(AG1)

    AGconcat = Conv1D(filters=128,
                      kernel_size=11,
                      padding='same',
                      activation=tfa.activations.mish)(AGconcat)

    AG2 = concatenate([ACNN3, GCNN3])
    AG2 = Conv1D(filters=128,
                 kernel_size=11,
                 padding='same',
                 activation=tfa.activations.mish)(AG2)
    AG2 = Flatten(name='AG2F')(AG2)

    AGconLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True,
                                        # return_state=True,
                                        go_backwards=True,
                                        name='BiLSTM1'))(AGconcat)
    AGconLSTM = tfa.activations.mish(AGconLSTM)
    FlattenAG = Flatten(name='FlattenAG')(AGconLSTM)
    AG = concatenate([AG1, AG2, FlattenAG])
    AG = Dense(units=256,
               activation=tfa.activations.mish)(AG)
    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AG)
    x = concatenate([AG, Fdense])
    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    output = Dense(4, activation='linear', name='quat')(x)

    model = Model(inputs=[acc, gyro, fs], outputs=output)
    model.summary()
    return model

##########################################################################################################################

# model_checkpoint copy


def arch_3_model(window_size):
    Acc = Input((window_size, 3), name='acc')
    Gyro = Input((window_size, 3), name='gyro')

    AG1 = concatenate([Acc, Gyro], axis=2, name='AGconcat')
    AG1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AG1)
    AG1 = tfa.activations.mish(AG1)
    AG1 = Flatten()(AG1)

    fs = Input((1,), name='fs')
    ACNN = Conv1D(filters=133,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN')(Acc)
    ACNN = Conv1D(filters=109,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN1')(ACNN)
    ACNN = MaxPooling1D(pool_size=3,
                        name='MaxPooling1D')(ACNN)
    ACNN = Flatten(name='ACNNF')(ACNN)

    GCNN = Conv1D(filters=142,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN')(Gyro)
    GCNN = Conv1D(filters=116,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN1')(GCNN)
    GCNN = MaxPooling1D(pool_size=3,
                        name='GyroMaxPool1D')(GCNN)
    GCNN = Flatten(name='GCNNF')(GCNN)

    AG = concatenate([ACNN, GCNN, AG1])
    AG = Dense(units=100,
               activation=tfa.activations.mish)(AG)
    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AG)
    x = concatenate([AG, Fdense])
    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    output = Dense(4, activation='linear', name='quat')(x)
    model = Model(inputs=[Acc, Gyro, fs], outputs=output)
    model.summary()
    return model


def Base_arch_3_model(window_size):
    Acc = Input((window_size, 3), name='acc')
    Gyro = Input((window_size, 3), name='gyro')

    AG1 = concatenate([Acc, Gyro], axis=2, name='AGconcat')
    AG1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AG1)
    AG1 = tfa.activations.mish(AG1)
    AG1 = Flatten()(AG1)

    fs = Input((1,), name='fs')
    ACNN = Conv1D(filters=133,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN')(Acc)
    ACNN = Conv1D(filters=109,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN1')(ACNN)
    ACNN = MaxPooling1D(pool_size=3,
                        name='MaxPooling1D')(ACNN)
    ACNN = Flatten(name='ACNNF')(ACNN)

    GCNN = Conv1D(filters=142,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN')(Gyro)
    GCNN = Conv1D(filters=116,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN1')(GCNN)
    GCNN = MaxPooling1D(pool_size=3,
                        name='GyroMaxPool1D')(GCNN)
    GCNN = Flatten(name='GCNNF')(GCNN)

    AG = concatenate([ACNN, GCNN, AG1])
    AG = Dense(units=100,
               activation=tfa.activations.mish)(AG)
    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AG)
    x = concatenate([AG, Fdense])
    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    output = Dense(4, activation='linear', name='quat')(x)
    model = Model(inputs=[Acc, Gyro, fs], outputs=output)
    model.summary()
    return model


######

# GRU model

def GRU_AGF(window_size):
    acc = Input((window_size, 3), name='acc')
    gyro = Input((window_size, 3), name='gyro')
    fs = Input((1,), name='fs')
    #Fs = Reshape((1, 1))(fs)
    # Fs = 1/Fs
    Fs = Lambda(lambda x: 1 / x)(fs)
    # repeat Fs 3 times
    Fs = L

    conc = concatenate([acc, gyro, Fs], axis=1)
    GRU1 = Bidirectional(LSTM(200, return_sequences=True))(conc)
    GRU2 = Bidirectional(LSTM(200, return_sequences=True))(GRU1)
    GRU2 = Flatten()(GRU2)
    quat = Dense(4, activation='linear', name='quat')(GRU2)
    # normalized to a Euclidean norm of 1
    quat = Lambda(lambda x: K.l2_normalize(x, axis=1))(quat)

    model = Model(inputs=[acc, gyro, fs], outputs=quat)
    model.summary()
    return model


# Best
def arch_5_model(window_size):
    Acc = Input((window_size, 3), name='Acc')
    Acc = Lambda(lambda x: tf.math.l2_normalize(
        x, epsilon=0, axis=2), name='AccNorm')(Acc)
    Acc = GaussianNoise(0.1)(Acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')
    # concAcc = Dropout(0.2)(concAcc)

    Gyro = Input((window_size, 3), name='Gyro')
    Gyro = Lambda(lambda x: tf.math.l2_normalize(
        x, epsilon=0, axis=2), name='GyroNorm')(Gyro)
    Gyro = GaussianNoise(0.1)(Gyro)

    conc = Conv1D(128, 11, padding="causal",
                  activation=tfa.activations.mish, name='Conv1')(concAcc)

    conc = Dense(100, activation=tfa.activations.mish)(conc)

    gyro = Reshape(target_shape=(3, window_size))(Gyro)
    conc = concatenate([conc, Gyro], axis=1)
    BiLSTM = Flatten()(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(512, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: tf.math.l2_normalize(
        x, epsilon=0, axis=1), name='QuatNorm')(quat)
    model = Model(inputs=[Acc, Gyro, fs], outputs=[quat])
    model.summary()
    return model


def arch_4_model(window_size):
    acc = Input((window_size, 3), name='Acc')
    Acc = Lambda(lambda x: K.l2_normalize(x, axis=2), name='AccNorm')(acc)
    Acc = GaussianNoise(0.1)(Acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')
    # concAcc = Dropout(0.2)(concAcc)

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = Lambda(lambda x: K.l2_normalize(x, axis=2), name='GyroNorm')(gyro)
    Gyro = GaussianNoise(0.1)(Gyro)

    AGconc = concatenate([Acc, Gyro])
    AGconc = Bidirectional(LSTM(128, return_sequences=True))(AGconc)
    AGconc = Dropout(0.2)(AGconc)
    AGconc = tfa.activations.mish(AGconc)
    AGconc = Dense(512, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)
    #conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="causal",
                  activation=tfa.activations.mish, name='Conv1')(concAcc)

    # BiLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True))(conc)
    # BiLSTM = Dropout(0.2)(BiLSTM)

    conc = Dense(100, activation=tfa.activations.mish)(conc)

    Gyro = Reshape(target_shape=(3, window_size))(Gyro)
    conc = concatenate([conc, Gyro], axis=1)
    BiLSTM = Flatten()(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(512, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense, AGconc])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: k.l2_normalize(x, axis=-1), name='QuatNorm')(quat)

    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model


def BestTest(window_size):
    acc = Input((window_size, 3), name='Acc')
    Acc = Lambda(lambda x: K.l2_normalize(x, axis=2), name='AccNorm')(acc)
    Acc = GaussianNoise(0.1)(Acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')
    # concAcc = Dropout(0.2)(concAcc)

    gyro = Input((window_size, 3), name='Gyro')
    #Gyro = Lambda(lambda x: K.l2_normalize(x, axis=2), name='GyroNorm')(gyro)
    Gyro = GaussianNoise(0.1)(gyro)
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)

    conv1Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv1Gyro1')(Gyro1)
    pool1gyro = MaxPooling1D(3, name='MaxPool1Gyro')(conv1Gyro)

    conv2Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv2Gyro1')(Gyro2)
    pool2gyro = MaxPooling1D(3, name='MaxPool2Gyro')(conv2Gyro)

    conv3Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv3Gyro1')(Gyro3)
    pool3gyro = MaxPooling1D(3, name='MaxPool3Gyro')(conv3Gyro)

    concGyro = concatenate(
        [pool1gyro, pool2gyro, pool3gyro], name='ConcatenateCNN2')
    # concGyro = Dropout(0.2)(concGyro)

    AGconc = concatenate([acc, Gyro])
    AGconc = Bidirectional(LSTM(128, return_sequences=True))(AGconc)
    AGconc = Dropout(0.2)(AGconc)

    AGconc = Dense(512, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)
    conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="causal",
                  activation=tfa.activations.mish, name='Conv1')(conc)

    # BiLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True))(conc)
    # BiLSTM = Dropout(0.2)(BiLSTM)

    BiLSTM = Dense(512, activation=tfa.activations.mish)(conc)

    BiLSTM = Flatten()(BiLSTM)

    fs = Input((1,), name='Fs')
    fsDense = Dense(512, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense, AGconc])

    quat = Dense(4, activation="linear", name='Quat')(concat)
    #quat = Lambda(lambda x: K.l2_normalize(x), name='QuatNorm')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model
BestTest(100)
# plot the model
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# BestTest(100).save('BestTest.h5')


def BestAlt(window_size):
    acc = Input((window_size, 3), name='Acc')
    # Acc = acc
    Acc = Lambda(lambda x: k.l2_normalize(x, axis=2), name='AccNorm')(acc)
    Acc = GaussianNoise(0.1)(Acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')
    # concAcc = Dropout(0.2)(concAcc)

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = Lambda(lambda x: k.l2_normalize(x, axis=2), name='GyroNorm')(gyro)
    Gyro = GaussianNoise(0.1)(Gyro)
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)

    conv1Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv1Gyro1')(Gyro1)
    pool1gyro = MaxPooling1D(3, name='MaxPool1Gyro')(conv1Gyro)

    conv2Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv2Gyro1')(Gyro2)
    pool2gyro = MaxPooling1D(3, name='MaxPool2Gyro')(conv2Gyro)

    conv3Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv3Gyro1')(Gyro3)
    pool3gyro = MaxPooling1D(3, name='MaxPool3Gyro')(conv3Gyro)

    concGyro = concatenate(
        [pool1gyro, pool2gyro, pool3gyro], name='ConcatenateCNN2')
    # concGyro = Dropout(0.2)(concGyro)

    AGconc = concatenate([Acc, Gyro])
    AGconc = Bidirectional(LSTM(128, return_sequences=True))(AGconc)
    AGconc = Dropout(0.2)(AGconc)
    AGconc = tfa.activations.mish(AGconc)
    AGconc = Dense(512, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)
    conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="causal",
                  activation=tfa.activations.mish, name='Conv1')(conc)

    # BiLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True))(conc)
    # BiLSTM = Dropout(0.2)(BiLSTM)

    BiLSTM = Dense(512, activation=tfa.activations.mish)(conc)

    BiLSTM = Flatten()(BiLSTM)

    fs = Input((1,), name='Fs')
    fsDense = Dense(512, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense, AGconc])

    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: k.l2_normalize(x), name='QuatNorm')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model


def Best_base(window_size):
    acc = Input((window_size, 3), name='Acc')
    Acc = Lambda(lambda x: k.l2_normalize(x, axis=1), name='AccNorm')(acc)
    Acc = GaussianNoise(0.1)(Acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = Lambda(lambda x: k.l2_normalize(x, axis=1), name='GyroNorm')(gyro)
    Gyro = GaussianNoise(0.1)(Gyro)
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)

    conv1Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv1Gyro1')(Gyro1)
    pool1gyro = MaxPooling1D(3, name='MaxPool1Gyro')(conv1Gyro)

    conv2Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv2Gyro1')(Gyro2)
    pool2gyro = MaxPooling1D(3, name='MaxPool2Gyro')(conv2Gyro)

    conv3Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv3Gyro1')(Gyro3)
    pool3gyro = MaxPooling1D(3, name='MaxPool3Gyro')(conv3Gyro)

    concGyro = concatenate(
        [pool1gyro, pool2gyro, pool3gyro], name='ConcatenateCNN2')
    AGconc = concatenate([Acc, Gyro])
    AGconc = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AGconc)
    AGconc = Dense(512, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)
    conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="causal",
                  activation=tfa.activations.mish, name='Conv1')(conc)

    # BiLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True))(conc)
    # BiLSTM = Dropout(0.2)(BiLSTM)

    BiLSTM = Dense(512, activation=tfa.activations.mish)(conc)

    BiLSTM = Flatten()(BiLSTM)

    fs = Input((1,), name='Fs')
    fsDense = Dense(512, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense, AGconc])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model


def Best_base0(window_size):
    Acc = Input((window_size, 3), name='Acc')
    Acc = Lambda(lambda x: k.l2_normalize(x, axis=1), name='AccNorm')(Acc)
    Acc = GaussianNoise(0.1)(Acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="causal",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')

    Gyro = Input((window_size, 3), name='Gyro')
    Gyro = Lambda(lambda x: k.l2_normalize(x, axis=1), name='GyroNorm')(Gyro)
    Gyro = GaussianNoise(0.1)(Gyro)
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)

    conv1Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv1Gyro1')(Gyro1)
    pool1gyro = MaxPooling1D(3, name='MaxPool1Gyro')(conv1Gyro)

    conv2Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv2Gyro1')(Gyro2)
    pool2gyro = MaxPooling1D(3, name='MaxPool2Gyro')(conv2Gyro)

    conv3Gyro = Conv1D(128, 11, padding="causal",
                       activation=tfa.activations.mish, name='Conv3Gyro1')(Gyro3)
    pool3gyro = MaxPooling1D(3, name='MaxPool3Gyro')(conv3Gyro)

    concGyro = concatenate(
        [pool1gyro, pool2gyro, pool3gyro], name='ConcatenateCNN2')
    AGconc = concatenate([Acc, Gyro])
    AGconc = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AGconc)
    AGconc = Dense(512, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)
    conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="causal",
                  activation=tfa.activations.mish, name='Conv1')(conc)

    # BiLSTM = Bidirectional(CuDNNLSTM(128, return_sequences=True))(conc)
    # BiLSTM = Dropout(0.2)(BiLSTM)

    BiLSTM = Dense(512, activation=tfa.activations.mish)(conc)

    BiLSTM = Flatten()(BiLSTM)

    fs = Input((1,), name='Fs')
    fsDense = Dense(512, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense, AGconc])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    model = Model(inputs=[Acc, Gyro, fs], outputs=[quat])
    model.summary()
    return model

def arch_61(window_size):
    acc = Input((window_size, 3), name='Acc')
    Acc = Lambda(lambda x: k.l2_normalize(x, axis=2), name='AccNorm')(acc)
    Acc = GaussianNoise(0.1)(Acc)
    
    gyro = Input((window_size, 3), name='Gyro')
    Gyro = Lambda(lambda x: k.l2_normalize(x, axis=2), name='GyroNorm')(gyro)
    Gyro = GaussianNoise(0.1)(Gyro)

    BiLSTMA = Bidirectional(LSTM(128, return_sequences=True))(Acc)
    BiLSTMA = tfa.activations.mish(BiLSTMA)
    BiLSTMA = Flatten()(BiLSTMA)

    BiLSTMG = Bidirectional(LSTM(128, return_sequences=True))(Gyro)
    BiLSTMG = tfa.activations.mish(BiLSTMG)
    BiLSTMG = Flatten()(BiLSTMG)

    BiLSTM = concatenate([BiLSTMA, BiLSTMG])
    BiLSTM = Dense(256, activation=tfa.activations.mish)(BiLSTM)
    fs = Input((1,), name='Fs')
    fsDense = Dense(256, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: K.l2_normalize(x), name='QuatNorm')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model


def Barch_6(window_size):
    Acc = Input((window_size, 3), name='Acc')
    Gyro = Input((window_size, 3), name='Gyro')
    acc = GaussianNoise(0.1)(Acc)
    BiLSTMA = Bidirectional(LSTM(128, return_sequences=True))(acc)
    BiLSTMA = tfa.activations.mish(BiLSTMA)
    BiLSTMA = Flatten()(BiLSTMA)

    gyro = GaussianNoise(0.1)(Gyro)
    BiLSTMG = Bidirectional(LSTM(128, return_sequences=True))(gyro)
    BiLSTMG = tfa.activations.mish(BiLSTMG)
    BiLSTMG = Flatten()(BiLSTMG)

    BiLSTM = concatenate([BiLSTMA, BiLSTMG])
    BiLSTM = Dense(256, activation=tfa.activations.mish)(BiLSTM)
    fs = Input((1,), name='Fs')
    fsDense = Dense(256, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: K.l2_normalize(x), name='QuatNorm')(quat)

    model = Model(inputs=[Acc, Gyro, fs], outputs=[quat])
    model.summary()
    return model