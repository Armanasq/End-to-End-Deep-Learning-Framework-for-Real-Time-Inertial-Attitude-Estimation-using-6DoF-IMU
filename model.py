import tensorflow_addons as tfa
import keras.backend as K
from tensorflow.keras import backend as k

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
# Shuffling the data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['PYTHONHASHSEED'] = '0'
class AttitudeEstimationPINN(Model):
    def __init__(self):
        super(AttitudeEstimationPINN, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(64, activation='relu')
        self.dense4 = Dense(4, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        outputs = self.dense4(x)
        return outputs



def AttLayer(q):
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    roll = tf.math.atan2(2*(w*x + y*z),
                       (1-2*(x*x + y*y)))
    pitch = tf.math.asin(2*(w*y - x*z))
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    qx = (tf.math.sin(roll/2) * tf.math.cos(pitch/2) * tf.math.cos(zero_float64) -
          tf.math.cos(roll/2) * tf.math.sin(pitch/2) * tf.math.sin(zero_float64))
    qx = tf.reshape(qx, (tf.shape(roll)[0], 1))
    qy = (tf.math.cos(roll/2) * tf.math.sin(pitch/2) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.cos(pitch/2) * tf.math.sin(zero_float64))
    qy = tf.reshape(qy, (tf.shape(roll)[0], 1))
    qz = (tf.math.cos(roll/2) * tf.math.cos(pitch/2) * tf.math.sin(zero_float64) -
          tf.math.sin(roll/2) * tf.math.sin(pitch/2) * tf.math.cos(zero_float64))
    qz = tf.reshape(qz, (tf.shape(roll)[0], 1))
    qw = (tf.math.cos(roll/2) * tf.math.cos(pitch/2) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.sin(pitch/2) * tf.math.sin(zero_float64))
    qw = tf.reshape(qw, (tf.shape(roll)[0], 1))
    quat = tf.concat([qw, qx, qy, qz], axis=-1)
    return quat

def Model_A(window_size):
    D = 256
    Gn = 0.25
    acc = Input((window_size, 3), name='Acc')
    Acc = GaussianNoise(Gn)(acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="same",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="same",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="same",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = GaussianNoise(Gn)(gyro)
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)

    conv1Gyro = Conv1D(128, 11, padding="same",
                       activation=tfa.activations.mish, name='Conv1Gyro1')(Gyro1)
    pool1gyro = MaxPooling1D(3, name='MaxPool1Gyro')(conv1Gyro)

    conv2Gyro = Conv1D(128, 11, padding="same",
                       activation=tfa.activations.mish, name='Conv2Gyro1')(Gyro2)
    pool2gyro = MaxPooling1D(3, name='MaxPool2Gyro')(conv2Gyro)

    conv3Gyro = Conv1D(128, 11, padding="same",
                       activation=tfa.activations.mish, name='Conv3Gyro1')(Gyro3)
    pool3gyro = MaxPooling1D(3, name='MaxPool3Gyro')(conv3Gyro)

    concGyro = concatenate(
        [pool1gyro, pool2gyro, pool3gyro], name='ConcatenateCNN2')
    AGconc = concatenate([Acc, Gyro])
    AGconc = Bidirectional(LSTM(128, return_sequences=True))(AGconc)
    AGconc = Dropout(0.2)(AGconc)
    AGconc = Dense(D, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)
    
    conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="same",
                  activation=tfa.activations.mish, name='Conv1')(conc)
    conc = GaussianNoise(Gn)(conc)
    conc = Dense(D, activation=tfa.activations.mish)(conc)
    conc = Flatten()(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(D, activation=tfa.activations.mish)(fs)

    concat = concatenate([conc, fsDense, AGconc])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: k.l2_normalize(x, axis=1), name='QuatNorm')(quat)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    model = Model(inputs=[Acc, Gyro, fs], outputs=[quat])
    model.summary()
    return model

def Model_B(window_size=200):
    Gn = 0.25
    acc = Input((window_size, 3), name='Acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    A1LSTM = Bidirectional(LSTM(50, return_sequences=True, name='A1LSTM'))(Acc1)
    A1LSTM = Dropout(0.2, name='DropoutA1LSTM')(A1LSTM)

    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    A2LSTM = Bidirectional(LSTM(50, return_sequences=True, name='A2LSTM'))(Acc2)
    A2LSTM = Dropout(0.2, name='DropoutA2LSTM')(A2LSTM)

    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)
    A3LSTM = Bidirectional(LSTM(50, return_sequences=True, name='A3LSTM'))(Acc3)
    A3LSTM = Dropout(0.2, name='DropoutA3LSTM')(A3LSTM)

    Aconc = concatenate([A1LSTM, A2LSTM, A3LSTM], name='Aconc')
    Aconc = Flatten(name='FlattenAconc')(Aconc)

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    G1LSTM = Bidirectional(LSTM(50, return_sequences=True, name='G1LSTM'))(Gyro1)
    G1LSTM = Dropout(0.2, name='DropoutG1LSTM')(G1LSTM)

    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    G2LSTM = Bidirectional(LSTM(50, return_sequences=True, name='G2LSTM'))(Gyro2)
    G2LSTM = Dropout(0.2, name='DropoutG2LSTM')(G2LSTM)

    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)
    G3LSTM = Bidirectional(LSTM(50, return_sequences=True, name='G3LSTM'))(Gyro3)
    G3LSTM = Dropout(0.2, name='DropoutG3LSTM')(G3LSTM)

    Gconc = concatenate([G1LSTM, G2LSTM, G3LSTM], name='Gconc')
    Gconc = Flatten(name='FlattenGconc')(Gconc)

    conc = concatenate([Aconc, Gconc], name='Conc')
    conc = Dense(256, activation="relu", name='DenseConc')(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(conc.shape[1], activation="relu", name='FsDense')(fs)

    conc2 = concatenate([conc, fsDense], name='Conc2')
    conc2 = Dense(256, activation="relu", name='DenseConc2')(conc2)
    conc2 = GaussianNoise(Gn, name='GaussianNoiseConc2')(conc2)
    quat = Dense(4, activation="linear", name='Quat')(conc2)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=[quat])
    model.summary()
    return model


def Model_C1(window_size):
    Gn = 0.25
    acc = Input((window_size, 3), name='acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    gyro = Input((window_size, 3), name='gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    AGconcat = concatenate([Acc, Gyro], axis=2, name='AGconcat')

    fs = Input((1,), name='fs')

    LSTM1 = Bidirectional(LSTM(96,
                                    return_sequences=True), name='LSTM1')(AGconcat)
    GNoise = GaussianNoise(Gn, name='GNoise')(LSTM1)
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
    ACNN = MaxPooling1D(pool_size=15,
                        name='MaxPooling1D')(ACNN)

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
    GCNN = MaxPooling1D(pool_size=15,
                        name='GyroMaxPool1D')(GCNN)

    AGM = concatenate([ACNN, GCNN])
    AGM = Conv1D(filters=128,
                 kernel_size=11,
                 padding='same',
                 activation=tfa.activations.mish,
                 name='AGM')(AGM)
    AGM = Flatten(name='AGMFlatten')(AGM)
    AGconLSTM = Bidirectional(LSTM(130,
                                        return_sequences=True,
                                        go_backwards=True,
                                        name='BiLSTM1'))(AGconcat)
    AGconLSTM = tfa.activations.mish(AGconLSTM)
    FlattenAG = Flatten(name='FlattenAG')(AGconLSTM)

    Fdense = Dense(units=160,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AGM)
    x = concatenate([AG, Fdense, FlattenAG])

    x = Dense(units=210,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    quat = Dense(4, activation='linear', name='quat')(x)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(quat)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=quat)
    model.summary()
    return model

def Model_C(window_size):
    Gn = 0.25
    acc = Input((window_size, 3), name='acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    gyro = Input((window_size, 3), name='gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    AGconcat = concatenate([Acc, Gyro], axis=2, name='AGconcat')

    fs = Input((1,), name='fs')

    LSTM1 = Bidirectional(LSTM(128,
                                    return_sequences=True), name='LSTM1')(AGconcat)
    GNoise = GaussianNoise(Gn, name='GNoise')(LSTM1)
    ACNN = Conv1D(filters=128,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN')(Acc)
    ACNN = Conv1D(filters=128,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN1')(ACNN)
    ACNN = MaxPooling1D(pool_size=3,
                        name='MaxPooling1D')(ACNN)

    GCNN = Conv1D(filters=128,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN')(Gyro)
    GCNN = Conv1D(filters=128,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN1')(GCNN)
    GCNN = MaxPooling1D(pool_size=3,
                        name='GyroMaxPool1D')(GCNN)

    AGM = concatenate([ACNN, GCNN])
    AGM = Conv1D(filters=128,
                 kernel_size=11,
                 padding='same',
                 activation=tfa.activations.mish,
                 name='AGM')(AGM)
    AGM = Flatten(name='AGMFlatten')(AGM)
    AGconLSTM = Bidirectional(LSTM(128,
                                        return_sequences=True,
                                        #go_backwards=True,
                                        name='BiLSTM1'))(AGconcat)
    AGconLSTM = tfa.activations.mish(AGconLSTM)
    FlattenAG = Flatten(name='FlattenAG')(AGconLSTM)

    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AGM)
    x = concatenate([AG, Fdense, FlattenAG])

    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    quat = Dense(4, activation='linear', name='quat')(x)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(quat)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=quat)
    model.summary()
    return model

def Model_C_Test(window_size):
    acc = Input((window_size, 3), name='acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    gyro = Input((window_size, 3), name='gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    AGconcat = concatenate([Acc, Gyro], axis=2, name='AGconcat')

    fs = Input((1,), name='fs')
    ACNN = Conv1D(filters=174,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN')(Acc)
    ACNN = Conv1D(filters=174,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='ACNN1')(ACNN)
    ACNN = MaxPooling1D(pool_size=4,
                        name='MaxPooling1D')(ACNN)
    ACNN = Flatten(name='ACNNF')(ACNN)

    GCNN = Conv1D(filters=154,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN')(Gyro)
    GCNN = Conv1D(filters=154,
                  kernel_size=11,
                  padding='same',
                  activation=tfa.activations.mish,
                  name='GCNN1')(GCNN)
    GCNN = MaxPooling1D(pool_size=4,
                        name='GyroMaxPool1D')(GCNN)
    GCNN = Flatten(name='GCNNF')(GCNN)

    AGconLSTM = Bidirectional(LSTM(130,
                                   return_sequences=True,
                                   # return_state=True,
                                   go_backwards=True,
                                   name='BiLSTM1'))(AGconcat)
    AGconLSTM = tfa.activations.mish(AGconLSTM)
    FlattenAG = Flatten(name='FlattenAG')(AGconLSTM)
    AG = concatenate([ACNN, GCNN, FlattenAG])
    AG = Dense(units=130,
               activation=tfa.activations.mish)(AG)
    Fdense = Dense(units=160,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    AG = Flatten(name='AGF')(AG)
    x = concatenate([AG, Fdense])

    x = Dense(units=210,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    output = Dense(4, activation='linear', name='quat')(x)
    output = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(output)
    model = Model(inputs=[acc, gyro, fs], outputs=output)
    model.summary()
    return model


def Model_D(window_size):
    Gn = 0.25
    Acc = Input((window_size, 3), name='Acc')
    Gyro = Input((window_size, 3), name='Gyro')

    acc = GaussianNoise(Gn)(Acc)
    BiLSTMA = Bidirectional(LSTM(128, return_sequences=True))(acc)
    BiLSTMA = Flatten()(BiLSTMA)
    
    gyro = GaussianNoise(Gn)(Gyro)
    BiLSTMG = Bidirectional(LSTM(128, return_sequences=True))(gyro)
    BiLSTMG = Flatten()(BiLSTMG)

    BiLSTM = concatenate([BiLSTMA, BiLSTMG])
    BiLSTM = Dense(256, activation=tfa.activations.mish)(BiLSTM)
    fs = Input((1,), name='Fs')
    fsDense = Dense(256, activation=tfa.activations.mish)(fs)

    concat = concatenate([BiLSTM, fsDense])
    concat = Dense(256, activation=tfa.activations.mish)(concat)
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1))(quat)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    model = Model(inputs=[Acc, Gyro, fs], outputs=[quat])
    model.summary()
    return model

def roll_from_quaternion_layer(q):
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    roll = tf.math.atan2(2*(w*x + y*z),
                       (1-2*(x*x + y*y)))
    pitch = tf.math.asin(2*(w*y - x*z))
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    qx = (tf.math.sin(roll/2) * tf.math.cos(zero_float64) * tf.math.cos(zero_float64) -
          tf.math.cos(roll/2) * tf.math.sin(zero_float64) * tf.math.sin(zero_float64))
    qx = tf.reshape(qx, (tf.shape(roll)[0], 1))
    qy = (tf.math.cos(roll/2) * tf.math.sin(zero_float64) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.cos(zero_float64) * tf.math.sin(zero_float64))
    qy = tf.reshape(qy, (tf.shape(roll)[0], 1))
    qz = (tf.math.cos(roll/2) * tf.math.cos(zero_float64) * tf.math.sin(zero_float64) -
          tf.math.sin(roll/2) * tf.math.sin(zero_float64) * tf.math.cos(zero_float64))
    qz = tf.reshape(qz, (tf.shape(roll)[0], 1))
    qw = (tf.math.cos(roll/2) * tf.math.cos(zero_float64) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.sin(zero_float64) * tf.math.sin(zero_float64))
    qw = tf.reshape(qw, (tf.shape(roll)[0], 1))
    quat = tf.concat([qw, qx, qy, qz], axis=-1)
    return quat
def pitch_from_quaternion_layer(q):
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    pitch = tf.math.asin(-2*(w*y - x*z))
    quat_pitch = tf.concat([tf.cos(pitch/2), tf.zeros_like(pitch), tf.sin(pitch/2), tf.zeros_like(pitch)], axis=-1)
    return pitch
def yaw_from_quaternion_layer(q):
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    yaw = tf.math.atan2(2*(w*x + y*z), (w*w - x*x - y*y + z*z))
    quat_yaw = tf.concat([tf.cos(yaw/2), tf.zeros_like(yaw), tf.zeros_like(yaw), tf.sin(yaw/2)], axis=-1)
    return quat_yaw

def roll_est(window_size):
    Gn = 0.25
    acc = Input((window_size, 3), name='acc')
    Acc = GaussianNoise(Gn, name='GaussianNoiseAcc')(acc)
    gyro = Input((window_size, 3), name='gyro')
    Gyro = GaussianNoise(Gn, name='GaussianNoiseGyro')(gyro)
    fs = Input((1,), name='fs')
    Fs = GaussianNoise(0.1, name='GaussianNoiseFs')(fs)
    
    ALSTM = Bidirectional(LSTM(128,
                                   return_sequences=True,
                                   # return_state=True,
                                   #go_backwards=True,
                                   name='BiLSTM1'))(AGconcat)
    
    
    GLSTM = Bidirectional(LSTM(128,
                                   return_sequences=True,
                                   # return_state=True,
                                   #go_backwards=True,
                                   name='BiLSTM2'))(AGconcat)
    
    AGconcat = concatenate([Acc, Gyro])
    AGLSTM = Bidirectional(LSTM(128,
                                   return_sequences=True,
                                   # return_state=True,
                                   #go_backwards=True,
                                   name='BiLSTM3'))(AGconcat)
    
    Fdense = Dense(units=256,
                   activation=tfa.activations.mish,
                   name='Fdense')(fs)
    x = Dense(units=256,
              activation=tfa.activations.mish)(x)
    x = Flatten(name='output')(x)
    quat = Dense(4, activation='linear', name='quat')(x)
    quat = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='NormalizeQuat')(quat)
    quat = Lambda(roll_from_quaternion_layer, name='Attitude')(quat)
    model = Model(inputs=[acc, gyro, fs], outputs=quat)
    model.summary()
    return model


