import sys
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from riann.riann import RIANN
from dataset_loader import *
from ahrs.filters import EKF, Madgwick, Mahony, Complementary
import tensorflow_addons as tfa
from util import *
from keras.models import load_model
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pickle
import os
from train_A_G_Fs import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
riann = RIANN()
n = 40
window_size, stride = 100, 4

Bench_Name = [ 'Proposed_Model_A','Proposed_Model_B','Proposed_Model_C','RIANN', "EKF", 'CF', 'Madgwick', 'Mahony']
dataset_name = ['BROAD', 'OxIOD', 'Sassari', 'RoNIN', 'RIDI','RepoIMU_TStick']


# Load Data from:
# BROAD, OxIOD, Sassari, RoNIN, RIDI, euroc
# Load Model, in each fucntion, predict and calculate the error, then store the error in dataframe and save it
# error would be consist of:
# Simultanously, use EKF, Madgwick, Mahony, and RIANN for comparison study
# The result would be saved in a dataframe, and then save it in a csv file
# The saved dataframe would be consists of RMSE_roll, RMSE_Pitch, and Total_Rotation_Error for each trial and caluclted by the DL model, EKF, Madgwick, Mahony, and RIANN
#                           Total_Rotation_Error             |                  RMSE_roll                   |                       RMSE_pitch              |
# Trial No,     DL Model | EKF | Madgwick | Mahony | RIANN  ||| DL Model | EKF | Madgwick | Mahony | RIANN ||| DL Model | EKF | Madgwick | Mahony | RIANN  |||
#
window_size = 100
stride = 4



def model_load(model_name):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    model = load_model(("./%s.hdf5" % model_name),
                       custom_objects={
                               'lossDCM2Quat': lossDCM2Quat,
                               "metric_dcm2quat_angle": metric_dcm2quat_angle,
                               'AdamW': tfa.optimizers.AdamW,
                               'Quat_mult': QQuat_mult, 'QQuat_mult': QQuat_mult,
                               "Quat_error_angle": Quat_error_angle, "Quat_error": Quat_error,
                               'TCN': TCN})
    return model

def predict_quaternion(model_name, acc, gyro, fs, quat, window_size, stride):
    try:
        quat = Att(quat)
        model = model_load(model_name)
        [gyro, acc, fs], [quat] = load_dataset_A_G_Fs(gyro, acc, quat, window_size, stride, fs)
        quat_pred = model.predict([acc, gyro, fs], batch_size=50, verbose=1)
        return quat_pred, quat
    except Exception as e:
        raise Exception("An error occurred during quaternion prediction: " + str(e))

def RIANN_pred(acc,gyro,fs):
    quat_pred = riann.predict(acc,gyro,fs)
    return quat_pred
def EKF_pred(dataset_name,acc,gyro,fs):
    quat_pred = EKF(acc=acc, gyr=gyro, frequency=fs, frame='ENU').Q
    if dataset_name == 'BROAD':
        roll, pitch, yaw = quat2eul(quat_pred)
        quat_pred = eul2quat(0, -pitch, roll-np.pi)
    
    return quat_pred
def CF_pred(acc,gyro,fs):
    quat_pred = Complementary(acc=acc, gyr=gyro, frequency=fs).Q
    return quat_pred
def Madgwick_pred(acc,gyro,fs):
    quat_pred = Madgwick(acc=acc, gyr=gyro, frequency=fs).Q
    return quat_pred
def Mahony_pred(acc,gyro,fs):
    quat_pred = Mahony(acc=acc, gyr=gyro, frequency=fs).Q
    return quat_pred

def Total_Error(pred, true):
    Quat_true = tf.Variable(true, dtype=tf.float64)
    Quat_pred = tf.Variable(pred, dtype=tf.float64)
    Total_Rotation_Error = Quat_error_angle(Quat_true, Quat_pred).numpy()
    Total_Rotation_Error_mean = np.mean(np.sqrt(Total_Rotation_Error**2))
    return Total_Rotation_Error, Total_Rotation_Error_mean

def Calc_Error(dataset_name,bench_name, acc, gyro, fs, quat):
    if "Model" in bench_name:
        pred,quat = globals()[bench_name + '_pred'](acc,gyro,fs,quat)
    elif bench_name == "EKF":
        pred = EKF_pred(dataset_name,acc,gyro,fs)
    else:
        pred = globals()[bench_name + '_pred'](acc, gyro, fs)
    pred = Att(pred)
    quat = Att(quat)
    Total_Rotation_Error, Total_Rotation_Error_mean = Total_Error(pred, quat)
    return Total_Rotation_Error, Total_Rotation_Error_mean


def Bench(bench_list, dataset_name):
    for dataset in dataset_name:
        header = np.hstack(['Trial No,', bench_list])
        globals()['df_total_error_mean_' + dataset] = pd.DataFrame(columns=header, index=None)
        for bench_name in bench_list:
            globals()['df_total_error_all_trial_' + bench_name + '_' + dataset] = pd.DataFrame(index=None)
            
    for bench_name in bench_list:
        globals()['Total_Rotation_Error_all_' + bench_name] = pd.DataFrame( index=None)
        print(globals()['Total_Rotation_Error_all_' + bench_name])
        
    for bench_name in bench_list:
        
        for dataset in dataset_name:
            globals()['dataset'] = dataset
            Total_Rotation_Error_all = []
            path = globals()[dataset + '_path']
            dataset_loader = globals()[dataset + '_data']
            ################ remove the below line
            path = path()
            for trial in path:
                acc, gyro, mag, quat, fs = dataset_loader(trial)
                trial_name = trial 
                if "trial_imu" in trial:
                    trial_name = trial.replace('trial_imu', ' Trial No, ')
                elif "/syn" in trial:
                    trial_name = trial.replace('/syn', '')
                quat = Att(quat)
                #acc, gyro, mag, quat, = acc[:200,:], gyro[:200,:], mag[:200,:], quat[:200,:]
                Total_Rotation_Error, Total_Rotation_Error_mean = Calc_Error(dataset,bench_name, acc, gyro, fs, quat)
                if trial_name not in globals()['df_total_error_mean_' + dataset]['Trial No,'].values:
                    globals()['df_total_error_mean_' + dataset] = pd.concat([globals()['df_total_error_mean_' + dataset],
                                                                             pd.DataFrame(np.hstack([trial_name, Total_Rotation_Error_mean]).reshape(1, -1), columns=['Trial No,', bench_name])])
                globals()['df_total_error_mean_' + dataset].loc[globals()['df_total_error_mean_' + dataset]['Trial No,'] == trial_name, bench_name] = Total_Rotation_Error_mean 
                Total_Rotation_Error_all = np.append(Total_Rotation_Error_all, Total_Rotation_Error)
                globals()['df_total_error_all_trial_' + bench_name + '_' + dataset] = pd.concat([globals()['df_total_error_all_trial_' + bench_name + '_' + dataset], pd.DataFrame(Total_Rotation_Error, columns=[trial_name])], axis=1)

            globals()['df_total_error_mean_' + dataset].to_csv('df/Mean/df_total_error_mean_' + dataset + '.csv', index=False)
            globals()['df_total_error_all_trial_' + bench_name + '_' + dataset].to_csv('df/All_Trial/df_total_error_all_trial_' + bench_name + '_' + dataset + '.csv', index=False)
            globals()['Total_Rotation_Error_all_' + bench_name] = pd.concat([globals()['Total_Rotation_Error_all_' + bench_name], pd.DataFrame(Total_Rotation_Error_all, columns=[dataset])], axis=1)
            #print(globals()['Total_Rotation_Error_all_' + bench_name])
            del Total_Rotation_Error_all
        globals()['Total_Rotation_Error_all_' + bench_name].to_csv('df/Whole/Total_Rotation_Error_all_' + bench_name + '.csv', index=False)
    #print(globals()['df_total_error_mean_' + dataset])
    

Bench(Bench_Name, dataset_name)



