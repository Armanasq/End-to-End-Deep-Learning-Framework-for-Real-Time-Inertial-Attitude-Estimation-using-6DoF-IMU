from util import *
from dataset_loader import *
from model_A_G_Fs import *
from train_A_G_Fs import *
from keras.models import load_model
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow_addons as tfa
from riann.riann import RIANN
from keras_radam import RAdam
from keras.callbacks import *
from keras import backend as K
import tensorflow.keras.backend as K
from keras.regularizers import l2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
riann = RIANN()
q = 7
YAW = 0
stride = 4
window_size = 1
if window_size == 1:
    stride = 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
fs = 286
model_name = "model_checkpoint.hdf5"

broad_test = 37
oxiod_test = []
oxiod_test.append(
    'handheld/data1/syn/imu2')
'''oxiod_test.append(
    'handheld/data1/syn/imu5')
oxiod_test.append(
    'handheld/data1/syn/imu6')
oxiod_test.append(
    'handheld/data3/syn/imu1')
oxiod_test.append(
    'handheld/data4/syn/imu1')
oxiod_test.append(
    'handheld/data4/syn/imu3')
oxiod_test.append(
    'handheld/data5/syn/imu1')'''


def data_broad(window_size, stride):
    # broad data
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    acc_temp, gyro_temp, mag_temp, quat_temp, fs = BROAD_data(
        BROAD_path()[broad_test-1])
    acc = np.concatenate((acc, acc_temp), axis=0)
    gyro = np.concatenate((gyro, gyro_temp), axis=0)
    mag = np.concatenate((mag, mag_temp), axis=0)
    quat = np.concatenate((quat, quat_temp), axis=0)

    # acc = np.asanyarray(acc)
    # gyro = np.asanyarray(gyro)
    # mag = np.asanyarray(mag)
    # quat = np.asanyarray(quat)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_oxiod(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    for i in range(len(oxiod_test)):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = OxIOD_data(
            oxiod_test[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    quat = eul2quat(yaw_ref, pitch_ref, roll_ref)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_repoIMU_TStick(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    df_TStick = RepoIMU_TStick_path()
    for i in range(len(df_TStick)-5, len(df_TStick)-3):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = RepoIMU_TStick_data(
            df_TStick[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_repoIMU_Pendulum(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    df_Pendulum = RepoIMU_Pendulum_path()
    for i in range(len(df_Pendulum)-5, len(df_Pendulum)-3):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = RepoIMU_Pendulum_data(
            df_Pendulum[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_sassari(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    MIMU = ["XS1", "AP2", "SH1", "XS2", "AP1", "SH2"]
    file_list = Sassari_path()
    for i in range(len(file_list)//2, len(file_list)):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = Sassari_data(
            file_list[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return [gyro, acc, mag, fs], [quat]


def data_ronin(window_size, stride, i):
    file_list = RoNIN_path()

    acc, gyro, mag, quat, fs = RoNIN_data(file_list[-i])
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def data_ridi(window_size, stride, i):

    file_list = RIDI_path()

    acc, gyro, mag, quat, fs = RIDI_data(file_list[-i])
    globals()['acc_main'] = acc
    globals()['gyro_main'] = gyro
    globals()['quat_main'] = quat
    globals()['fs_main'] = fs
    if window_size != 1:
        [gyro, acc, mag, fs], [quat] = load_dataset_A_G_M_Fs(
            gyro, acc, mag, quat, window_size, stride, fs)
    else:
        fs = np.ones(shape=(quat.shape[0], 1))*fs
    return [gyro, acc, mag, fs], [quat]


def read_file_names(path):
    file_names = []
    for file in os.listdir(path):
        if file.endswith(".hdf5") or file.endswith(".h5"):
            file_names.append(file)
    return file_names

from keras_radam import RAdam
def model_load(file_name):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #optimizer = RAdam()
    #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
    if file_name == "default":
        model = load_model(model_name,
                           custom_objects={
                               'lossDCM2Quat': lossDCM2Quat,
                               "metric_dcm2quat_angle": metric_dcm2quat_angle,
                               'AdamW': tfa.optimizers.AdamW,
                               'Quat_mult': QQuat_mult, 'QQuat_mult': QQuat_mult,
                               "Quat_error_angle": Quat_error_angle, "Quat_error": Quat_error,
                               'TCN': TCN,
                            'AdamW': tfa.optimizers.AdamW,
                               "RAdam": RAdam,
                               }, compile=False)
    else:
        file_path = os.path.dirname(os.path.realpath(__file__))
        model = tf.keras.models.load_model(
            file_path + "/BestModel/" + file_name, custom_objects={
                'lossDCM2Quat': lossDCM2Quat,
                "metric_dcm2quat_angle": metric_dcm2quat_angle,
                'AdamW': tfa.optimizers.AdamW,
                'Quat_mult': QQuat_mult, 'QQuat_mult': QQuat_mult,
                "Quat_error_angle": Quat_error_angle, "Quat_error": Quat_error,
                'TCN': TCN})
    # rempove the gassian noise layer
    return model


def pred(acc, gyro, fs):
    quat_pred = model.predict(
        [acc, gyro, fs], batch_size=100, verbose=1)
    quat_pred = Att(quat_pred)
    return quat_pred


def error(quat_ref, acc, gyro, fs):
    att_q = Att(quat_ref)
    quat_pred = pred(acc, gyro, fs)
    ''' plt.figure()
    plt.plot(quat_pred[:, 0], label="q0")
    plt.plot(quat_pred[:, 1], label="q1")
    plt.plot(quat_pred[:, 2], label="q2")
    plt.plot(quat_pred[:, 3], label="q3")'''
    quat_dl = Att(quat_pred)
    Quat_dl = tf.Variable(quat_dl, dtype=tf.float64)
    Quat_ref = tf.Variable(att_q, dtype=tf.float64)
    tot_err = np.mean(quat_error_angle(Quat_ref, Quat_dl))
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat_ref)
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    err_roll = (roll_pred-roll_ref)

    err_pitch = (pitch_pred-pitch_ref)
    for k in range(len(err_roll)):
        if err_roll[k] > np.pi:
            err_roll[k] = err_roll[k]-2*np.pi
        elif err_roll[k] < -np.pi:
            err_roll[k] = err_roll[k]+2*np.pi
        if err_pitch[k] > np.pi:
            err_pitch[k] = err_pitch[k]-2*np.pi
        elif err_pitch[k] < -np.pi:
            err_pitch[k] = err_pitch[k]+2*np.pi
    err_roll = err_roll * 180/np.pi
    err_pitch = err_pitch * 180/np.pi
    err_roll = np.sqrt(((err_roll)**2))
    err_pitch = np.sqrt(((err_pitch)**2))
    RMSE_roll = np.sqrt(np.mean((err_roll)**2))
    RMSE_pitch = np.sqrt(np.mean((err_pitch)**2))

    print("RMSE_total: ", tot_err, "RMSE_roll: ",
          RMSE_roll, "RMSE_pitch: ", RMSE_pitch)
    fs = fs[0] * stride
    t = np.arange(0, len(roll_pred)/fs, 1/fs)
    plt.figure()
    plt.plot(t[:len(roll_pred)], roll_pred, label="roll_pred")
    plt.plot(t[:len(roll_pred)], roll_ref, label="roll_ref")
    plt.legend()
    plt.title('roll')
    plt.xlabel('time')
    plt.ylabel('angle')

    plt.figure()
    plt.plot(t[:len(roll_pred)], pitch_pred, label="pitch_pred")
    plt.plot(t[:len(roll_pred)], pitch_ref, label="pitch_ref")
    plt.legend()
    plt.title('pitch')
    plt.xlabel('time')
    plt.ylabel('angle')
    plt.show()


def error_riann():
    attitude_riann = riann.predict(acc_main, gyro_main, fs_main)
    Quat_riann = tf.Variable(Att(attitude_riann), dtype=tf.float64)
    roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
    Quat_main = tf.Variable(Att(quat_main), dtype=tf.float64)
    tot_err_riann = np.mean(quat_error_angle(Quat_main, Quat_riann))
    roll_main, pitch_main, yaw_main = quat2eul(quat_main)

    err_roll_riann = (roll_riann-roll_main)
    err_pitch_riann = (pitch_riann-pitch_main)
    for k in range(len(err_roll_riann)):
        if err_roll_riann[k] > np.pi:
            err_roll_riann[k] = err_roll_riann[k]-2*np.pi
        elif err_roll_riann[k] < -np.pi:
            err_roll_riann[k] = err_roll_riann[k]+2*np.pi
        if err_pitch_riann[k] > np.pi:
            err_pitch_riann[k] = err_pitch_riann[k]-2*np.pi
        elif err_pitch_riann[k] < -np.pi:
            err_pitch_riann[k] = err_pitch_riann[k]+2*np.pi
    err_roll_riann = err_roll_riann * 180/np.pi
    err_pitch_riann = err_pitch_riann * 180/np.pi

    err_roll_riann = np.sqrt(((err_roll_riann)**2))
    err_pitch_riann = np.sqrt(((err_pitch_riann)**2))
    RMSE_roll_riann = np.sqrt(np.mean((err_roll_riann)**2))
    RMSE_pitch_riann = np.sqrt(np.mean((err_pitch_riann)**2))
    print("RMSE_total_riann: ", tot_err_riann, "RMSE_roll_riann: ",
          RMSE_roll_riann, "RMSE_pitch_riann: ", RMSE_pitch_riann)
def plot_riann():
    plt.figure()
    plt.plot( roll_riann, label="roll_riann")
    plt.plot( roll_main, label="roll_main")
    plt.legend()
    plt.title('roll')
    
    plt.figure()
    plt.plot( pitch_riann, label="pitch_riann")
    plt.plot( pitch_main, label="pitch_main")   
    plt.legend()
    plt.title('pitch')
    plt.show()


def main():

    for k in range(1):
        if i == 1:
            [gyro, acc, mag, fs], [quat_ref] = data_broad(window_size, stride)
        elif i == 2:
            print("oxiod")
            [gyro, acc, mag, fs], [quat_ref] = data_oxiod(window_size, stride)
        elif i == 3:
            [gyro, acc, mag, fs], [quat_ref] = data_repoIMU_TStick(
                window_size, stride)
        elif i == 4:
            print("ronin")
            [gyro, acc, mag, fs], [quat_ref] = data_ronin(
                window_size, stride, k)
        elif i == 5:
            [gyro, acc, mag, fs], [quat_ref] = data_ridi(
                window_size, stride, k)
        elif i == 6:
            [gyro, acc, mag, fs], [quat_ref] = data_sassari(
                window_size, stride)
        elif i == 7:
            [gyro, acc, mag, fs], [quat_ref] = data_repoIMU_Pendulum(
                window_size, stride)
        acc, gyro = acc.reshape(acc.shape[0], 1,acc.shape[1]), gyro.reshape(
        gyro.shape[0], 1,gyro.shape[1])
        if YAW == True:
            headq = Head(quat_ref)
            quat_pred = model.predict(
                [acc, gyro, mag, fs], batch_size=500, verbose=1)
            roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
            #roll_pred, pitch_pred, yaw_pred = quat2eul(quat)
            #acc_riann, gyro_riann = acc_main, gyro_main
            #attitude_riann = riann.predict(acc_riann, gyro_riann, fs_main)
            #roll_riann, pitch_riann, yaw_riann = quat2eul(attitude_riann)
            yaw_err = np.abs(yaw - yaw_pred)

            RMSE_yaw = np.sqrt(np.mean(yaw_err**2)) * 180/np.pi
            print("RMSE_yaw", RMSE_yaw)
        else:
            file_name = "default"
            if file_name == "default":
                globals()['model'] = model_load(file_name)
                print(file_name)
                error(quat_ref, acc, gyro, fs)
            else:
                file_path = os.path.dirname(os.path.realpath(__file__))
                file_names = read_file_names(file_path + "/BestModel")
                for file_name in file_names:
                    globals()['model'] = model_load(file_name)
                    print(file_name)
                    error(quat_ref, acc, gyro, fs)
            error_riann()
    if YAW == True:
        plt.figure()
        plt.plot(yaw_pred, label='Pred')
        plt.plot(yaw, label='ref')
        plt.legend()
        plt.show()

    '''
    else:
        fs = fs[0] * stride
        t = np.arange(0, len(roll_pred)/fs, 1/fs)
        t_main = np.arange(0, (len(roll_main))/fs_main, 1/fs_main)
        # boxplot error roll
        print(err_roll.shape, err_roll_riann.shape)
        plt.figure()
        plt.boxplot(x=[err_roll.reshape(len(err_roll),), err_roll_riann.reshape(
            len(err_roll_riann),)], showmeans=True, labels=["DL", "Riann"], showfliers=True)
        plt.title('Roll error')
        plt.ylabel('Error (rad)')
        plt.savefig('boxplot_roll.png')
        plt.figure()
        plt.boxplot(x=[err_pitch.reshape(len(err_pitch),), err_pitch_riann.reshape(
            len(err_pitch_riann),)], showmeans=True, labels=["DL", "Riann"], showfliers=True)
        plt.title('Pitch error')
        plt.ylabel('Error (rad)')
        plt.savefig('boxplot_pitch.png')
    
        # plt.show()
        # plot riann and main
        plt.figure()
        plt.plot(t_main[:len(roll_main)], roll_main *
                 180/np.pi, label="main_roll")
        plt.plot(t_main[:len(roll_main)], roll_riann *
                 180/np.pi, label="riann_roll")
        plt.legend()

        plt.figure()
        plt.plot(t_main[:len(roll_main)], pitch_main *
                 180/np.pi, label="main_pitch")
        plt.plot(t_main[:len(roll_main)], pitch_riann *
                 180/np.pi, label="riann_pitch")
        plt.legend()
        # plot roll and pitch
        plt.figure()
        plt.plot(t[:len(roll_pred)], roll_pred, label="roll_pred")
        plt.plot(t[:len(roll_pred)], roll_ref, label="roll_ref")
        plt.legend()
        plt.title('roll')
        plt.xlabel('time')
        plt.ylabel('angle')

        plt.figure()
        plt.plot(t[:len(roll_pred)], pitch_pred, label="pitch_pred")
        plt.plot(t[:len(roll_pred)], pitch_ref, label="pitch_ref")
        plt.legend()
        plt.title('pitch')
        plt.xlabel('time')
        plt.ylabel('angle')

        # plt.show()

'''


if __name__ == '__main__':
    file_name = "default"
    for i in range(1, 5):
        main()
