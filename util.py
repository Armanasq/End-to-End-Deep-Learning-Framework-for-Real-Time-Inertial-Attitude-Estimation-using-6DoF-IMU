
from ahrs.filters import EKF, Mahony, Madgwick, fourati
import numpy as np
import quaternion
from dataset_loader import *
import tensorflow as tf


def quat2eul(q):
    """
    The function takes in a quaternion and returns the roll, pitch, and yaw angles.

    :param q: quaternion
    :return: the roll, pitch and yaw angles of the quaternion.
    """
    normalized_array = q/np.linalg.norm(q, axis=1).reshape(len(q), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    roll_x = (np.arctan2(2*(w[:, 0]*x[:, 0] + y[:, 0]*z[:, 0]),
                         (1-2*(x[:, 0]*x[:, 0] + y[:, 0]*y[:, 0]))))
    pitch_y = (np.arcsin(2*(w[:, 0]*y[:, 0] - x[:, 0]*z[:, 0])))
    yaw_z = (np.arctan2(2*(w[:, 0]*z[:, 0] + x[:, 0]*y[:, 0]),
                        (1-2*(y[:, 0]*y[:, 0] + z[:, 0]*z[:, 0]))))
    return roll_x.reshape(len(roll_x), 1), pitch_y.reshape(len(roll_x), 1), yaw_z.reshape(len(roll_x), 1)


def eul2quat(yaw, pitch, roll):
    """
    > The function takes in three angles (yaw, pitch, roll) and returns a quaternion (qw, qx, qy, qz)

    The function is written in Python, but it's not too hard to translate it to C++

    :param yaw: rotation around the z-axis
    :param pitch: rotation around the x-axis
    :param roll: rotation around the x-axis
    :return: The quaternion representation of the euler angles.
    """
    qx = (np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) -
          np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2))
    qy = (np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) +
          np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2))
    qz = (np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) -
          np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2))
    qw = (np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) +
          np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2))
    return np.hstack((qw, qx, qy, qz))


def Att(q):
    normalized_array = q/np.linalg.norm(q, axis=1).reshape(len(q), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    roll = (np.arctan2(2*(w[:, 0]*x[:, 0] + y[:, 0]*z[:, 0]),
                       (1-2*(x[:, 0]*x[:, 0] + y[:, 0]*y[:, 0]))))
    pitch = (np.arcsin(2*(w[:, 0]*y[:, 0] - x[:, 0]*z[:, 0])))

    qx = (np.sin(roll/2) * np.cos(pitch/2) * np.cos(0) -
          np.cos(roll/2) * np.sin(pitch/2) * np.sin(0)).reshape(len(pitch), 1)
    qy = (np.cos(roll/2) * np.sin(pitch/2) * np.cos(0) +
          np.sin(roll/2) * np.cos(pitch/2) * np.sin(0)).reshape(len(pitch), 1)
    qz = (np.cos(roll/2) * np.cos(pitch/2) * np.sin(0) -
          np.sin(roll/2) * np.sin(pitch/2) * np.cos(0)).reshape(len(pitch), 1)
    qw = (np.cos(roll/2) * np.cos(pitch/2) * np.cos(0) +
          np.sin(roll/2) * np.sin(pitch/2) * np.sin(0)).reshape(len(pitch), 1)
    quat = np.concatenate((qw, qx, qy, qz), axis=1)
    return quat


def Head(q):
    normalized_array = q/np.linalg.norm(q, axis=1).reshape(len(q), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    yaw = (np.arctan2(2*(w[:, 0]*z[:, 0] + x[:, 0]*y[:, 0]),
                      (1-2*(y[:, 0]*y[:, 0] + z[:, 0]*z[:, 0]))))

    qx = (np.sin(0) * np.cos(0) * np.cos(yaw/2) -
          np.cos(0) * np.sin(0) * np.sin(yaw/2)).reshape(len(yaw), 1)
    qy = (np.cos(0) * np.sin(0) * np.cos(yaw/2) +
          np.sin(0) * np.cos(0) * np.sin(yaw/2)).reshape(len(yaw), 1)
    qz = (np.cos(0) * np.cos(0) * np.sin(yaw/2) -
          np.sin(0) * np.sin(0) * np.cos(yaw/2)).reshape(len(yaw), 1)
    qw = (np.cos(0) * np.cos(0) * np.cos(yaw/2) +
          np.sin(0) * np.sin(0) * np.sin(yaw/2)).reshape(len(yaw), 1)

    quat = np.concatenate((qw, qx, qy, qz), axis=1)
    return quat


def Att_q(quat):
    """
    > The function takes in a quaternion and returns a quaternion that represents the same rotation but
    with the yaw component set to zero

    :param quat: the quaternion that represents the attitude of the drone
    :return: The quaternion of the attitude of the quadcopter.
    """
    roll, pitch, yaw_z = quat2eul(quat)
    Att_quat = eul2quat(0, pitch, roll)
    return np.asarray(Att_quat)


def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + \
            np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))

    return np.reshape(pred_l, (len(pred_l), 3))


def load_dataset_A_G_Fs(gyro_data, acc_data, ori_data, window_size, stride, fs):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    #x = []
    x_gyro = []
    x_acc = []
    y_q = []
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        x_gyro.append(gyro_data[idx + 1: idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1: idx + 1 + window_size, :])
        q_a = quaternion.from_float_array(
            ori_data[idx + window_size//2 - stride//2, :])
        y_q.append(quaternion.as_float_array(q_a))

    x_gyro = np.reshape(
        x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(
        x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_q = np.reshape(y_q, (len(y_q), y_q[0].shape[0]))
    fs = np.ones((len(x_gyro), 1))*fs
    # return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc, fs], [y_q]


def load_dataset_A_G_M_Fs(gyro_data, acc_data, mag_data, ori_data, window_size, stride, fs):
    x_gyro = []
    x_acc = []
    x_mag = []
    y_q = []
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        x_gyro.append(gyro_data[idx + 1: idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1: idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1: idx + 1 + window_size, :])
        q_a = quaternion.from_float_array(
            ori_data[idx + window_size//2 - stride//2, :])
        y_q.append(quaternion.as_float_array(q_a))

    x_gyro = np.reshape(
        x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(
        x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(
        x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_q = np.reshape(y_q, (len(y_q), y_q[0].shape[0]))
    fs = np.ones((len(x_gyro), 1))*fs
    return [x_gyro, x_acc, x_mag, fs], [y_q]


def Quat_error_angle3(y_true, y_pred):
    # quaternion error
    q_true = quaternion.from_float_array(y_true)
    q_pred = quaternion.from_float_array(y_pred)
    q_error = q_true * q_pred.inverse()
    q_error = quaternion.as_float_array(q_error)
    q_error = q_error / np.linalg.norm(q_error)

    return (q_error[0])


def yaw_err_rad(y_true, y_pred):
    while y_true > np.pi:
        y_true = y_true - 2*np.pi
    while y_true < -np.pi:
        y_true = y_true + 2*np.pi


def Quat_error(y_true, y_pred):
    # quaternion error
    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1

    return tf.abs(1-w)


def quat2dcm(quat):
    """
    This function converts a quaternion to a direction cosine matrix
    :param quat: a quaternion
    :return: a direction cosine matrix
    """
    normalized_array = quat/np.linalg.norm(quat, axis=1).reshape(len(quat), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    # flatten dcm to 1d array
    dcm = np.hstack((w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y), 2*(x*y + w*z), w**2 -
                    x**2 + y**2 - z**2, 2*(y*z - w*x), 2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 + z**2))
    return dcm


def quat2dcm3x3(quat):
    """
    This function converts a quaternion to a direction cosine matrix
    :param quat: a quaternion
    :return: a direction cosine matrix
    """
    normalized_array = quat/np.linalg.norm(quat, axis=1).reshape(len(quat), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    dcm3x3 = np.hstack((w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y), 2*(x*y + w*z), w **
                       2 - x**2 + y**2 - z**2, 2*(y*z - w*x), 2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 + z**2))
    dcm3x3 = np.reshape(dcm3x3, (len(dcm3x3), 3, 3))
    return dcm3x3


def dcm2quat(dcm):
    # accepet vectorize dcm (n,9)
    dcm = np.reshape(dcm, (len(dcm), 3, 3))

    w = np.sqrt(0.25*(1 + dcm[:, 0, 0] + dcm[:, 1, 1] +
                dcm[:, 2, 2])).reshape(len(dcm), 1)
    x = 0.25*(1 + dcm[:, 0, 0] - dcm[:, 1, 1] -
              dcm[:, 2, 2]).reshape(len(dcm), 1)
    y = 0.25*(1 - dcm[:, 0, 0] + dcm[:, 1, 1] -
              dcm[:, 2, 2]).reshape(len(dcm), 1)
    z = 0.25*(1 - dcm[:, 0, 0] - dcm[:, 1, 1] +
              dcm[:, 2, 2]).reshape(len(dcm), 1)
    return np.concatenate((w, x, y, z), axis=1)


def Quat_error_angle(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the predicted quaternion, and then calculates the
    angle between the two quaternions

    :param y_true: the true quaternion
    :param y_pred: the predicted quaternion
    :return: The angle between the two quaternions.
    """
    # remove nan values
    # remove nan values

    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    w = tf.boolean_mask(w, tf.math.is_finite(w))
    angle = (tf.abs(
        2 * tf.math.acos(tf.keras.backend.clip(tf.math.sqrt(tf.math.square(w)), -.99999999999, .99999999999))))
    angle = tf.boolean_mask(angle, tf.math.is_finite(angle))
    # avoid nan
    angle = tf.where(tf.math.is_nan(angle), tf.zeros_like(angle), angle)
    angle = tf.where(tf.math.is_inf(angle), tf.zeros_like(angle), angle)
    

    return angle * 180/np.pi


def tf_dcm2quat(dcm):
    # tensorflow tensors
    # accepet vectorize dcm (n,9)
    dcm = tf.reshape(dcm, (tf.shape(dcm)[0], 3, 3))
    w = tf.sqrt(0.25*(1 + dcm[:, 0, 0] + dcm[:, 1, 1] + dcm[:, 2, 2]))
    x = 0.25*(1 + dcm[:, 0, 0] - dcm[:, 1, 1] - dcm[:, 2, 2])
    y = 0.25*(1 - dcm[:, 0, 0] + dcm[:, 1, 1] - dcm[:, 2, 2])
    z = 0.25*(1 - dcm[:, 0, 0] - dcm[:, 1, 1] + dcm[:, 2, 2])
    return tf.stack((w, x, y, z), axis=1)


def tf_dcm2quat2(dcm):
    w = tf.sqrt(0.25*(1 + dcm[:, 0, 0] + dcm[:, 1, 1] + dcm[:, 2, 2]))
    x = 0.25*(1 + dcm[:, 0, 0] - dcm[:, 1, 1] - dcm[:, 2, 2])
    y = 0.25*(1 - dcm[:, 0, 0] + dcm[:, 1, 1] - dcm[:, 2, 2])
    z = 0.25*(1 - dcm[:, 0, 0] - dcm[:, 1, 1] + dcm[:, 2, 2])
    return tf.stack((w, x, y, z), axis=1)


def lossDCM2Quat(y_true, y_pred):
    # convert dcm to quaternion in tensorflow
    y_true = tf_dcm2quat(y_true)
    y_pred = tf_dcm2quat(y_pred)
    # quaternion error
    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    w = tf.boolean_mask(w, tf.math.is_finite(w))

    return tf.abs(1-w)


def lossDCM(y_true, y_pred):
    # recive 3x3 dcm and clculate dcm
    # inverse y_pred
    y_true


def metric_dcm2quat_angle(y_true, y_pred):
    y_true = tf_dcm2quat2(y_true)
    y_pred = tf_dcm2quat2(y_pred)
    err = Quat_error_angle(y_true, y_pred)
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.where(tf.math.is_nan(err), tf.zeros_like(err), err)
    err = tf.where(tf.math.is_inf(err), tf.zeros_like(err), err)
    err = tf.reduce_mean(err)
    return err


def metric_dcm2quat_angle2(y_true, y_pred):
    y_true = tf_dcm2quat(y_true)
    y_pred = tf_dcm2quat(y_pred)
    err = Quat_error_angle(y_true, y_pred)
    err = tf.boolean_mask(err, tf.math.is_finite(err))
    err = tf.where(tf.math.is_nan(err), tf.zeros_like(err), err)
    err = tf.where(tf.math.is_inf(err), tf.zeros_like(err), err)
    err = tf.reduce_mean(err)
    return err


def quat_error_angle(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the predicted quaternion, and then calculates the
    angle between the two quaternions

    :param y_true: the true quaternion
    :param y_pred: the predicted quaternion
    :return: The angle between the two quaternions.
    """

    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    w = tf.boolean_mask(w, tf.math.is_finite(w))
    angle = (tf.abs(
        2 * tf.math.acos(tf.keras.backend.clip(tf.math.sqrt(tf.math.square(w)), -.99999999999, .99999999999))))
    angle = tf.boolean_mask(angle, tf.math.is_finite(angle))
    # avoid nan
    angle = tf.where(tf.math.is_nan(angle), tf.zeros_like(angle), angle)
    angle = tf.where(tf.math.is_inf(angle), tf.zeros_like(angle), angle)
    angle = angle.numpy().reshape(len(angle), 1)
    return (angle * 180/np.pi)


def QQuat_mult(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the first one, and then multiplies the two
    quaternions together.

    The function returns the absolute value of the vector part of the resulting quaternion.

    The reason for this is that the vector part of the quaternion is the axis of rotation, and the
    absolute value of the vector part is the angle of rotation.

    The reason for normalizing the first quaternion is that the first quaternion is the predicted
    quaternion, and the predicted quaternion is not always normalized.

    The reason for returning the absolute value of the vector part of the resulting quaternion is that
    the angle of rotation is always positive.

    The reason for returning the vector part of the resulting quaternion is that the axis of rotation is
    always a vector.

    :param y_true: the ground truth quaternion
    :param y_pred: the predicted quaternion
    :return: The absolute value of the quaternion multiplication of the predicted and true quaternions.
    """
    # to increase the computation speed
    # remove nan values
    # remove nan values

    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    w = tf.subtract(w, 1)
    x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    z = w0*z1 + x0*y1 - y0*x1 + z0*w1
    #quat_pred = tf.concat([w, x, y, z], axis=1)
    #quat_pred = tf.linalg.norm(quat_pred, axis=1)
    #loss = tf.abs(tf.subtract(quat_pred, [1.0, 0, 0, 0]))
    loss = tf.abs(tf.concat(values=[w, x, y, z], axis=-1))
    # increase the loss robustness
    #loss = tf.boolean_mask(loss, tf.math.is_finite(loss))
    # avoid nan
    #loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    #loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    # increase the loss efficiency
    ##########################################################################
    #  truncated loss
    loss = tf.where(tf.math.greater(loss, 1.0), tf.ones_like(loss), loss)
    loss = tf.where(tf.math.less(loss, -1.0), tf.ones_like(loss), loss)
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    ##########################################################################
    return tf.reduce_sum(loss, axis=-1)#tf.abs(x), tf.abs(y),tf.abs(z) ,tf.abs(w)  #loss #
'''

def data_broad(window_size, stride):
    # broad data
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    broad_set = [1, 2, 3, 4, 5, 6, 7, 8, 18, 20, 21, 22, 29]
    for i in broad_set:
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = BROAD_data(
            BROAD_path(i)[0], BROAD_path(i)[1])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    return [gyro, acc, fs], [quat]


def data_oxiod(window_size, stride):
    path = []
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data2/syn/imu1.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data2/syn/imu2.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu2.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu4.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu4.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu2.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu7.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu4.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu5.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu3.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu2.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data2/syn/imu3.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu1.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu3.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu5.csv')
    path.append(
        'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu4.csv')
    for i in range(len(path)):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = OxIOD_data(
            dataset_path+path[i])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    quat = eul2quat(yaw_ref, pitch_ref, roll_ref - np.pi/2 - np.pi/4)
    return [gyro, acc, fs], [quat]


def data_repoIMU_TStick(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    df_TStick, df_Pendulum = repoIMU_path()
    for i in range(len(df_TStick.values)//2):
        acc_temp, gyro_temp, mag_temp, quat_temp, fs = repoIMU_data(
            df_TStick.values[i][0])
        acc = np.concatenate((acc, acc_temp), axis=0)
        gyro = np.concatenate((gyro, gyro_temp), axis=0)
        mag = np.concatenate((mag, mag_temp), axis=0)
        quat = np.concatenate((quat, quat_temp), axis=0)
    return [gyro, acc, fs], [quat]


def data_sassari(window_size, stride):
    acc, gyro, mag, quat = np.zeros((0, 3)), np.zeros(
        (0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
    MIMU = ["XS1", "AP2", "SH1", "XS2", "AP1", "SH2"]
    file_list = sassari_path()
    for i in range(len(file_list)):
        for j in range(len(MIMU)//2):
            acc_temp, gyro_temp, mag_temp, quat_temp, fs = sassari_data(
                file_list[i], MIMU[j])
            acc = np.concatenate((acc, acc_temp), axis=0)
            gyro = np.concatenate((gyro, gyro_temp), axis=0)
            mag = np.concatenate((mag, mag_temp), axis=0)
            quat = np.concatenate((quat, quat_temp), axis=0)
    return [gyro, acc, fs], [quat]'''


def test_mahony():
    [gyro, acc, fs], [quat] = data_broad(window_size, stride)
    mahony = Mahony(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = mahony.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')
    plt.show()

    [gyro, acc, fs], [quat] = data_sassari(window_size, stride)
    mahony = Mahony(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = mahony.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')
    plt.show()

    [gyro, acc, fs], [quat] = data_repoIMU_TStick(window_size, stride)
    mahony = Mahony(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = mahony.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()

    [gyro, acc, fs], [quat] = data_oxiod(window_size, stride)
    mahony = Mahony(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = mahony.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()


def test_madgwick():
    [gyro, acc, fs], [quat] = data_broad(window_size, stride)
    madgwick = Mdgwick(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = madgwick.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')
    plt.show()

    [gyro, acc, fs], [quat] = data_sassari(window_size, stride)
    mahony = Mdgwick(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = mahony.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')
    plt.show()

    [gyro, acc, fs], [quat] = data_repoIMU_TStick(window_size, stride)
    madgwick = Mdgwick(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = madgwick.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()

    [gyro, acc, fs], [quat] = data_oxiod(window_size, stride)
    madgwick = Mdgwick(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = madgwick.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()


def test_EKF():
    [gyro, acc, fs], [quat] = data_broad(window_size, stride)
    ekf = EKF(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = ekf.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()

    [gyro, acc, fs], [quat] = data_sassari(window_size, stride)
    ekf = EKF(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = ekf.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()

    [gyro, acc, fs], [quat] = data_repoIMU_TStick(window_size, stride)
    ekf = EKF(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = ekf.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()

    [gyro, acc, fs], [quat] = data_oxiod(window_size, stride)
    ekf = EKF(gyr=gyro, acc=acc, frequency=fs)
    quat_pred = ekf.Q
    roll_pred, pitch_pred, yaw_pred = quat2eul(quat_pred)
    roll_ref, pitch_ref, yaw_ref = quat2eul(quat)
    roll_remse = np.sqrt(np.mean((roll_pred - roll_ref)**2)) * 180 / np.pi
    pitch_rmse = np.sqrt(np.mean((pitch_pred - pitch_ref)**2)) * 180 / np.pi
    print("roll_rmse: ", roll_remse, "pitch_rmse: ", pitch_rmse)
    plt.figure()
    plt.plot(roll_pred, label='roll_pred')
    plt.plot(roll_ref, label='roll_ref')
    plt.legend()
    plt.title('Roll')

    plt.figure()
    plt.plot(pitch_pred, label='pitch_pred')
    plt.plot(pitch_ref, label='pitch_ref')
    plt.legend()
    plt.title('Pitch')

    plt.show()


def roll_from_quaternion(quat):
    roll = np.arctan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2))
    return roll

def quaternion_form_of_roll_angle(roll):
    quat = np.zeros((roll.shape[0], 4))
    quat[:, 0] = np.cos(roll / 2)
    quat[:, 1] = np.sin(roll / 2)
    return quat

def quaternion_roll(q):
    normalized_array = q/np.linalg.norm(q, axis=1).reshape(len(q), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    roll = np.arctan2(2*(w[:, 0]*x[:, 0] + y[:, 0]*z[:, 0]), 1 - 2*(x[:, 0]**2 + y[:, 0]**2))
    q_new = np.zeros((roll.shape[0], 4))
    q_new[:, 0] = np.cos(roll/2)
    q_new[:, 1] = np.sin(roll/2)
    
    return q_new