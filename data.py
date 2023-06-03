import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate

from keras.utils import Sequence


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def load_euroc_mav_dataset(imu_data_filename, gt_data_filename):
    gt_data = pd.read_csv(gt_data_filename).values
    imu_data = pd.read_csv(imu_data_filename).values

    gyro_data = interpolate_3dvector_linear(
        imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc_data = interpolate_3dvector_linear(
        imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data


def load_oxiod_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]

    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, pos_data, ori_data


def load_broad_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    gyro_data = imu_data[:, 3:6]
    acc_data = imu_data[:, 0:3]
    mag_data = imu_data[:, 6:9]

    pos_data = gt_data[:, 0:3]
    ori_data = gt_data[:, 3:7]

    return gyro_data, acc_data, mag_data, pos_data, ori_data


def force_quaternion_uniqueness(q):
    q_data = q
    # force the quaternion to be unique
    for i in range(q_data.shape[0]):
        if q_data[i, 0] < 0:
            q_data[i, :] = -q_data[i, :]
        
            # print("done")
    return q_data
    '''
    q_data = q

    if np.absolute(q_data[:, 0]) > 1e-05:
        if q_data[:, 0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[:, 1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[:, 2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[:, 3] < 0:
            return -q
        else:
            return q
'''


def cartesian_to_spherical_coordinates(point_cartesian):
    delta_l = np.linalg.norm(point_cartesian)

    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0


def load_dataset_6d_rvec(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    #imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    #gt_data = np.genfromtxt(gt_data_filename, delimiter=',')

    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    #imu_data = imu_data[1200:-300]
    #gt_data = gt_data[1200:-300]

    gyro_acc_data = np.concatenate(
        [imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)

    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    init_q = quaternion.from_float_array(
        ori_data[window_size//2 - stride//2, :])

    init_rvec = np.empty((3, 1))
    cv2.Rodrigues(quaternion.as_rotation_matrix(init_q), init_rvec)

    init_tvec = pos_data[window_size//2 - stride//2, :]

    x = []
    y_delta_rvec = []
    y_delta_tvec = []

    for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
        x.append(gyro_acc_data[idx + 1: idx + 1 + window_size, :])

        tvec_a = pos_data[idx + window_size//2 - stride//2, :]
        tvec_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(
            ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(
            ori_data[idx + window_size//2 + stride//2, :])

        rmat_a = quaternion.as_rotation_matrix(q_a)
        rmat_b = quaternion.as_rotation_matrix(q_b)

        delta_rmat = np.matmul(rmat_b, rmat_a.T)

        delta_rvec = np.empty((3, 1))
        cv2.Rodrigues(delta_rmat, delta_rvec)

        delta_tvec = tvec_b - np.matmul(delta_rmat, tvec_a.T).T

        y_delta_rvec.append(delta_rvec)
        y_delta_tvec.append(delta_tvec)

    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_rvec = np.reshape(
        y_delta_rvec, (len(y_delta_rvec), y_delta_rvec[0].shape[0]))
    y_delta_tvec = np.reshape(
        y_delta_tvec, (len(y_delta_tvec), y_delta_tvec[0].shape[0]))

    return x, [y_delta_rvec, y_delta_tvec], init_rvec, init_tvec


def load_dataset_6d_quat(gyro_data, acc_data, ori_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    #x = []
    x_gyro = []
    x_acc = []
    y_q = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1: idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1: idx + 1 + window_size, :])

        q_a = quaternion.from_float_array(
            ori_data[idx + window_size//2 - stride//2, :])

        y_q.append(quaternion.as_float_array(q_a))

    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(
        x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(
        x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_q = np.reshape(y_q, (len(y_q), y_q[0].shape[0]))

    # return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_q]


def load_dataset_6d_quat_fs(gyro_data, acc_data, ori_data, window_size=200, stride=10, fs=286):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    #x = []
    x_gyro = []
    x_acc = []
    y_q = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1: idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1: idx + 1 + window_size, :])

        q_a = quaternion.from_float_array(
            ori_data[idx + window_size//2 - stride//2, :])

        y_q.append(quaternion.as_float_array(q_a))

    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    fs = np.ones((len(x_gyro), 1))*fs
    temp = np.zeros((len(x_gyro), window_size, 4))
    x_gyro = np.reshape(
        x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    temp[:, :, 0:3] = x_gyro
    temp[:, :, 3] = fs
    x_gyro = temp
    x_acc = np.reshape(
        x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    temp[:, :, 0:3] = x_acc
    temp[:, :, 3] = fs
    x_acc = temp
    y_q = np.reshape(y_q, (len(y_q), y_q[0].shape[0]))

    # return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_q]


def load_dataset_3d(gyro_data, acc_data, loc_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    l0 = loc_data[window_size//2 - stride//2 - stride, :]
    l1 = loc_data[window_size//2 - stride//2, :]
    init_l = l1
    delta_l, init_theta, init_psi = cartesian_to_spherical_coordinates(l1 - l0)

    #x = []
    x_gyro = []
    x_acc = []
    y_delta_l = []
    y_delta_theta = []
    y_delta_psi = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1: idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1: idx + 1 + window_size, :])

        delta_l0, theta0, psi0 = cartesian_to_spherical_coordinates(
            loc_data[idx + window_size//2 - stride//2, :] - loc_data[idx + window_size//2 - stride//2 - stride, :])

        l0 = loc_data[idx + window_size//2 - stride//2, :]
        l1 = loc_data[idx + window_size//2 + stride//2, :]

        delta_l, theta1, psi1 = cartesian_to_spherical_coordinates(l1 - l0)

        delta_theta = theta1 - theta0
        delta_psi = psi1 - psi0

        if delta_theta < -np.pi:
            delta_theta += 2 * np.pi
        elif delta_theta > np.pi:
            delta_theta -= 2 * np.pi

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y_delta_l.append(np.array([delta_l]))
        y_delta_theta.append(np.array([delta_theta]))
        y_delta_psi.append(np.array([delta_psi]))

    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(
        x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(
        x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
    y_delta_theta = np.reshape(
        y_delta_theta, (len(y_delta_theta), y_delta_theta[0].shape[0]))
    y_delta_psi = np.reshape(
        y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

    # return x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi
    return [x_gyro, x_acc], [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi


def load_dataset_2d(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    #imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    #gt_data = np.genfromtxt(gt_data_filename, delimiter=',')

    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    #imu_data = imu_data[1200:-300]
    #gt_data = gt_data[1200:-300]

    gyro_acc_data = np.concatenate(
        [imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)

    loc_data = gt_data[:, 2:4]

    #l0 = loc_data[0, :]
    #l1 = loc_data[window_size, :]

    #l0 = loc_data[window_size - stride - stride, :]
    #l1 = loc_data[window_size - stride, :]

    l0 = loc_data[window_size//2 - stride//2 - stride, :]
    l1 = loc_data[window_size//2 - stride//2, :]

    l_diff = l1 - l0
    psi0 = np.arctan2(l_diff[1], l_diff[0])
    init_l = l1
    init_psi = psi0

    x = []
    y_delta_l = []
    y_delta_psi = []

    # for idx in range(stride, gyro_acc_data.shape[0] - window_size - 1, stride):
    # for idx in range(window_size, gyro_acc_data.shape[0] - window_size - 1, stride):
    for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
        x.append(gyro_acc_data[idx + 1: idx + 1 + window_size, :])

        #l0_diff = loc_data[idx, :] - loc_data[idx - stride, :]
        #l0_diff = loc_data[idx, :] - loc_data[idx - window_size, :]
        #l0_diff = loc_data[idx + window_size - stride, :] - loc_data[idx + window_size - stride - stride, :]
        l0_diff = loc_data[idx + window_size//2 - stride//2, :] - \
            loc_data[idx + window_size//2 - stride//2 - stride, :]
        psi0 = np.arctan2(l0_diff[1], l0_diff[0])

        #l0 = loc_data[idx, :]
        #l0 = loc_data[idx + window_size - stride, :]
        #l1 = loc_data[idx + window_size, :]

        #l0 = loc_data[idx, :]
        #l1 = loc_data[idx + stride, :]

        l0 = loc_data[idx + window_size//2 - stride//2, :]
        l1 = loc_data[idx + window_size//2 + stride//2, :]

        l_diff = l1 - l0
        psi1 = np.arctan2(l_diff[1], l_diff[0])
        delta_l = np.linalg.norm(l_diff)
        delta_psi = psi1 - psi0

        #psi0 = psi1

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y_delta_l.append(np.array([delta_l]))
        y_delta_psi.append(np.array([delta_psi]))

        #y_delta_l.append(np.array([delta_l / (window_size / 100)]))
        #y_delta_psi.append(np.array([delta_psi / (window_size / 100)]))

    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
    y_delta_psi = np.reshape(
        y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

    return x, [y_delta_l, y_delta_psi], init_l, init_psi


def get_data(name):
    imu_dataset = np.zeros((0, 9))
    gt_dataset = np.zeros((0, 7))
    mypath = 'data/'+name+'/'
    files = [f for f in os.listdir(
        mypath) if os.path.isfile(os.path.join(mypath, f))]
    for i in range(len(files)):
        if "IMU" in files[i]:
            print(files[i])
            # Read the IMU data
            imu_data = (pd.read_csv('data/'+name+'/' + files[i]).values)
            # Read the ground truth data
            gt_data = (pd.read_csv('data/'+name+'/' +
                       files[i].replace("IMU", "GT")).values)

            # store imported gt data and imu data in variables
            imu_dataset = np.vstack((imu_dataset, imu_data))
            gt_dataset = np.vstack((gt_dataset, gt_data))
    return imu_dataset, gt_dataset

    window_size = 200
    stride = 10

    x_gyro = []
    x_acc = []

    y_delta_p = []
    y_delta_q = []

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data2/syn/imu1.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data2/syn/imu2.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu2.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu4.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu4.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu2.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu7.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu4.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data4/syn/imu5.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu3.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu2.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data2/syn/imu3.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu1.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu3.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/imu5.csv')
        imu_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/imu4.csv')

        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data2/syn/vi1.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data2/syn/vi2.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data5/syn/vi2.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/vi4.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data4/syn/vi4.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data4/syn/vi2.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/vi7.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data5/syn/vi4.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data4/syn/vi5.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/vi3.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/vi2.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data2/syn/vi3.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/vi1.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/vi3.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data3/syn/vi5.csv')
        gt_data_filenames.append(
            'Oxford Inertial Odometry Dataset/handheld/data1/syn/vi4.csv')

    elif args.dataset == 'euroc':
        imu_data_filenames.append('MH_01_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_03_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_05_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_02_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_01_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_03_difficult/mav0/imu0/data.csv')

        gt_data_filenames.append(
            'MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append(
            'MH_03_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append(
            'MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append(
            'V1_02_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append(
            'V2_01_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append(
            'V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv')

    elif args.dataset == 'broad':
        fs = 286
        for i in range(1, 5):
            imu_data_filenames.append('BROAD/trial_imu'+str(i)+'.csv')

            gt_data_filenames.append('BROAD/trial_gt'+str(i)+'.csv')
        for i in range(20, 21):
            imu_data_filenames.append('BROAD/trial_imu'+str(i)+'.csv')

            gt_data_filenames.append('BROAD/trial_gt'+str(i)+'.csv')
        for i in range(29, 30):
            imu_data_filenames.append('BROAD/trial_imu'+str(i)+'.csv')

            gt_data_filenames.append('BROAD/trial_gt'+str(i)+'.csv')

    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        if args.dataset == 'oxiod':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(
                cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'euroc':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(
                cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'broad':
            cur_gyro_data, cur_acc_data, cur_mag_data, cur_pos_data, cur_ori_data = load_broad_dataset(
                cur_imu_data_filename, cur_gt_data_filename)

        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)

        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)

        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)

    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)
