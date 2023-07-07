import os
import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import scipy as sp
import h5py
from keras.utils import Sequence
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'
dataset_path = ""


def quat2euler(q):
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
    return roll_x.reshape(len(roll_x), ), pitch_y.reshape(len(roll_x), ), yaw_z.reshape(len(roll_x), )



def calculate_yaw_from_rp_mag(roll, pitch, mag):
    # Normalize the magnetometer data
    mag = mag / np.linalg.norm(mag)

    # Convert roll and pitch angles to a rotation matrix
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    # Rotate the magnetometer data by the roll and pitch angles
    mag_rot = np.dot(R_pitch, np.dot(R_roll, mag))

    # Calculate the yaw angle using the dot product between the
    # rotated magnetometer data and the reference north direction
    north_reference = np.array([0, 0, 1])
    yaw = np.arctan2(np.dot(mag_rot, north_reference), mag_rot[2])

    return yaw
def calculate_yaw_from_rp_mag_vectorized(roll, pitch, mag):
    # Calculate yaw angle using roll, pitch, and magnetometer data
    yaw = np.arctan2(np.sin(roll)*mag[:,0]+np.cos(roll)*mag[:,1], np.cos(pitch)*mag[:,0]+np.sin(pitch)*mag[:,2])
    return yaw
def rotate_accelerometer_vectorized(acc, roll_error, pitch_error):
    # Create 3D rotation matrix for roll error correction
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll_error), -np.sin(roll_error)],
                   [0, np.sin(roll_error), np.cos(roll_error)]])

    # Create 3D rotation matrix for pitch error correction
    Ry = np.array([[np.cos(pitch_error), 0, np.sin(pitch_error)], 
                   [0, 1, 0],
                   [-np.sin(pitch_error), 0, np.cos(pitch_error)]])
    
    # Create 3D rotation matrix for yaw error correction
    '''Rz = np.array([[np.cos(yaw_error), -np.sin(yaw_error), 0],
                     [np.sin(yaw_error), np.cos(yaw_error), 0],
                        [0, 0, 1]])'''
    # Rotate the accelerometer data to correct for roll and pitch errors
    acc_corrected= np.matmul(np.matmul(Rx,Ry), acc.T).T
    return acc_corrected
def change_coordinate_frame(acc, roll, pitch, yaw):
    # Create 3D rotation matrix for roll correction
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    # Create 3D rotation matrix for pitch correction
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], 
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    # Create 3D rotation matrix for yaw correction
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
    # Rotate the accelerometer data to align with the body frame
    acc_transformed = np.matmul(np.matmul(np.matmul(Rx,Ry), Rz), acc.T).T
    return acc_transformed

def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def BROAD_path():
    ''' i = [1,40]
    '''
    imu_path = []
    for i in range(1, 40):
        imu_path.append('trial_imu{}'.format(i))
    #gt_path = dataset_path + 'BROAD/trial_gt{}.csv'.format(i)
    #imu_path = dataset_path + 'BROAD/trial_imu{}.csv'.format(i)
    return imu_path

def BROAD_data(path):
    path = dataset_path+'BROAD/'+ path+'.csv'
    fs = 286
    imu_filename = path
    df = pd.read_csv(imu_filename, header=0).values
    acc = df[:, 0:3]
    gyro = df[:, 3:6]
    mag = df[:, 6:9]
    gt_filename = imu_filename.replace('imu', 'gt')
    df = pd.read_csv(gt_filename, header=0).values
    quat = df[:, 3:7]
    pose = df[:, 0:3]
    return acc, gyro, mag, quat,  fs


def EuRoC_MAV_path():
    imu_path = []
    gt_path = []
    os.chdir(dataset_path+"EuRoC_MAV_Dataset/")
    # list folders in directory
    folder_list = [folder for folder in os.listdir(
        '.') if os.path.isdir(folder)]
    for i in range(len(folder_list)):
        imu_path.append(os.path.join(folder_list[i], 'mav0/imu0/data.csv'))
        gt_path.append(os.path.join(
            folder_list[i], 'mav0/state_groundtruth_estimate0/data.csv'))
    df = pd.DataFrame({'imu_path': imu_path, 'gt_path': gt_path})
    return df


def EuRoC_MAV_data(imu_filename, gt_filename):
    fs = 200
    gt_data = pd.read_csv(gt_filename).values
    imu_data = pd.read_csv(imu_filename).values

    gyro = interpolate_3dvector_linear(
        imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc = interpolate_3dvector_linear(
        imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    # pos = gt_data[:, 1:4]
    quat = gt_data[:, 4:8]
    return acc, gyro, quat


def OxIOD_path():
    path = []
    file_path = []
    # change dirctroy to file path
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(dataset_path+"Oxford Inertial Odometry Dataset/")
    # dir_list = [f for f in os.listdir(oxiod) if os.path.isdir(f)]
    dir_list = [name for name in os.listdir(
        ".") if os.path.isdir(name) if not os.path.isfile(name)]
    dir_list.sort()
    dir_list.remove('test')
    dir_list.remove('large scale')
    for i in range(len(dir_list)):
        sub_folders = [name for name in os.listdir(
            dir_list[i]) if os.path.isdir(os.path.join(dir_list[i], name))]
        # ignore "test" folder
        
            
        for k in range(int(len(sub_folders))):
            path = (
                    dir_list[i]+'/'+sub_folders[k]+'/syn/')
            
                
            files = os.listdir(path)
            if 'Readme.txt' in files:
                files.remove('Readme.txt')
            for j in range(1, len(files)):
                if 'imu' in str(files[j]):
                    files[j] = files[j].replace('.csv', '')
                    file_path.append(path+files[j])
    file_path = [x for x in file_path if 'nexus 5' not in x]
    file_path.sort()
    #df = pd.DataFrame({"OxIOD File Path": file_path})
    #df.to_csv(dataset_path+"OxIOD_file_path.csv", index=False)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return file_path
print("OxIOD_path length", len(OxIOD_path()))

def OxIOD_data(filename):
    fs = 100
    filename = dataset_path+'Oxford Inertial Odometry Dataset/'+filename +".csv"
    oxiod_imu = pd.read_csv(filename).values
    oxiod_gt = pd.read_csv(filename.replace('imu', 'vi')).values
    oxiod_imu = oxiod_imu[1200:-300]
    oxiod_gt = oxiod_gt[1200:-300]
    acc = oxiod_imu[:, 7:10] * 9.81
    gyro = oxiod_imu[:, 4:7]
    mag = oxiod_imu[:, 13:16]
    ori = np.concatenate((oxiod_gt[:, 8:9], oxiod_gt[:, 5:8]), axis=1)
    pose = oxiod_gt[:, 2:5]
    
    return acc, gyro, mag, ori, pose, fs


def RepoIMU_TStick_path():
    path = []
    file_path = []
    os.chdir(dataset_path+"Repo IMU/TStick/")
    dir_list_TStick = [name for name in os.listdir('.')]
    # remove '.csv'
    dir_list_TStick = [x.replace('.csv', '') for x in dir_list_TStick]
    dir_list_TStick.sort()
    #df_TStick = pd.DataFrame({"TStick": dir_list_TStick})
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return dir_list_TStick
print("RepoIMU_TStick_path length", len(RepoIMU_TStick_path()))
def RepoIMU_Pendulum_path():
    path = []
    file_path = []
    os.chdir(dataset_path+"Repo IMU/Pendulum/")
    # list files in Pendulum folder
    dir_list_Pendulum = [name for name in os.listdir('.')]
    dir_list_Pendulum = [x.replace('.csv', '') for x in dir_list_Pendulum]
    dir_list_Pendulum.sort()
    #df_Pendulum = pd.DataFrame({"Pendulum": dir_list_Pendulum})
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return dir_list_Pendulum

def RepoIMU_Pendulum_data(filename):
    path = 'Pendulum/'
    filename = dataset_path + "Repo IMU/" + path + filename+ '.csv'
    fs = 100
    df = pd.read_csv(filename, delimiter=';', header=1)
    quat = df.values[2:, 1:5]

    acc = df.values[2:, 5:8]
    gyro = df.values[2:, 8:11]
    mag = df.values[2:, 11:14]
    return acc, gyro, mag, quat, fs

def RepoIMU_TStick_data(filename):
    path = 'TStick/'
    filename = dataset_path + "Repo IMU/" + path + filename+ '.csv'
    fs = 100
    df = pd.read_csv(filename, delimiter=';', header=1)
    quat = df.values[2:, 1:5]

    acc = df.values[2:, 5:8]
    gyro = df.values[2:, 8:11]
    mag = df.values[2:, 11:14]
    return acc, gyro, mag, quat, fs




def RIDI_path():
    os.chdir(dataset_path+"RIDI/data_publish_v2/")
    dir_list = [name for name in os.listdir('.') if os.path.isdir(name)]
    dir_list.sort()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return dir_list
print("RIDI_path length", len(RIDI_path()))

def RIDI_data(filename):
    fs = 200
    df = pd.read_csv(dataset_path+'RIDI/data_publish_v2/'+filename+'/processed/data.csv', index_col=0)
    acc = df[['acce_x', 'acce_y', 'acce_z']].values
    gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = df[['magnet_x', 'magnet_y', 'magnet_z']].values

    quat = df[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    pose = df[['pos_x', 'pos_y', 'pos_z']]
    return acc, gyro, mag, quat,  fs


def RoNIN_path():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    folder = ['train_dataset_2', 'unseen_subjects_test_set',
              'seen_subjects_test_set', 'train_dataset_1']
    for i in range(len(folder)):
        os.chdir(dataset_path+"Ronin/"+folder[i])
        folder_name = prefixed = [filename for filename in os.listdir(
            '.') if filename.startswith("a")]
        globals()['df{}'.format(i)] = pd.DataFrame(
            {folder[i]: folder_name})
        globals()['df{}'.format(i)] = \
            folder[i]+"/" + globals()['df{}'.format(i)]
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # df1.values[:,0] , df2.values[:,0], df3.values[:,0], 
    df = []
    for i in range(len(folder)):
        df = np.append(df, globals()['df{}'.format(i)].values[:, 0])
    df.sort()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return df
print("RoNIN_path length", len(RoNIN_path()))

def RoNIN_data(filename):
    '''folder_name = prefixed = [filename for filename in os.listdir('.') if filename.startswith("a")]
    RoNIN_data(RoNIN_path()[0].values[0][0]
    '''
    filename = dataset_path + "Ronin/" + filename
    
    fs = 200
    load_file = filename + '/data.hdf5'
    df = h5py.File(load_file, 'r')
    header = np.array(df.get('synced'))
    for i in range(len(np.array(df.get('synced')))):
        if header[i] == 'acce':
            acc = np.array(df.get('synced')[header[i]])
        if header[i] == 'gyro':
            gyro = np.array(df.get('synced')[header[i]])
        if header[i] == 'magnet':
            mag = np.array(df.get('synced')[header[i]])
    header = np.array(df.get('pose'))
    for i in range(len(np.array(df.get('pose')))):
        if header[i] == 'ekf_ori':
            quat = np.array(df.get('pose')[header[i]])
        if header[i] == 'tango_pos':
            pose = np.array(df.get('pose')[header[i]])
    return acc, gyro, mag, quat,  fs

def Sassari_path():
    os.chdir(dataset_path+"Sassari/")
    file_list = [name for name in os.listdir('.') if os.path.isfile(name)]
    file_list = [ s[:-4] for s in file_list ]
    # sort file names A-Z
    file_list.sort()
    # add MIMU = XS1, XS2, AP1, AP2, SH1, SH2 to all file names
    file_list = [ s + '/' + mimu for s in file_list for mimu in ['AP1', 'AP2', 'SH1', 'SH2','XS1', 'XS2'] ]
    # remove .mat from all file names
    
    #file_list = [dataset_path+"Sassari/" + s for s in file_list]
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return file_list

def Sassari_data(filename):
    mimu = filename[-3:]
    filename = filename[:-4]
    filename = dataset_path + "Sassari/" + filename + ".mat"
    fs = 100
    mat = sp.io.loadmat(filename)
    data = mat[mimu]
    imu = data[:, 1: 10]
    quat = data[:, 10: 14]
    acc = imu[:, 0:3]
    gyro = imu[:, 3:6]
    mag = imu[:, 6:9]
    return acc, gyro, mag, quat, fs
