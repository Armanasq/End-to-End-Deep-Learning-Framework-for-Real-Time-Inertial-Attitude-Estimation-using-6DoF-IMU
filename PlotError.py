# plot and boxplot error of the model from csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import matplotlib

file_path = os.path.dirname(os.path.abspath(__file__)) + '/Error/'
def plot_Broad():
    err = pd.read_csv("./Error/model_checkpoint_oxiod_set.csv")
    yticklabels_oxiod= (err.values[1:-1,0])
    err = pd.read_csv("./Error/model_checkpoint_RIDI_RMSE_Alt.csv")
    yticklabels_RIDI= (err.values[1:-1,0])
    err = pd.read_csv("./Error/model_checkpoint_ronin_RMSE.csv")
    yticklabels_ronin= (err.values[1:-1,0])
    err = pd.read_csv("./Error/model_checkpoint_Sassari_RMSE_Alt.csv")
    yticklabels_Sassari= (err.values[1:-1,0])
    err = pd.read_csv("./Error/model_checkpoint_TStick_RMSE_alt.csv")
    yticklabels_TStick= (err.values[1:-1,0])
    
    # read all files in the folder 'total'
    os.chdir(file_path + 'total')
    dir_list = [name for name in os.listdir('.') if os.path.isfile(name)]
    
    # read all files in the folder 'total' csv files
    #print(dir_list)
    for i in range(len(dir_list)):
        if dir_list[i].endswith('.csv'):
            print(dir_list[i])
            
            error = pd.read_csv(dir_list[i])
            df = pd.DataFrame(error.values[1:, 1:], columns=error.columns[1:])
            # find best figure size for the plot
            if len(df.columns) > 10:
                fig, ax = plt.subplots(figsize=(10,len(df.columns)//2))
            else:  
                fig, ax = plt.subplots(figsize=(10, 10))
            # split the figure into 2 parts if the number of columns is larger than 50
            ax = sns.boxplot(data=df, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.7, saturation=1, ax=ax, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
            # adaptive boxplot width
            
            # change boxplot width
            #ax = sns.boxplot(data=df, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, ax=ax, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
            # set a -- line at 0
            plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
            #plt.axvline(0, color='black', linewidth=0.8)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().set_facecolor((238/255, 238/255, 228/255))
            #plt.grid(color='white', linestyle='--', linewidth=0.8)
            
            if "oxiod" in dir_list[i]:
                ax.set_yticklabels(yticklabels_oxiod, fontsize=5)
            elif "RIDI" in dir_list[i]:
                ax.set_yticklabels(yticklabels_RIDI, fontsize=5)
            elif "ronin" in dir_list[i]:
                ax.set_yticklabels(yticklabels_ronin, fontsize=5)
            elif "sassari" in dir_list[i]:
                ax.set_yticklabels(yticklabels_Sassari, fontsize=5)
            elif "TStick" in dir_list[i]:
                ax.set_yticklabels(yticklabels_TStick, fontsize=5)
            elif "broad" in dir_list[i]:
                ax.set_yticklabels(np.arange(1, 40), fontsize=5)
            # degree symbol
            ax.set_xlabel('Total Attitude Error (°)')
            #plt.xlim(0)
            ax.set_ylabel('Trial Name')
            if "oxiod" in dir_list[i]:
                title = 'OxIOD'
            elif "ridi" in dir_list[i]:
                title = 'RIDI'
            elif "RoNIN" in dir_list[i]:
                title = 'RoNIN'
            elif "sassari" in dir_list[i]:
                title = 'Sassari'
            elif "TStick" in dir_list[i]:
                title = 'RepoIMU TStick'
            elif "broad" in dir_list[i]:
                title = 'BROAD'
            
            if "dl" in dir_list[i]:
                title += " Proposed Model"
            elif "riann" in dir_list[i]:
                title += " RIANN"
            elif "ekf" in dir_list[i]:
                title += " EKF"
            elif "madgwick" in dir_list[i]:
                title += " Madgwick"
            elif "mahony" in dir_list[i]:
                title += " Mahony"

            title += " Error"
            title = str(title)
            print(title)
            ax.set_title(title)
            plt.savefig(file_path + 'fig/Boxplot ' +str(title)+ '.png', dpi=300, bbox_inches='tight',  format='png')
            title = []
    #plt.xticks(range(1, len(error.columns)), error.columns[1:], rotation=90)
    #plt.setp(ax.get_xticklabels(), rotation=90)
    #boxplot.set_title('Boxplot of error')
    #boxplot.set_xlabel('Error')
    #boxplot.set_ylabel('Value')
    
        
    
def plot_all():
    oxiod_name = pd.read_csv(file_path + 'oxiod_set_name.csv').values
    ridi_name = pd.read_csv(file_path + 'ridi_set_name.csv').values
    ronin_name = pd.read_csv(file_path + 'ronin_set_name.csv').values
    # remove ' , ) from the name
    #print(ronin_name)
    sassari_name = pd.read_csv(file_path + 'sassari_set_name.csv').values
    tstick_name = pd.read_csv(file_path + 'repo_set_name.csv').values
    os.chdir(file_path + 'total')
    dir_list = [name for name in os.listdir('.') if os.path.isfile(name)]
    df_oxiod = pd.DataFrame()
    df_ridi = pd.DataFrame()
    df_ronin = pd.DataFrame()
    df_sassari = pd.DataFrame()
    df_tstick = pd.DataFrame()
    df_broad = pd.DataFrame()
    header = pd.MultiIndex.from_product([['Proposed Model', 'RIANN', 'EKF', 'Madgwick', 'Mahony'], ['Error']])
    
    
    
    # read all files in the folder 'total' csv files
    #print(dir_list)
    for i in range(len(dir_list)):
        if dir_list[i].endswith('.csv'):
            print(dir_list[i])
            # 
            error = pd.read_csv(dir_list[i])
            df = pd.DataFrame(error)
            # print header
            #print(df.columns)
           
            # drop columns that have not the name in *_name.csv in header

            if "ridi" in dir_list[i]:
                df = df[df.columns.intersection(ridi_name[:, 0])] 
                print("===================================================================================== done =====================================================================================")
            elif "RoNIN" in dir_list[i]:
                df = df[df.columns.intersection(ronin_name[:, 0])]
                print("===================================================================================== done =====================================================================================")
            
            #df = df.drop(df[df['Unnamed: 0'] == 'handheld/data1/imu1'].index)
            #print(df.iloc[0, :].values)
            #df = df.iloc[1:, 1:]
            df = df.values.flatten()
            if "oxiod" in dir_list[i]:
                if "dl" in dir_list[i]:
                    df_oxiod['Proposed Model'] = pd.Series(df)
                elif "riann" in dir_list[i]:
                    df_oxiod['RIANN'] = pd.Series(df)
                elif "ekf" in dir_list[i]:
                    df_oxiod['EKF'] = pd.Series(df)
                elif "madgwick" in dir_list[i]:
                    df_oxiod['Madgwick'] = pd.Series(df)
                elif "mahony" in dir_list[i]:
                    df_oxiod['Mahony'] = pd.Series(df)
            elif "ridi" in dir_list[i]:
                if "dl" in dir_list[i]:
                    df_ridi['Proposed Model'] = pd.Series(df)
                elif "riann" in dir_list[i]:  
                    df_ridi['RIANN'] = pd.Series(df)
                elif "ekf" in dir_list[i]:
                    df_ridi['EKF'] = pd.Series(df)
                elif "madgwick" in dir_list[i]:
                    df_ridi['Madgwick'] = pd.Series(df)
                elif "mahony" in dir_list[i]:
                    df_ridi['Mahony'] = pd.Series(df)
                    
            elif "RoNIN" in dir_list[i]:
                if "dl" in dir_list[i]:
                    df_ronin['Proposed Model'] = pd.Series(df)
                elif "riann" in dir_list[i]:
                    df_ronin['RIANN'] = pd.Series(df)
                elif "ekf" in dir_list[i]:
                    df_ronin['EKF'] = pd.Series(df)
                elif "madgwick" in dir_list[i]:
                    df_ronin['Madgwick'] = pd.Series(df)
                elif "mahony" in dir_list[i]:
                    df_ronin['Mahony'] = pd.Series(df)
            elif "sassari" in dir_list[i]:
                if "dl" in dir_list[i]:
                    df_sassari['Proposed Model'] = pd.Series(df)
                elif "riann" in dir_list[i]:
                    df_sassari['RIANN'] = pd.Series(df)
                elif "ekf" in dir_list[i]:
                    df_sassari['EKF'] = pd.Series(df)
                elif "madgwick" in dir_list[i]:
                    df_sassari['Madgwick'] = pd.Series(df)
                elif "mahony" in dir_list[i]:
                    df_sassari['Mahony'] = pd.Series(df)
            elif "TStick" in dir_list[i]:
                if "dl" in dir_list[i]:
                    df_tstick['Proposed Model'] = pd.Series(df)
                elif "riann" in dir_list[i]:
                    df_tstick['TStickRIANN'] = pd.Series(df)
                elif "ekf" in dir_list[i]:
                    df_tstick[' EKF'] = pd.Series(df)
                elif "madgwick" in dir_list[i]:
                    df_tstick[' Madgwick'] = pd.Series(df)
                elif "mahony" in dir_list[i]:
                    df_tstick[' Mahony'] = pd.Series(df)
            elif "broad" in dir_list[i]:
                if "dl" in dir_list[i]:
                    df_broad['Proposed Model'] = pd.Series(df)
                elif "riann" in dir_list[i]:
                    df_broad[' RIANN'] = pd.Series(df)
                elif "ekf" in dir_list[i]:
                    df_broad[' EKF'] = pd.Series(df)
                elif "madgwick" in dir_list[i]:
                    df_broad[' Madgwick'] = pd.Series(df)
                elif "mahony" in dir_list[i]:
                    df_broad[' Mahony'] = pd.Series(df)
    # change os directory to file path
    os.chdir(file_path)
    df_oxiod.to_csv('oxiod.csv')
    df_ridi.to_csv('ridi.csv')
    df_ronin.to_csv('ronin.csv')
    df_sassari.to_csv('sassari.csv')
    df_tstick.to_csv('tstick.csv')
    df_broad.to_csv('broad.csv')
    
    # boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.boxplot(data=df_broad, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.savefig('broad.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=df_oxiod, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.savefig('oxiod.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=df_ridi, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.savefig('ridi.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=df_ronin, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.savefig('ronin.png', dpi=300, bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=df_sassari, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.savefig('sassari.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=df_tstick, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.8, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.savefig('tstick.png', dpi=300, bbox_inches='tight')
    
def plot_main():
    tstick = pd.read_csv('tstick_final.csv')
    #tstick = tstick.rename(columns={' TStick RIANN': 'RIANN',' EKF': 'EKF',' Madgwick': 'Madgwick',' Mahony': 'Mahony'})
    #tstick.to_csv('tstick.csv')
    oxiod = pd.read_csv('oxiod_final.csv')
    #oxiod = oxiod.rename(columns={' RIANN': 'RIANN',' EKF': 'EKF',' Madgwick': 'Madgwick',' Mahony': 'Mahony'})
    #oxiod.to_csv('oxiod.csv')
    ridi = pd.read_csv('ridi_final.csv')
    #ridi = ridi.rename(columns={' RIANN': 'RIANN',' EKF': 'EKF',' Madgwick': 'Madgwick',' Mahony': 'Mahony'})
    #ridi.to_csv('ridi.csv')
    ronin = pd.read_csv('ronin_final.csv')
    #ronin = ronin.rename(columns={' RIANN': 'RIANN',' EKF': 'EKF',' Madgwick': 'Madgwick',' Mahony': 'Mahony'})
    #ronin.to_csv('ronin.csv')
    sassari = pd.read_csv('sassari_final.csv')
    #sassari = sassari.rename(columns={' RIANN': 'RIANN',' EKF': 'EKF',' Madgwick': 'Madgwick',' Mahony': 'Mahony'})
    #sassari.to_csv('sassari.csv')
    #tstick = tstick.rename(columns={'TStick RIANN': 'RIANN'})
    broad = pd.read_csv('broad_final.csv')
    '''
    tstick_2 = pd.read_csv('tstick_2.csv')
    oxiod_2 = pd.read_csv('oxiod_2.csv')
    ridi_2 = pd.read_csv('ridi_2.csv')
    ronin_2 = pd.read_csv('ronin_2.csv')
    sassari_2 = pd.read_csv('sassari_2.csv')
    broad = pd.read_csv('broad.csv')
    broad_2 = pd.read_csv('broad_2.csv')
        # add Proposed Model A from oxiod_2 to oxiod
    oxiod['Proposed Model A'] = oxiod_2['Proposed Model A']
    ridi['Proposed Model A'] = ridi_2['Proposed Model A']
    ronin['Proposed Model A'] = ronin_2['Proposed Model A']
    sassari['Proposed Model A'] = sassari_2['Proposed Model A']
    tstick['Proposed Model A'] = tstick_2['Proposed Model A']
    broad['Proposed Model A'] = broad_2['Proposed Model A']
    # rename 'Proposed Model' to 'Proposed Model B'
    oxiod = oxiod.rename(columns={'Proposed Model': 'Proposed Model B'})
    ridi = ridi.rename(columns={' Proposed Model': 'Proposed Model B'})
    ronin = ronin.rename(columns={' Proposed Model': 'Proposed Model B'})
    sassari = sassari.rename(columns={' Proposed Model': 'Proposed Model B'})
    broad = broad.rename(columns={' Proposed Model': 'Proposed Model B'})
    tstick = tstick.rename(columns={' Proposed Model': 'Proposed Model B'})
    broad = broad.rename(columns={' RIANN': 'RIANN',' EKF': 'EKF',' Madgwick': 'Madgwick',' Mahony': 'Mahony'})
    '''

    # change the order of the columns to Proposed Model, RIANN, EKF, Madgwick, Mahony
    oxiod = oxiod[[     'Proposed Model A', 'Proposed Model B',     'RIANN','EKF','Madgwick','Mahony']]
    oxiod = oxiod[~(oxiod[['Proposed Model A', 'Proposed Model B', 'RIANN', 'EKF', 'Madgwick', 'Mahony']] > 180)]
    print("Done")
    tstick = tstick[[   'Proposed Model A', 'Proposed Model B',     'RIANN','EKF','Madgwick','Mahony']]
    tstick = tstick[~(tstick[['Proposed Model A', 'Proposed Model B', 'RIANN', 'EKF', 'Madgwick', 'Mahony']] > 180)]
    ridi = ridi[[       'Proposed Model A', 'Proposed Model B',     'RIANN','EKF','Madgwick','Mahony']]
    ridi = ridi[~(ridi[['Proposed Model A', 'Proposed Model B', 'RIANN', 'EKF', 'Madgwick', 'Mahony']] > 180)]
    ronin = ronin[[     'Proposed Model A', 'Proposed Model B',     'RIANN','EKF','Madgwick','Mahony']]
    ronin = ronin[~(ronin[['Proposed Model A', 'Proposed Model B', 'RIANN', 'EKF', 'Madgwick', 'Mahony']] > 180)]
    sassari = sassari[[ 'Proposed Model A', 'Proposed Model B',     'RIANN','EKF','Madgwick','Mahony']]
    sassari = sassari[~(sassari[['Proposed Model A', 'Proposed Model B', 'RIANN', 'EKF', 'Madgwick', 'Mahony']] > 180)]
    broad = broad[[     'Proposed Model A', 'Proposed Model B',     'RIANN','EKF','Madgwick','Mahony']]
    broad = broad[~(broad[['Proposed Model A', 'Proposed Model B', 'RIANN', 'EKF', 'Madgwick', 'Mahony']] > 180)]
    '''
    oxiod.to_csv('oxiod_final.csv', index=False)
    tstick.to_csv('tstick_final.csv', index=False)
    ridi.to_csv('ridi_final.csv', index=False)
    ronin.to_csv('ronin_final.csv', index=False) 
    sassari.to_csv('sassari_final.csv', index=False)
    broad.to_csv('broad_final.csv', index=False)
    '''
    # absolute value of the error
    '''    
    oxiod = oxiod.applymap(lambda x: abs(x))
    tstick = tstick.applymap(lambda x: abs(x))
    ridi = ridi.applymap(lambda x: abs(x))
    ronin = ronin.applymap(lambda x: abs(x))
    sassari = sassari.applymap(lambda x: abs(x))
    broad = broad.applymap(lambda x: abs(x))
    '''
    # remove outliers

    # print("min values ", oxiod.min(), tstick.min(), ridi.min(), ronin.min(), sassari.min(), broad.min())
    print("max values ", oxiod.max(), tstick.max(), ridi.max(), ronin.max(), sassari.max(), broad.max()) 

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=oxiod, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Oxiod Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('oxiod.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=ridi, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('RIDI Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('ridi.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=ronin, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('RoNIN Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('ronin.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=sassari, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Sassari Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('sassari.png', dpi=300, bbox_inches='tight')
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=broad, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('BROAD Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('broad.png', dpi=300, bbox_inches='tight')
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=tstick, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('RepoIMU TStick Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('tstick.png', dpi=300, bbox_inches='tight')

def RMSE_2():
    ridi_name = pd.read_csv(file_path + 'ridi_set_name.csv').values
    ronin_name = pd.read_csv(file_path + 'ronin_set_name.csv').values
    # remove ' , ) from the name
    #print(ronin_name)
    sassari_name = pd.read_csv(file_path + 'sassari_set_name.csv').values
    tstick_name = pd.read_csv(file_path + 'repo_set_name.csv').values
    os.chdir(file_path + 'total')
    dir_list = [name for name in os.listdir('.') if os.path.isfile(name)]
    df_oxiod = pd.DataFrame()
    df_ridi = pd.DataFrame()
    df_ronin = pd.DataFrame()
    df_sassari = pd.DataFrame()
    df_tstick = pd.DataFrame()
    df_broad = pd.DataFrame()
    header = pd.MultiIndex.from_product([['Proposed Model', 'RIANN', 'EKF', 'Madgwick', 'Mahony'], ['Error']])
    
    
    
    # read all files in the folder 'total' csv files
    #print(dir_list)
    for i in range(len(dir_list)):
        if dir_list[i].endswith('.csv'):
            print(dir_list[i])
            # 
            error = pd.read_csv(dir_list[i])
            df = pd.DataFrame(error)
            # print header
            #print(df.columns)
           
            # drop columns that have not the name in *_name.csv in header

            if "ridi" in dir_list[i]:
                df = df[df.columns.intersection(ridi_name[:, 0])] 
                print("===================================================================================== done =====================================================================================")
            elif "RoNIN" in dir_list[i]:
                df = df[df.columns.intersection(ronin_name[:, 0])]
                print("===================================================================================== done =====================================================================================")
def plot_df():
    df_cf = pd.read_csv("./df/Whole/Total_Rotation_Error_all_CF.csv")
    df_cf = pd.DataFrame(df_cf)
    df_cf.columns = df_cf.columns + "_CF"
    
    '''df_ekf = pd.read_csv("./df/Whole/Total_Rotation_Error_all_EKF.csv")
    df_ekf = pd.DataFrame(df_ekf)
    df_ekf.columns = df_ekf.columns + "_EKF"'''
    
    df_Madgwick = pd.read_csv("./df/Whole/Total_Rotation_Error_all_Madgwick.csv")
    df_Madgwick = pd.DataFrame(df_Madgwick)
    df_Madgwick.columns = df_Madgwick.columns + "_Madgwick"
    
    df_Mahony = pd.read_csv("./df/Whole/Total_Rotation_Error_all_Mahony.csv")
    df_Mahony = pd.DataFrame(df_Mahony)
    df_Mahony.columns = df_Mahony.columns + "_Mahony"
    
    df_riann = pd.read_csv("./df/Whole/Total_Rotation_Error_all_RIANN.csv")
    df_riann = pd.DataFrame(df_riann)
    df_riann.columns = df_riann.columns + "_RIANN"
    
    df_Proposed_Model_A = pd.read_csv("./df/Whole/Total_Rotation_Error_all_Proposed_Model_A.csv")
    df_Proposed_Model_A = pd.DataFrame(df_Proposed_Model_A)
    df_Proposed_Model_A.columns = df_Proposed_Model_A.columns + "_Proposed_Model_A" 
    
    df_Proposed_Model_B = pd.read_csv("./df/Whole/Total_Rotation_Error_all_Proposed_Model_B.csv")
    df_Proposed_Model_B = pd.DataFrame(df_Proposed_Model_B)
    df_Proposed_Model_B.columns = df_Proposed_Model_B.columns + "_Proposed_Model_B"
    
    df_Proposed_Model_C = pd.read_csv("./df/Whole/Total_Rotation_Error_all_Proposed_Model_C.csv")
    df_Proposed_Model_C = pd.DataFrame(df_Proposed_Model_C)
    df_Proposed_Model_C.columns = df_Proposed_Model_C.columns + "_Proposed_Model_C"
    
    df = pd.concat([df_cf, df_Madgwick, df_Mahony, df_riann, df_Proposed_Model_A, df_Proposed_Model_B, df_Proposed_Model_C], axis=1)
    del df_Madgwick, df_Mahony, df_riann, df_Proposed_Model_A, df_Proposed_Model_B, df_Proposed_Model_C
    print(df)
    # header  BROAD_CF  OxIOD_CF  Sassari_CF  RoNIN_CF   RIDI_CF  RepoIMU_TStick_CF   BROAD_EKF  OxIOD_EKF  Sassari_EKF  RoNIN_EKF  RIDI_EKF  RepoIMU_TStick_EKF
    # sort df by BROAD_CF
    df_BROAD = df.filter(regex='BROAD')
    df_BROAD = df.reindex(sorted(df_BROAD.columns), axis=1)
    df_BROAD.columns = df_BROAD.columns.str.replace('BROAD_', '')
    
    df_OxIOD = df.filter(regex='OxIOD')
    df_OxIOD = df.reindex(sorted(df_OxIOD.columns), axis=1)
    df_OxIOD.columns = df_OxIOD.columns.str.replace('OxIOD_', '')
    
    df_Sassari = df.filter(regex='Sassari')
    df_Sassari = df.reindex(sorted(df_Sassari.columns), axis=1)
    df_Sassari.columns = df_Sassari.columns.str.replace('Sassari_', '')
    
    df_RoNIN = df.filter(regex='RoNIN')
    df_RoNIN = df.reindex(sorted(df_RoNIN.columns), axis=1)
    df_RoNIN.columns = df_RoNIN.columns.str.replace('RoNIN_', '')
    
    df_RIDI = df.filter(regex='RIDI')
    df_RIDI = df.reindex(sorted(df_RIDI.columns), axis=1)
    df_RIDI.columns = df_RIDI.columns.str.replace('RIDI_', '')
    
    df_RepoIMU_TStick = df.filter(regex='RepoIMU_TStick')
    df_RepoIMU_TStick = df.reindex(sorted(df_RepoIMU_TStick.columns), axis=1)
    df_RepoIMU_TStick.columns = df_RepoIMU_TStick.columns.str.replace('RepoIMU_TStick_', '')
    del df
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=df_BROAD, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('./fig/Total_Rotation_Error_all_BROAD.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=df_OxIOD, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('./fig/Total_Rotation_Error_all_OxIOD.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=df_Sassari, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('./fig/Total_Rotation_Error_all_Sassari.png', dpi=300, bbox_inches='tight')
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=df_RoNIN, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('./fig/Total_Rotation_Error_all_RoNIN.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=df_RIDI, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('./fig/Total_Rotation_Error_all_RIDI.png', dpi=300, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(data=df_RepoIMU_TStick, orient="h", palette="Set2", showfliers=False, linewidth=0.8, width=0.4, saturation=1, fliersize=0.5, whis=1.5, notch=False, medianprops={'linewidth': 0.8}, boxprops={'linewidth': 0.8}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, flierprops={'linewidth': 0.8},  meanprops={'linewidth': 0.8, 'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 4}, meanline=True,  showcaps=True,  showbox=True, zorder=1, )
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    #plt.axvline(0, color='black', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.ylabel('Model')
    plt.savefig('./fig/Total_Rotation_Error_all_RepoIMU_TStick.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
def plt_scatter1():
    df_BROAD = pd.read_csv('./df/Mean/df_total_error_mean_BROAD.csv')
    df_Sassari = pd.read_csv('./df/Mean/df_total_error_mean_Sassari.csv')
    df_RoNIN = pd.read_csv('./df/Mean/df_total_error_mean_RoNIN.csv')
    df_RIDI = pd.read_csv('./df/Mean/df_total_error_mean_RIDI.csv')
    df_RepoIMU_TStick = pd.read_csv('./df/Mean/df_total_error_mean_RepoIMU_TStick.csv')
    df_OxIOD = pd.read_csv('./df/Mean/df_total_error_mean_OxIOD.csv')
    # [ 'Proposed_Model_A','Proposed_Model_B','Proposed_Model_C','RIANN', "EKF", 'CF', 'Madgwick', 'Mahony']
    plt.scatter(df_BROAD['Proposed_Model_A'],df_BROAD["Trial No,"],  s=10, c='black', marker='o', label='BROAD')
    plt.scatter(df_BROAD['Proposed_Model_B'],df_BROAD["Trial No,"],  s=10, c='red', marker='o', label='Sassari')
    plt.scatter(df_BROAD['Proposed_Model_C'],df_BROAD["Trial No,"],  s=10, c='blue', marker='o', label='RoNIN')
    plt.scatter(df_BROAD['RIANN'],df_BROAD["Trial No,"],  s=10, c='green', marker='o', label='RIDI')
    plt.scatter(df_BROAD['EKF'],df_BROAD["Trial No,"],  s=10, c='gray', marker='o', label='RepoIMU_TStick')
    plt.scatter(df_BROAD['CF'],df_BROAD["Trial No,"],  s=10, c='orange', marker='o', label='OxIOD')
    plt.scatter(df_BROAD['Madgwick'],df_BROAD["Trial No,"],  s=10, c='purple', marker='o', label='Madgwick')
    plt.scatter(df_BROAD['Mahony'],df_BROAD["Trial No,"],  s=10, c='brown', marker='o', label='Mahony')
    plt.xlim(0, 10)
    plt.gca().set_facecolor((238/255, 238/255, 228/255))
    plt.title('Error')
    plt.xlabel('RMSE (deg)')
    plt.grid()
    
    plt.legend()
    plt.show()
def plt_scatter():
    # read all csv files in the folder ./df/All_Trial/
    df_BROAD = []
    df_OxIOD = []
    df_RIDI = []
    df_RoNIN = []
    df_Sassari = []
    
    dir_list = os.listdir('./df/All_Trial/')
    dir_list.sort()
    dir_list = [x for x in dir_list if not "EKF" in x]
    color = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
#for i in range(len(dir_list)):
    #df = pd.read_csv('./df/All_Trial/'+dir_list[i])
    # calculate the mean and std of each column
    #plt.figure(figsize=(10, 10))
    sns.set_style("darkgrid")

# Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    BROAD_list = [x for x in dir_list if "BROAD" in x]
    for i in range(len(BROAD_list)):
        df = pd.read_csv('./df/All_Trial/'+BROAD_list[i])
        df_mean = df.mean()
        df_std = df.std()
        name = BROAD_list[i].replace('df_total_error_all_trial_', '')
        name = name.replace('_BROAD.csv', '')
        # append the mean and std to the dataframe with index, columns = df.columns
        #df_BROAD = pd.DataFrame([df_mean, df_std], index=['mean', 'std'], columns=df.columns)
        
        #plt.plot(df_BROAD.columns, df_BROAD.loc['mean'], c=color[i], label=name)
        #plt.fill_between(df_BROAD.columns, df_BROAD.loc['mean']-df_BROAD.loc['std'], df_BROAD.loc['mean']+df_BROAD.loc['std'], color=color[i], alpha=0.2)
        #plt.errorbar(df.columns, df.values[0], yerr=df.values[1], fmt='.k', c=color[i], label=name)
        
        #print(x, "\t", y)
        # mean vs column name
        #plt.scatter(df_BROAD.loc['mean'], df_BROAD.columns, s=10, c=color[i], marker='o', label=name)
        #plt.scatter(x,y, s=10, c=color[i], marker='o', label=name)
        #for j in range(len(df_BROAD.columns)):
            #plt.errorbar(df_BROAD.loc['mean', df_BROAD.columns[j]], df_BROAD.loc['std', df_BROAD.columns[j]], xerr=0, yerr=0, fmt='o', c=color[i])
            #plt.text(df_BROAD.loc['mean', df_BROAD.columns[j]], df_BROAD.loc['std', df_BROAD.columns[j]], df_BROAD.columns[j], fontsize=8)
        
        # 'linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'
        #plt.plot(df_BROAD.columns, df_BROAD.loc['mean'], c=color[i], label=name)
        #plt.fill_between(df_BROAD.columns, df_BROAD.loc['mean']-df_BROAD.loc['std'], df_BROAD.loc['mean']+df_BROAD.loc['std'], color=color[i], alpha=0.2)
        
        #plt.legend()
        
        #df_BROAD = df_BROAD.stack().reset_index()
        #df_BROAD.columns = ['stat', 'columns', 'value']
        # Plot the mean values with a line
        #sns.lineplot(data=df_BROAD, x='columns', y='value', hue='stat', ci='sd', color=color[i], label=name)

        # Set labels and title for the plot
        '''
        ax.set_xlabel('Columns')
        ax.set_ylabel('Mean values')
        ax.set_title(f'{name} Mean values over time')'''
        #plt.hist(df.values.flatten(), bins=1000, color=color[i], alpha=0.5, label=name)
        df['timestep'] = range(df.shape[0])
        
        df_pivot = df.melt(id_vars=['timestep'], var_name='trial', value_name='error')
        df_pivot = df.pivot(index='timestep', columns=' Trial No, 1')

        # Create a figure and axis for the plot
        fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Use imshow to plot the heatmap of the error values
        sns.heatmap(df_pivot, cmap='coolwarm', ax=axs[1])

        # Use lineplot to plot the mean error values over time
        sns.lineplot(data=df_pivot.mean(axis=1), x=df_pivot.index, y=df_pivot.mean(axis=1), ax=axs[0], ci='sd', color='black')

    # Add a legend
    # Set labels and title for the plot
    axs[0].set_title('Mean error over time')
    axs[1].set_title('Heatmap of error')
    axs[1].set_xlabel('Trial')
    axs[1].set_ylabel('Timestep')
    fig.suptitle('Error over time for each trial')

    #plt.xlim(0, 15)
    plt.grid( linestyle='--', linewidth=0.5, color='white')
    plt.show()
            
    
#plot_main()
plt_scatter()