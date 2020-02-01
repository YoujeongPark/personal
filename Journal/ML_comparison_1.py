import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy import signal

data_number = 7521
section_row = 2

"""
membrane_stress = 2 
bending_stress = 3 

"""




def create_features(data, seg_id, X, direction):

    fs = 20

    #print(type(data))


    f, Pxx_den = signal.welch(data, fs, nperseg=1024)
    #plt.semilogy(f, Pxx_den)
    # plt.plot(f, Pxx_den)
    # plt.show()
    X.loc[seg_id, 'data_length'] = data_length
    X.loc[seg_id, 'signal.welch.mean' + direction] = Pxx_den.mean()
    X.loc[seg_id, 'signal.welch.std' + direction] = Pxx_den.std()
    X.loc[seg_id, 'signal.welch.max' + direction] = Pxx_den.max()
    X.loc[seg_id, 'signal.welch.min' + direction] = Pxx_den.min()


    pd_series_Pxx_den = pd.Series(Pxx_den)


    for windows in [10,500]:
        Pxx_den_mean = pd_series_Pxx_den.rolling(windows).mean().dropna().values
        Pxx_den_std = pd_series_Pxx_den.rolling(windows).std().dropna().values  # dropna() NaN 값이 있으면 삭제

        X.loc[seg_id, 'welch_ave_roll_mean_' + str(windows) + direction] = Pxx_den_mean.mean()
        X.loc[seg_id, 'welch_std_roll_mean_' + str(windows) + direction] = Pxx_den_mean.std()
        X.loc[seg_id, 'welch_max_roll_mean_' + str(windows) + direction] = Pxx_den_mean.max()
        X.loc[seg_id, 'welch_min_roll_mean_' + str(windows) + direction] = Pxx_den_mean.min()

        X.loc[seg_id, 'welch_ave_roll_std_' + str(windows) + direction] = Pxx_den_std.mean()
        X.loc[seg_id, 'welch_std_roll_std_' + str(windows) + direction] = Pxx_den_std.std()
        X.loc[seg_id, 'welch_max_roll_std_' + str(windows) + direction] = Pxx_den_std.max()
        X.loc[seg_id, 'welch_min_roll_std_' + str(windows) + direction] = Pxx_den_std.min()



    fft_data = pd.Series(np.fft.fft(data))
    # X.loc[seg_id, 'mean' + direction] = data.mean()
    # X.loc[seg_id, 'std' + direction] = data.std()
    # X.loc[seg_id, 'max' + direction] = data.max()
    # X.loc[seg_id, 'min' + direction] = data.min()

    # FFT transform values
    realFFT = np.real(fft_data)
    imagFFT = np.imag(fft_data)

    X.loc[seg_id, 'Rmean' + direction] = realFFT.mean()
    X.loc[seg_id, 'Rstd' + direction] = realFFT.std()
    X.loc[seg_id, 'Rmax' + direction] = realFFT.max()
    X.loc[seg_id, 'Rmin' + direction] = realFFT.min()


    X.loc[seg_id, 'Imean' + direction] = imagFFT.mean()
    X.loc[seg_id, 'Istd' + direction] = imagFFT.std()
    X.loc[seg_id, 'Imax' + direction] = imagFFT.max()
    X.loc[seg_id, 'Imin' + direction] = imagFFT.min()



    for windows in [10, 500]:
        fft_x_roll_mean = fft_data.rolling(windows).mean().dropna().values
        fft_x_roll_std = fft_data.rolling(windows).std().dropna().values  # dropna() NaN 값이 있으면 삭제

        #print(fft_x_roll_mean)

        X.loc[seg_id, 'fft_ave_roll_mean_' + str(windows) + direction] = fft_x_roll_mean.mean()
        X.loc[seg_id, 'fft_std_roll_mean_' + str(windows) + direction] = fft_x_roll_mean.std()
        X.loc[seg_id, 'fft_max_roll_mean_' + str(windows) + direction] = fft_x_roll_mean.max()
        X.loc[seg_id, 'fft_min_roll_mean_' + str(windows) + direction] = fft_x_roll_mean.min()

        X.loc[seg_id, 'fft_ave_roll_std_' + str(windows) + direction]  = fft_x_roll_std.mean()
        X.loc[seg_id, 'fft_std_roll_std_' + str(windows) + direction]  = fft_x_roll_std.std()
        X.loc[seg_id, 'fft_max_roll_std_' + str(windows) + direction]  = fft_x_roll_std.max()
        X.loc[seg_id, 'fft_min_roll_std_' + str(windows) + direction]  = fft_x_roll_std.min()



        # X.loc[seg_id, 'q01_roll_std_' + str(windows) + direction] = np.quantile(x_roll_std, 0.01)
        # X.loc[seg_id, 'q025_roll_std_' + str(windows) + direction] = np.quantile(x_roll_std, 0.25)
        # X.loc[seg_id, 'q05_roll_std_' + str(windows) + direction] = np.quantile(x_roll_std, 0.50)
        # X.loc[seg_id, 'q75_roll_std_' + str(windows) + direction] = np.quantile(x_roll_std, 0.75)
        # X.loc[seg_id, 'q99_roll_std_' + str(windows) + direction] = np.quantile(x_roll_std, 0.99)

        # X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)+ direction] = np.mean(np.diff(x_roll_mean))
        # X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)+ direction] = np.mean(
        #     np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        # X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)+ direction] = np.abs(x_roll_mean).max()





if __name__ == "__main__":

    is_local = False
    if (is_local):
        print("fail")
    else:
        PATH = "../earthquake_data/"

    data_list = os.listdir(PATH)
    print(data_list)

    #print(os.listdir(PATH))
    file_len = len(os.listdir(PATH))

    f"파일 갯수는 {file_len!r} 입니다 "

    len_list_path = len(os.listdir(PATH))
    data_list = os.listdir(PATH)
    data_output = pd.DataFrame(dtype=np.float64)

    #data_number
    for seg_id in range(0,file_len):
    #for seg_id in range(0, file_len):
        path = PATH + data_list[seg_id]
        data_df = pd.read_csv(path, names=['time', '_x', '_y', '_z'],header=None, error_bad_lines=False )
        data_length = len(data_df['time'].to_list())
        print(data_length)
        for direction in ['_x', '_y', '_z']:
            data = pd.Series(data_df[direction].values)
            create_features(data, seg_id, data_output, direction)








    data_output.to_csv('x_data.csv', index=False)

    """
    X_Data Scaler   
    """

    scaler = StandardScaler()
    scaler.fit(data_output)
    scaled_data_output = pd.DataFrame(scaler.transform(data_output), columns=data_output.columns)
    scaled_data_output.to_csv('scaled_data_x_data.csv', index=False)





    """
    Y Data       
    """


















# RF_model = RandomForestRegressor()
# RF_model.fit(X_tr,y_tr)
# RF_score = RF_model.score(X_val,y_val)
# print("RF_score",RF_score)
#
#
# SVR_model = SVR()
# SVR_model.fit(X_tr,y_tr)
# SVR_score = SVR_model.score(X_val,y_val)
# print("SVR_score",SVR_score)
#
# KN_model = KNeighborsRegressor()
# KN_model.fit(X_tr,y_tr)
# KN_score = KN_model.score(X_val,y_val)
# print("KN_score",KN_score)
#
# GradientBoostingRegressor_model = ensemble.GradientBoostingRegressor()
# GradientBoostingRegressor_model.fit(X_tr,y_tr)
# GradientBoostingRegresso_score= GradientBoostingRegressor_model.score(X_val,y_val)
# Result_predict = GradientBoostingRegressor_model.predict(X_val)
# print("GradientBoostingRegresso_score",GradientBoostingRegresso_score)


