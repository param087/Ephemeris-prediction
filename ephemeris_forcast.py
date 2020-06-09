# import lib
import pandas as pd
import numpy as np
from fbprophet import Prophet
import glob
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import cmath
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import statistics
import datetime	
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import gnsscal
from datetime import date


random.seed(42)
days_to_predict = 1
periods =  24 * days_to_predict


#import dataset
df_final = pd.DataFrame()
df_final = df_final.append(pd.read_csv("csv_files/211011117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211021117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211031117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211041117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211051117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211061117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211071117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211081117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211091117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211101117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211111117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211121117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211131117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211141117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211151117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211161117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211171117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211181117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211191117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211201117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211211117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211221117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211231117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211241117.17N.csv"))
df_final = df_final.append(pd.read_csv("csv_files/211251117.17N.csv"))


df_expect = pd.DataFrame()
df_expect = df_expect.append(pd.read_csv("csv_files/211261117.17N.csv"))

#sort dataset y time
df_final['epoch_time']=pd.to_datetime(df_final['epoch_time'], format='%Y-%m-%d %H:%M:%S')
df_final = df_final.sort_values(by=['epoch_time'])

df_expect['epoch_time']=pd.to_datetime(df_expect['epoch_time'], format='%Y-%m-%d %H:%M:%S')
df_expect = df_expect.sort_values(by=['epoch_time'])

columns = df_final.columns.values
prns = df_final['prn'].unique()

def percentError( predicted,expected):
    error_per = []
    for (e,p) in zip(expected, predicted):
        if e != 0:
            error_per.append((abs(e- p)/e)*100)
    if len(error_per) == 0:
        return 0
    return statistics.mean(error_per)

min_max_scaler_dict = {}

#store normalization model of features
for column in columns:
    if column != 'epoch_time':
        x = df_final[column].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        min_max_scaler_dict[column] = scaler.fit(x)

#empty dataframe for result
final_dataframe = pd.DataFrame()
original_dataframe = pd.DataFrame()
error = {}
plotdata = {}


#training loop for all satellites and features
for prn in prns:
    #print(prn)
    a = datetime.datetime.now()
    error[prn] = {}
    plotdata[prn] = {}
    ec = {}
    dc = {}
    df_final_prn = df_final.loc[df_final['prn'] == prn]
    df_expect_prn = df_expect.loc[df_expect['prn'] == prn].reset_index()

    for column in list(df_final.columns.values):
        if column == 'mean_anomaly': 
            df_t = pd.DataFrame()
            df_t['ds'] = df_final_prn['epoch_time']
            df_t['y'] = df_final_prn[column]
            model = Prophet()
            model.fit(df_t)
            df_p = pd.DataFrame()
            df_p['ds'] = df_expect_prn['epoch_time']
            df_predict_prn = model.predict(df_p)
            e  = {}
            e['mean_absolute_error'] = mean_absolute_error(df_expect_prn[column].values, df_predict_prn['yhat'].values)
            e['mean_squared_error'] = mean_squared_error(df_expect_prn[column].values, df_predict_prn['yhat'].values)
            e['percentError'] = percentError(df_expect_prn[column], df_predict_prn['yhat'])
            ec[column]= e
            d = {}
            d['expected'] = df_expect_prn[column].values
            d['predicted'] = df_predict_prn['yhat'].values
            d['index'] =  df_expect_prn['epoch_time'].values
            dc[column] = d
            original_dataframe[column] = df_expect_prn[column]
            final_dataframe[column] = pd.DataFrame(data = df_predict_prn['yhat'].values, columns=['m'])

        elif column == "OMEGA":

            df = pd.DataFrame()
            df['epoch_time'] = df_expect_prn['epoch_time']
            t_len = df_final_prn.shape[0]
            df['OMEGA'] = df_final_prn['OMEGA'][t_len-1:t_len][t_len-1:t_len]

            prev_week, _ = gnsscal.date2gpswd(df_final_prn['epoch_time'][t_len-1:t_len].dt.date.values[0])
            this_week, _ = gnsscal.date2gpswd(df_expect_prn['epoch_time'][0:1].dt.date.values[0])

            if this_week != prev_week:
                hdiff = (24 - df_final_prn['epoch_time'][t_len-1:t_len].dt.hour.values[0]) +df_expect_prn['epoch_time'][:1].dt.hour.values[0]
                df['OMEGA'][:1] =  df_final_prn['OMEGA'][t_len-1:t_len].values[0] - hdiff * 6 * (10**-6) - 0.12
            else:
                hdiff = (24 - df_final_prn['epoch_time'][t_len-1:t_len].dt.hour.values[0]) +df_expect_prn['epoch_time'][:1].dt.hour.values[0]
                df['OMEGA'][:1] =  df_final_prn['OMEGA'][t_len-1:t_len].values[0] - hdiff * 6 * (10**-6)

            for i in range(1, df_expect_prn.shape[0]):

                prev_week, _ = gnsscal.date2gpswd(df_expect_prn['epoch_time'][i-1:i].dt.date.values[0])
                this_week, _ = gnsscal.date2gpswd(df_expect_prn['epoch_time'][i: i+1].dt.date.values[0])

                if this_week != prev_week:
                    hdiff = (24 - df_expect_prn['epoch_time'][i-1: i].dt.hour.values[0]) + df['epoch_time'][i:i+1].dt.hour.values[0]
                    df['OMEGA'][i: i+1] =  df['OMEGA'][i-1: i].values[0] - hdiff * 6 * (10**-6) - 0.12
                else:
                    hdiff = (24 - df_expect_prn['epoch_time'][i-1: i].dt.hour.values[0]) + df['epoch_time'][i:i+1].dt.hour.values[0]
                    df['OMEGA'][i:i+1] =  df['OMEGA'][i-1:i].values[0] - hdiff * 6 * (10**-6)
            original_dataframe[column] = df_expect_prn[column]
            final_dataframe[column] = df['OMEGA']
            e  = {}
            e['mean_absolute_error'] = mean_absolute_error(df_expect_prn[column].values, df['OMEGA'].values)
            e['mean_squared_error'] = mean_squared_error(df_expect_prn[column].values, df['OMEGA'].values)
            e['percentError'] = percentError(df_expect_prn[column].values, df['OMEGA'].values)
            ec[column]= e

        elif column == 'sv_clock_bias' or column == 'sv_clock_drift' or column == 'sv_clock_drift_rate'  or  column == 'essentricity'  or column == 'sqrt_semi_major_axis'  or column == 'OMEGA' or column == 'inclination'   or column == 'OMEGA_dot' or column == 'inclination_rate' or column == 'codes' or column == 'gps_week' or column == 'l2_p_data_flag' or column == 'sv_accuracy' or column == 'sv_health' or column == 'tgd'  or column == 'fit_interval':
            df_t = pd.DataFrame()
            df_t['ds'] = df_final_prn['epoch_time']
            df_t['y'] = min_max_scaler_dict[column].transform(df_final_prn[column].values.reshape(-1, 1))
            model = Prophet()
            model.fit(df_t)
            df_p = pd.DataFrame()
            df_p['ds'] = df_expect_prn['epoch_time']
            df_predict_prn = model.predict(df_p)
            e  = {}
            e['mean_absolute_error'] = mean_absolute_error(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            e['mean_squared_error'] = mean_squared_error(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            e['percentError'] = percentError(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            ec[column]= e
            d = {}
            d['expected'] = min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1)
            d['predicted'] = df_predict_prn['yhat'].values
            d['index'] =  df_expect_prn['epoch_time'].values
            dc[column] = d
            original_dataframe[column] = df_expect_prn[column]
            final_dataframe[column] = pd.DataFrame(data = min_max_scaler_dict[column].inverse_transform(df_predict_prn['yhat'].values.reshape(-1, 1)), columns=['m'])

        elif column == 'iode' or column == 'correction_radius_sine' or  column == 'correction_latitude_cosine' or column == 't_tx' or column == 'correction_latitude_sine' or  column == 'correction_inclination_cosine' or   column == 'correction_radius_cosine' or column == 'mean_motion' or  column == 'correction_inclination_sine'   or column == 'omega':
            df_t = pd.DataFrame()
            df_t['ds'] = df_final_prn['epoch_time']
            df_t['y'] = min_max_scaler_dict[column].transform(df_final_prn[column].values.reshape(-1, 1))
            model = Prophet(changepoint_prior_scale=0.5)
            model.fit(df_t)
            df_p = pd.DataFrame()
            df_p['ds'] = df_expect_prn['epoch_time']
            df_predict_prn = model.predict(df_p)
            e  = {}
            e['mean_absolute_error'] = mean_absolute_error(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            e['mean_squared_error'] = mean_squared_error(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            e['percentError'] = percentError(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            ec[column]= e
            d = {}
            d['expected'] = min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1)
            d['predicted'] = df_predict_prn['yhat'].values
            d['index'] =  df_expect_prn['epoch_time'].values
            dc[column] = d
            original_dataframe[column] = df_expect_prn[column]
            final_dataframe[column] = pd.DataFrame(data = min_max_scaler_dict[column].inverse_transform(df_predict_prn['yhat'].values.reshape(-1, 1)), columns=['m'])

        elif column == 'time_of_ephemeris' or column == 'iodc' :
            df_t = pd.DataFrame()
            df_t['ds'] = df_final_prn['epoch_time']
            df_t['y'] = min_max_scaler_dict[column].transform(df_final_prn[column].values.reshape(-1, 1))
            model = Prophet(changepoint_prior_scale=1.0)
            model.fit(df_t)
            df_p = pd.DataFrame()
            df_p['ds'] = df_expect_prn['epoch_time']
            df_predict_prn = model.predict(df_p)
            e  = {}
            e['mean_absolute_error'] = mean_absolute_error(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            e['mean_squared_error'] = mean_squared_error(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            e['percentError'] = percentError(min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1), df_predict_prn['yhat'].values)
            ec[column]= e
            d = {}
            d['expected'] = min_max_scaler_dict[column].transform(df_expect_prn[column].values.reshape(-1, 1)).reshape(-1)
            d['predicted'] = df_predict_prn['yhat'].values
            d['index'] =  df_expect_prn['epoch_time'].values
            dc[column] = d
            original_dataframe[column] = df_expect_prn[column]
            final_dataframe[column] = pd.DataFrame(data = min_max_scaler_dict[column].inverse_transform(df_predict_prn['yhat'].values.reshape(-1, 1)), columns=['m'])
            
        else:
            final_dataframe[column] = df_expect_prn[column]
            original_dataframe[column] = df_expect_prn[column]
    error[prn]=ec
    plotdata[prn]=dc
    b = datetime.datetime.now()
    print(b-a)

#store the result
final_dataframe.to_csv('../../../../../pred_25_1_prn_2.csv')
original_dataframe.to_csv('../../../../../org_25_1_prn_2.csv')

#print error
final_errors = {}
for column in columns:
    total = 0.0
    for prn in prns:
        if column == "prn" or column == 'epoch_time':
            continue
        x =  error[prn][column]['mean_squared_error']
        total += x
    final_errors[column] = total/31
print(final_errors)

final_errors = {}
for column in columns:
    total = 0.0
    for prn in prns:
        if  column == "prn" or column == 'epoch_time':
            continue
        x =  error[prn][column]['mean_absolute_error']
        total += x
    final_errors[column] = total/31
print(final_errors)
