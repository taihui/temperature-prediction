import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.random import randn

def discover_rawdata():
    data = pd.read_csv("dataset/sudeste.csv")
    #plt.plot("scatter", data['lon'], data['lat'])
    #plt.show()
    station_list = data['wsid'].unique()
    station_dict = {}
    station_dict_drop_na = {}
    for sid in station_list:
        station_data = data[data['wsid']== sid]
        station_data_drop_na = station_data.dropna()
        station_dict[sid] = len(station_data)
        station_dict_drop_na[sid] = len(station_data_drop_na)

    sorted_station_dict = sorted(station_dict.items(), key=lambda x: x[1])
    sorted_station_dict_drop_na = sorted(station_dict_drop_na.items(), key=lambda x: x[1])


    sorted_station_dict_file = 'doc/sorted_station_dict.txt'
    sorted_station_dict_drop_na_file = 'doc/sorted_station_dict_drop_na.txt'

    ssdf = open(sorted_station_dict_file,'a')
    for item in sorted_station_dict:
        ssdf.write(str(item[0]) + "," + str(item[1]) + "\n")
    ssdf.close()

    ssdf_dn = open(sorted_station_dict_drop_na_file,'a')
    for item in sorted_station_dict_drop_na:
        ssdf_dn.write(str(item[0]) + "," + str(item[1]) + "\n")
    ssdf_dn.close()

def temp_distribution(datafile = "dataset/station_394raw.csv"):
    data = pd.read_csv(datafile)

    yr_data_2006 = data[data['yr'] == 2006]
    temperature_2006 = yr_data_2006['temp'].tolist()

    yr_data_2013 = data[data['yr'] == 2013]
    temperature_2013 = yr_data_2013['temp'].tolist()

    yr_data_2014 = data[data['yr'] == 2014]
    temperature_2014 = yr_data_2014['temp'].tolist()

    yr_data_2015 = data[data['yr'] == 2015]
    temperature_2015 = yr_data_2015['temp'].tolist()

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.hist(temperature_2006)
    plt.title('Temperature Distribution (2006)')
    plt.xlabel('Celsius Degree')
    plt.ylabel('Hours')


    plt.subplot(2, 2, 2)
    plt.hist(temperature_2013)
    plt.title('Temperature Distribution (2013)')
    plt.xlabel('Celsius Degree')
    plt.ylabel('Hours')


    plt.subplot(2, 2, 3)
    plt.hist(temperature_2014)
    plt.title('Temperature Distribution (2014)')
    plt.xlabel('Celsius Degree')
    plt.ylabel('Hours')


    plt.subplot(2, 2, 4)
    plt.hist(temperature_2015)
    plt.title('Temperature Distribution (2015)')
    plt.xlabel('Celsius Degree')
    plt.ylabel('Hours')

    plt.tight_layout()

    plt.savefig('figure/temperature_distribution.png')
    #plt.show()

def station_record_distribution (raw_file = 'doc/sorted_station_dict.txt', dropna_file = 'doc/sorted_station_dict_drop_na.txt'):
    raw_f = open(raw_file, 'r')
    dropna_f = open(dropna_file, 'r')

    raw_line = raw_f.readline().strip()
    raw_dict = {}
    while raw_line:
        raw_line_arr = raw_line.split(',')
        raw_dict[int(raw_line_arr[0])] = int(raw_line_arr[1])
        raw_line = raw_f.readline().strip()

    dropna_line = dropna_f.readline().strip()
    dropna_dict = {}
    while dropna_line:
        dropna_line_arr = dropna_line.split(',')
        dropna_dict[int(dropna_line_arr[0])] = int(dropna_line_arr[1])
        dropna_line = dropna_f.readline().strip()
    raw_value = []
    dropna_value = []
    key_list = sorted (raw_dict.keys())
    key_len = len(key_list)
    for cur_key in key_list:
        raw_value.append(raw_dict[cur_key])
        dropna_value.append(dropna_dict[cur_key])

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.scatter(key_list, raw_value)
    #plt.hist(raw_value,bins=122)
    plt.title('Record Distribution (Raw)')
    plt.xlabel('Station ID')
    plt.ylabel('Numbers')


    plt.subplot(1, 2, 2)
    plt.scatter(key_list, dropna_value)
    #plt.hist(dropna_value, bins=122)
    plt.title('Record Distribution (DropNAN)')
    plt.xlabel('Station ID')
    plt.ylabel('Numbers')

    plt.tight_layout()

    plt.savefig('figure/record_distribution.png')




def get_station_data(sid):
    data = pd.read_csv("dataset/sudeste.csv")
    # get a particular station
    new_data = data[data['wsid'] == sid]
    new_data.to_csv("dataset/station_" + str(sid) + "raw.csv")
    print(len(new_data))


def fill_NAN(data):
    # fill null for prcp
    prcp_mean = data["prcp"].mean()
    data['prcp'].fillna(prcp_mean, inplace=True)

    # fill null for gbrd
    gbrd_mean = data["gbrd"].mean()
    data["gbrd"].fillna(gbrd_mean, inplace=True)

    # fill null for temp
    temp_mean = data["temp"].mean()
    data["temp"].fillna(temp_mean, inplace=True)

    # fill null for dewp
    dewp_mean = data["dewp"].mean()
    data["dewp"].fillna(dewp_mean, inplace=True)

    # fill null for tmax
    tmax_mean = data["tmax"].mean()
    data["tmax"].fillna(tmax_mean, inplace=True)

    # fill null for dmax
    dmax_mean = data["dmax"].mean()
    data["dmax"].fillna(dmax_mean, inplace=True)

    # fill null for tmin
    tmin_mean = data["tmin"].mean()
    data["tmin"].fillna(tmin_mean, inplace=True)

    # fill null for dmin
    dmin_mean = data["dmin"].mean()
    data["dmin"].fillna(dmin_mean, inplace=True)

    # fill null for wdsp
    wdsp_mean = data["wdsp"].mean()
    data["wdsp"].fillna(wdsp_mean, inplace=True)

    # fill null for gust
    gust_mean = data["gust"].mean()
    data["gust"].fillna(gust_mean, inplace=True)

    return data

def data_clean(datafile = "dataset/station_394raw.csv"):
    data = pd.read_csv(datafile)
    # select the year 2005/2006/2007
    yr_data = data[(data['yr'] >= 2013) & (data['yr'] <= 2015)]

    # raw data summary
    print("Raw data summary:")
    print("The number of raw data records:" + str(len(yr_data)))
    print(yr_data.isna().sum())
    print(" ")

    for yr in [2013, 2014, 2015]:
        temp_data = data[data['yr'] == yr]
        print('Year:' + str(yr) + "; " + str(len(temp_data)))
        print(temp_data.isna().sum())
        print(" ")


    # deal with the NAN: get the day's mean and them replace the NAN with the day's mean
    yr_list = [2013, 2014, 2015]
    mn_list = [1,2,3,4,5,6,7,8,9,10,11,12]
    data_arr = []

    for yr in yr_list:
        for mn in mn_list:
            if mn in [1,3,5,7,8,10,12]: #31days
                for dnum in range(1,32):
                    fill_data = yr_data[(yr_data['yr'] == yr) & (yr_data['mo'] == mn) & (yr_data['da'] == dnum)]
                    fill_data = fill_NAN(fill_data)
                    data_arr.append(fill_data)

            elif mn in [4,6,9,11]: #30days
                for dnum in range(1, 31):
                    fill_data = yr_data[(yr_data['yr'] == yr) & (yr_data['mo'] == mn) & (yr_data['da'] == dnum)]
                    fill_data = fill_NAN(fill_data)
                    data_arr.append(fill_data)
            else:
                for dnum in range(1, 29):
                    fill_data = yr_data[(yr_data['yr'] == yr) & (yr_data['mo'] == mn) & (yr_data['da'] == dnum)]
                    fill_data = fill_NAN(fill_data)
                    data_arr.append(fill_data)

    target_data = pd.concat(data_arr,axis=0)
    target_data = target_data.reset_index(drop=True)
    #target_data['mdct'] = pd.to_datetime(target_data['mdct'])
    target_data.sort_values('mdct', inplace=True)


    #selected_data = target_data[column_name]
    target_data.to_csv("dataset/station_394clean.csv", index= False)

    # raw data summary
    print("Raw data summary:")
    print("The number of raw data records:" + str(len(target_data)))
    print(target_data.isna().sum())
    print(" ")

def discover_seldata(selfile = "dataset/station_394clean.csv"):
    data = pd.read_csv(selfile)
    data_length = len(data)
    for i in range(data_length):
        temp_value = data.loc[i:i, 'prcp'].tolist()
        try:
            int(temp_value[0])
        except:
            if i == 0:
                data.loc[i:i, 'prcp'] = float(0)
            else:
                fill_value = data.loc[i-1:i-1, 'prcp'].tolist()
                data.loc[i:i, 'prcp'] = fill_value[0]
    data.to_csv("dataset/station_394clean_full.csv", index=False)

    # raw data summary
    print("Raw data summary:")
    print("The number of raw data records:" + str(len(data)))
    print(data.isna().sum())
    print(" ")


def map_temp_class(selfile = "dataset/station_394clean_full.csv"):
    data = pd.read_csv(selfile)
    temperature = data['temp'].tolist()
    plt.hist(temperature)
    plt.title('Temperature Distribution')
    plt.xlabel('Celsius Degree')
    plt.ylabel('Hours')
    plt.show()
    #plt.savefig('temperature distribution.pdf')

def display_tempInfo (file = "dataset/station_394clean_full.csv"):
    #'mdct',
    data = pd.read_csv(file)
    temp = data['temp'].tolist()
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(range(len(temp)), temp)
    plt.title('Temperature Range (2013-2015)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (Celsius Degree)')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(temp[0:168])), temp[0:168])
    plt.title('Temperature Range (7 days)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (Celsius Degree)')


    plt.tight_layout()


    plt.savefig('figure/temperature_range.png')




def select_feature (file = "dataset/station_394clean_full.csv"):
    data = pd.read_csv(file)
    train_length = len(data)
    #'mdct',
    #column_name_X = ['prcp', 'stp', 'smax', 'smin', 'gbrd', 'dewp','tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']
    column_name_X = ['prcp', 'stp', 'smax', 'smin', 'gbrd', 'temp', 'dewp','tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust']
    column_name_Y = ['temp']

    selected_data_X = data[column_name_X]
    selected_data_Y = data[column_name_Y]
    # normalized the features
    # (data - Mean(data)) / (Std(data))

    raw_length = len(selected_data_X)
    train_length = int(0.8 * raw_length)
    test_length = raw_length - train_length



    train_X = selected_data_X.iloc[0:train_length]
    train_Y = selected_data_Y.loc[0:train_length]

    test_X = selected_data_X.loc[train_length:raw_length]
    test_Y = selected_data_Y.loc[train_length:raw_length]

    train_X.to_csv("dataset/station_394_train_X_V2.csv", index=False)
    train_Y.to_csv("dataset/station_394_train_Y.csv", index=False)

    test_X.to_csv("dataset/station_394_test_X_V2.csv", index=False)
    test_Y.to_csv("dataset/station_394_test_Y.csv", index=False)


def normalize_data(Xfile="dataset/station_394_train_X.csv", Yfile="dataset/station_394_train_Y.csv"):
    train_X = pd.read_csv(Xfile)
    train_Y = pd.read_csv(Yfile)

    normalized_X = (train_X - train_X.mean()) / train_X.std()
    normalized_Y = (train_Y - train_Y.mean())/ train_Y.std()

    plt.figure(1)

    plt.boxplot((train_X['prcp'].tolist()
                 , train_X['stp'].tolist()
                 , train_X['smax'].tolist()
                 , train_X['smin'].tolist()
                 , train_X['gbrd'].tolist()
                 , train_X['dewp'].tolist()
                 , train_X['tmax'].tolist()
                 , train_X['dmax'].tolist()
                 , train_X['tmin'].tolist()
                 , train_X['dmin'].tolist()
                 , train_X['hmdy'].tolist()
                 , train_X['hmax'].tolist()
                 , train_X['hmin'].tolist()
                 , train_X['wdsp'].tolist()
                 , train_X['wdct'].tolist()
                 , train_X['gust'].tolist()), labels=('pp',
                                                              'sp',
                                                              'smx',
                                                              'smn',
                                                              'gd',
                                                              'dp',
                                                              'tmx',
                                                              'dmx',
                                                              'tmn',
                                                              'dmn',
                                                              'hy',
                                                              'hmx',
                                                              'hmn',
                                                              'wp',
                                                              'wt',
                                                              'gt'))
    plt.title('Boxplot for Features (Raw)')
    plt.xlabel('Feature Name')
    plt.ylabel('Value')



    plt.tight_layout()
    plt.savefig('figure/feature_boxplot.png')






if __name__ == '__main__':
    #get_station_data(178)
    #data_clean()
    #discover_rawdata()
    #get_station_data(394)
    #data_clean()
    #discover_seldata()
    #map_temp_class()
    #temp_distribution()
    #station_record_distribution()
    #data_clean()
    #discover_seldata()
    #select_feature()
    #display_tempInfo()
    #select_feature()
    select_feature()
    pass



