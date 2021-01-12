import numpy as np
import pandas as pd
import pickle
import argparse

# Specifying filter window for the FIR filter
filter_window = 3

def learn_parameters(df,counter=[0]):
    country_wise_split = np.array(df)
    M, N = country_wise_split.shape
    recovery_rates = np.zeros(M)
    transmission_rates = np.zeros(M)
    for i in range(M - 1):
        try:
            delta_recovery = country_wise_split[i + 1, 4] - country_wise_split[i, 4]
            infected_t = country_wise_split[i, 2]
            recovery_rates[i] = delta_recovery / infected_t
            dIdT = (country_wise_split[i + 1, 2] - infected_t) / infected_t
            transmission_rates[i] = dIdT + recovery_rates[i]
        except ZeroDivisionError:
            recovery_rates[i] = 0
            transmission_rates[i] = 0
    df['trate'] = transmission_rates
    df['rrate'] = recovery_rates

    if counter[0] == 0:
        with open('mypickle.pickle','wb') as f:
            pickle.dump(df,f)

    elif counter[0] > 0:
            df_pickled = pd.read_pickle('mypickle.pickle')
            df_new = df_pickled.append(df)
            with open('mypickle.pickle','wb') as f:
                pickle.dump(df_new,f)

    counter[0] += 1

def features_SIR_model(df):
    country_wise_split = np.array(df)
    tr = country_wise_split[:,5]
    rr = country_wise_split[:,6]
    M,N = country_wise_split.shape
    features_tr = []
    features_rr = []
    for i in range(M):
        if M < filter_window:
            features_tr.append(list(np.append(1, np.zeros(filter_window))))
            features_rr.append(list(np.append(1, np.zeros(filter_window))))
        else:
            if i < filter_window:
                features_tr.append(list(np.append(1, np.zeros(filter_window))))
                features_rr.append(list(np.append(1, np.zeros(filter_window))))

            else:
                features_tr.append(list(np.append(1,np.flip(tr[(i-filter_window):i]))))
                features_rr.append(list(np.append(1,np.flip(rr[(i-filter_window):i]))))

    df['tr_features'] = list(features_tr)
    df['rr_features'] = list(features_rr)

    return df

def main(input,countries):
    df= pd.read_csv(input)
    df_grouped = df.groupby(['ObservationDate','Country/Region'])['Confirmed','Deaths','Recovered'].agg(sum).reset_index()
    if countries:
        df_grouped = df_grouped[df_grouped['Country/Region'].isin(countries)]
    df_group_country = df_grouped.groupby(['Country/Region']).apply(lambda x:learn_parameters(x))
    df_with_params = pd.read_pickle('mypickle.pickle')
    df_with_params = df_with_params.drop_duplicates()
    df_features = df_with_params.groupby(['Country/Region']).apply(lambda x: features_SIR_model(x))
    SIR_parameters = pd.merge(df_features,df_grouped,on=['ObservationDate','Country/Region','Confirmed','Deaths','Recovered',])

    #Exploding the dataframe to 7 dimensions for transmission rate and recovery rate
    for i in range(filter_window+1):
        SIR_parameters['tr_feature{}'.format(str(i + 1))] = [k[i] if isinstance(k, list) else k for k in SIR_parameters['tr_features']]
        SIR_parameters['rr_feature{}'.format(str(i + 1))] = [k[i] if isinstance(k, list) else k for k in SIR_parameters['tr_features']]

    SIR_parameters[['ObservationDate','Country/Region','Confirmed','Deaths','Recovered','trate','rrate',
                    'tr_feature1','tr_feature2','tr_feature3','tr_feature4',
                    'rr_feature1','rr_feature2','rr_feature3','rr_feature4']].to_csv('covid_SIR.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculating features for SIR model:')
    parser.add_argument('Path', metavar='path', type=str,nargs='+',help='Input csv path')
    parser.add_argument('-c','--list',nargs='*',help='Pass the country filters as strings.',required=False)
    """How to pass the argument - python3 SIR_model_covid.py covid_data.csv -c Kuwait Macau"""
    args = parser.parse_args()
    d = vars(args)
    input = args.Path[0]
    countries = d['list']
    main(input,countries)
