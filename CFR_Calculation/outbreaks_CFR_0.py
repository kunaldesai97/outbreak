'''Python file to cluster outbreaks and calculate CFR'''
import pandas as pd
from datetime import date
import numpy as np

def time_delta(X):
    '''Function to calculate time difference between two dates'''
    d1 = date(2020, 3, 25)
    a = X.split("-")
    year = int(a[0])
    month = int(a[1])
    day = int(a[2])
    d0 = date(year,month,day)
    delta = d1 - d0
    return delta

def main():
    outbreaks = pd.read_csv('../WHO_data_extraction/Outbreaks.csv')
    diseases = pd.read_csv('diseases.csv')
    incubation = dict(zip(list(diseases['Disease']), list(diseases['Maximum_Incubation'])))

    df = pd.merge(outbreaks, diseases, how='inner', on=['Disease'])
    df['days'] = df['Date'].apply(time_delta)
    df['outbreak_cluster'] = 0
    df.to_csv('cfr_df.csv')
    df = df.sort_values(['days'], ascending=[1])
    
    countries = list(df['Country'].unique())
    diseases_list = list(df['Disease'].unique())
    country_disease = {}
    count = 0

    df_cfr = pd.DataFrame(columns = ['Country', 'Date', 'Disease', 'ID', 'Month', 'Update', 'Year', 'Cases', 'Death', 'Pathogens', 'Disease_Scientific_Name', 'Vaccination',
       'Lethality', 'Transmission_Medium', 'Minimum_Incubation', 'Maximum_Incubation', 'Host_for_Parasite', 'Source', 'Age_Group', 'Symptoms', 'days', 'outbreak_cluster', 'cfr'])
    for country in countries:
        for disease in diseases_list:
            name = str(country.strip())+'_'+str(disease.strip())
            df_country=df.copy()
            df_country = df_country[df_country['Country']==country]
            df_country = df_country[df_country['Disease']==disease]
            country_disease[name] = len(df_country)
            global Maximum_Incubation

            Maximum_Incubation = incubation[disease]
            
            global outbreak_count 
            outbreak_count = 0

            def cluster_number(X):
                '''Function to assign cluster number to outbreak'''
                global outbreak_count
                global Maximum_Incubation
                day = X['delta_time']
                #print(Maximum_Incubation, outbreak_count)
                x = np.timedelta64(day, 'ns')
                days = x.astype('timedelta64[D]')
                #days / np.timedelta64(1, 'D')
                da = days.astype(int)
                if da > 3*Maximum_Incubation:
                    outbreak_count += 1
                
                return outbreak_count
            
            def cfr_assign(X):
                '''Function to calculate CFR for the clustered outbreaks'''
                global cfr
                cluster = X['outbreak_cluster']
                cfr_calc = cfr[cluster]

                return(cfr_calc)

            if len(df_country) != 0 and len(df_country) > 1:
                print(count, name, len(df_country))
                df_country['days_lagged'] = df_country['days'].shift(1)
                df_country['days_lagged'].iloc[0]= df_country['days'].iloc[0]
                df_country['delta_time'] = df_country['days'] - df_country['days_lagged']
                df_country['outbreak_cluster']  = df_country.apply(cluster_number, 1)                

                df_country1 = df_country.groupby(['outbreak_cluster']).agg({'Death':'max','Cases':'max'})
                df_country1['cfr'] = df_country1['Death']/df_country1['Cases']*100
            
                global cfr 
                cfr = dict(zip(list(df_country1.index), list(df_country1['cfr'])))
                df_country['cfr'] = df_country.apply(cfr_assign,1)
                count += 1
                df_trial = df_country[['Country', 'Date', 'Disease', 'ID', 'Month', 'Update', 'Year', 'Cases', 'Death', 'Pathogens', 'Disease_Scientific_Name', 'Vaccination',
       'Lethality', 'Transmission_Medium', 'Minimum_Incubation', 'Maximum_Incubation', 'Host_for_Parasite', 'Source', 'Age_Group', 'Symptoms', 'days', 'outbreak_cluster', 'cfr']]
                df_trial.to_csv('cfr_files/'+name+'.csv')
                df_cfr = df_cfr.append(df_trial)

            elif len(df_country) != 0:
                print(count, name, len(df_country))

                df_country['cfr'] = df_country['Death']/df_country['Cases']
                
                count += 1
                df_trial = df_country[['Country', 'Date', 'Disease', 'ID', 'Month', 'Update', 'Year', 'Cases', 'Death', 'Pathogens', 'Disease_Scientific_Name', 'Vaccination',
       'Lethality', 'Transmission_Medium', 'Minimum_Incubation', 'Maximum_Incubation', 'Host_for_Parasite', 'Source', 'Age_Group', 'Symptoms', 'days', 'outbreak_cluster', 'cfr']]
                df_trial.to_csv('cfr_files_1/'+name+'.csv')
                df_cfr = df_cfr.append(df_trial)

    df_cfr.to_csv('df_cfr.csv')

if __name__ == '__main__':
    main()
