from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from sklearn import ensemble
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from TimeBasedCV import TimeBasedCV

def learning_beta(X1,y1):
    """This function is the driving function to learning transmission rate. Based on the training data
        we predict the test data values for the transmission rate(gamma) using GBT Regressor Model"""
    #GBT - Training Period 45, Test Period 10
    tscv = TimeBasedCV(train_period=40,test_period=10,freq='days')

    scores = []
    for train_index, test_index in tscv.split(X1, validation_split_date=None,date_column='ObservationDate'):

        data_train = X1.loc[train_index].drop('ObservationDate', axis=1)
        target_train = y1.loc[train_index]

        data_test = X1.loc[test_index].drop('ObservationDate',axis=1)
        target_test = y1.loc[test_index]

        #Ridge Regression Model
        # parameters = {'alpha': np.arange(0, 1, 0.01)}
        # ridge = Ridge()
        # ridge.fit(data_train, target_train)
        # predictions_beta = ridge.predict(data_test)
        # r2score = ridge.score(data_test,target_test)
        # grid_fit_beta = GridSearchCV(clf, params).fit(data_train, target_train)
        # best_clf_beta = grid_fit_beta.best_estimator_
        # print(grid_fit_beta.best_params_)
        # r2score = ridge.score(data_test,target_test)

        #Gradient Boosting Tree Regressor
        params = {'n_estimators': 80, 'max_depth': 4, 'min_samples_split': 3,
                  'learning_rate': 0.005, 'loss': 'ls'}
        model = ensemble.GradientBoostingRegressor(**params)
        parameters = {'loss':('ls','lad','huber','quantile')}

        grid_fit_gbr = GridSearchCV(model, parameters).fit(data_train, target_train)
        best_clf_gbr = grid_fit_gbr.best_estimator_
        predictions_beta = best_clf_gbr.predict(data_test)
        mse_beta = mean_squared_error(target_test, predictions_beta)
        scores.append(mse_beta)

        sns.lineplot(x=np.arange(1, len(data_test) + 1), y=predictions_beta, label='Prediction')
        plt.ylim(0, 0.8)
        sns.lineplot(x=np.arange(1, len(data_test) + 1), y=target_test, label='Original')
        plt.xlabel('Days')
        plt.ylabel('Transmission Rate')
        plt.title('Predicted transmission rate on test set')
        plt.show()

    average_mse = np.mean(scores)

    beta_avg = np.mean(predictions_beta)

    return beta_avg, average_mse

def learning_gamma(X2,y2):
    """This function is the driving function to learning recovery rate. Based on the training data
        we predict the test data values for the recovery rate(gamma) using GBT Regressor Model"""
    #GBT - Training Period 45, Test Period 10
    tscv = TimeBasedCV(train_period=40,test_period=10,freq='days')

    scores = []
    for train_index, test_index in tscv.split(X2, validation_split_date=None,date_column='ObservationDate'):

        data_train = X2.loc[train_index].drop('ObservationDate', axis=1)
        target_train = y2.loc[train_index]

        data_test = X2.loc[test_index].drop('ObservationDate',axis=1)
        target_test = y2.loc[test_index]

        #Ridge Regression Model
        # parameters = {'alpha': np.arange(0, 1, 0.01)}
        # ridge = Ridge()
        # ridge.fit(data_train, target_train)
        # predictions_beta = ridge.predict(data_test)
        # r2score = ridge.score(data_test,target_test)
        # grid_fit_beta = GridSearchCV(clf, params).fit(data_train, target_train)
        # best_clf_beta = grid_fit_beta.best_estimator_
        # print(grid_fit_beta.best_params_)
        # r2score = ridge.score(data_test,target_test)


        #Gradient Boosting Tree Regressor
        params = {'n_estimators': 80, 'max_depth': 4, 'min_samples_split': 3,
                  'learning_rate': 0.005, 'loss': 'ls'}
        model = ensemble.GradientBoostingRegressor(**params)
        parameters = {'loss':('ls','lad','huber','quantile')}

        grid_fit_gbr = GridSearchCV(model, parameters).fit(data_train, target_train)
        best_clf_gbr = grid_fit_gbr.best_estimator_
        predictions_gamma = best_clf_gbr.predict(data_test)
        mse_gamma = mean_squared_error(target_test, predictions_gamma)
        scores.append(mse_gamma)

        sns.lineplot(x=np.arange(1, len(data_test) + 1), y=predictions_gamma, label='Prediction')
        plt.ylim(0, 0.18)
        sns.lineplot(x=np.arange(1, len(data_test) + 1), y=target_test, label='Original')
        plt.xlabel('Days')
        plt.ylabel('Recovery Rate')
        plt.title(f'Predicted recovery rate on test set')
        plt.show()

    average_mse = np.mean(scores)
    print(predictions_gamma)
    gamma_avg = np.mean(predictions_gamma)

    return gamma_avg, average_mse

def infectivity():
    """To model the trend of infectivity based on the learnt transmission rates using SIR Model"""
    orig_data = pd.read_csv('covid_SIR.csv')
    orig_infectivity = orig_data['Confirmed']
    orig_recovered = orig_data['Recovered']
    orig_data_I = orig_infectivity.iloc[39:]
    orig_data_R = orig_recovered.iloc[39:]

    # Total population, N.
    N = 37742154
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 12978, 2577
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

    beta, gamma = 0.24757, 0.0055  # Inline with the current trend
    beta1, gamma1 = 0.216991, 0.0028  # Predicted transmission values

    # A grid of time points (in days)
    t = np.linspace(0, 75, 75)

    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    y1 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    ret1 = odeint(deriv, y1, t, args=(N, beta1, gamma1))

    S, I, R = ret.T
    S1, I1, R1 = ret1.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, I / 1000, 'r', alpha=0.5, lw=2, label='Observed trend')
    ax.plot(t, I1 / 1000, 'g', alpha=0.5, lw=2, label='Model Predicted')
    ax.plot(np.arange(0, len(orig_data_I)), np.array(orig_data_I), 'p', alpha=0.5, lw=2, label='Actual Data')
    ax.set_xlabel('Time in days of the Disease Outbreak')
    ax.set_ylabel('Population in 1000s')
    # plt.text(8,3,'This text ends at point (8,3)',horizontalalignment='right')
    plt.text(x=8, y=28000, s=f'Actual Beta: {beta}', horizontalalignment='left')
    plt.text(x=30, y=7550, s=f'Learned Beta: {beta1}')
    plt.title('Infected Numbers in US (COVID-19)')
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend(loc='upper left')
    # legend.get_frame().set_alpha(0.5)
    # plt.tight_layout()
    plt.show()

def recovery():
    """To model the trend of recovery based on the learnt transmission rates using SIR Model"""

    orig_data = pd.read_csv('covid_SIR.csv')
    orig_recovered = orig_data['Recovered']
    orig_data_R = orig_recovered.iloc[39:]

    # Total population, N.
    N = 37742154
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 12978.0, 2577.0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

    beta, gamma = 0.24757, 0.04508  # Inline with the current trend
    beta1, gamma1 = 0.216991, 0.0028  # Predicted transmission values

    # A grid of time points (in days)
    t = np.linspace(0, 75, 75)

    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    y1 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    ret1 = odeint(deriv, y1, t, args=(N, beta1, gamma1))

    S, I, R = ret.T
    S1, I1, R1 = ret1.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, R/1000, 'r', alpha=0.5, lw=2, label='Observed Trend')
    ax.plot(t, R1/1000, 'g', alpha=0.5, lw=2, label='Predicted Recovered')
    ax.plot(np.arange(0,len(orig_data_R)), np.array(orig_data_R),'s',alpha=0.5, lw=2, label='Actual Recovered')
    ax.set_xlabel('Time in days of the Disease Outbreak')
    ax.set_ylabel('Population in 1000s')
    plt.text(x=27, y=20000, s=f'Actual Gamma: {gamma}', horizontalalignment='left')
    plt.text(x=40, y=2500, s=f'Learned Gamma:{gamma1}',rotation=8)
    plt.title('Recovered Numbers in US (COVID-19)')
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':

    df = pd.read_csv('covid_SIR.csv',parse_dates=['ObservationDate'])

    X1 = df.iloc[7:,-8:-4]   #Data - Transmission Rates
    X2 = df.iloc[7:,-12:-8]   #Data Recovery rates
    X1['ObservationDate'] = df['ObservationDate']
    X2['ObservationDate'] = df['ObservationDate']
    y1 = df['trate']
    y2 = df['rrate']
    beta, error_b= learning_beta(X1,y1)
    print("The average value of learnt beta is {0} with and MSE of {1}".format(beta,error_b))
    gamma, error_g = learning_gamma(X2,y2)
    print("The average value of learnt gamma is {0} with an MSE of {1}".format(gamma,error_g))
    infectivity()
    recovery()