# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:16:49 2019

@author: Alex
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

#%%  Import Data

data = pd.read_excel('data/1hr/trn_test/trn1.xlsx')
data = data.loc[data.DayCnt>0]#.sample(500)
#data = data.head(5000)
X = data['Arrivals'].values 
dataTest = pd.read_excel('data/1hr/trn_test/test1.xlsx')
T = dataTest['Arrivals'].values

# Convert categorical variables to integer
hrs_idx = data['Hour'].values
hrs = np.arange(24)
n_hrs = len(hrs)

# Setup Bayesian Hierarchical Model 
with pm.Model() as countModel:
    
    # Define Hyper-Parameters
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=np.round(np.mean(X) + 3*np.std(X)))
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=np.round(np.mean(X) + 3*np.std(X))) 
    
    # Prior Definition
    mu = pm.Gamma('mu', mu=hyper_mu_mu, 
                        sigma=hyper_mu_sd,
                        shape=n_hrs)    
    
    # Data Likelihood
    y_like = pm.Poisson('y_like', 
                       mu=mu[hrs_idx], 
                       observed=X)   
    
    # Data Prediction
#    y_pred = pm.Poisson('y_pred', 
#                        mu=mu[hrs_idx], 
#                        shape=X.shape)
    
pm.model_to_graphviz(countModel)
    
#%% Hierarchical Model Inference

# Setup vars
smpls = 1000; burnin = 3000;

# Print Header
print('Poisson Likelihood')
print('Params: samples = ', smpls, ' | tune = ', burnin, '\n')
        
with countModel:
    trace = pm.sample(smpls, chains=4, tune=burnin, cores=1)#, NUTS={"target_accept": targetAcc})
    
    ppc = pm.sample_posterior_predictive(trace)
    #pm.traceplot(trace[burnin:], var_names=['mu'])                  

out_smryPoi = pd.DataFrame(pm.summary(trace))  

#%% #% TracePlot

import arviz as az

plt.style.use('default')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

plt.figure(figsize=(10,4))
pm.traceplot(trace[0::4], var_names=['mu'])  

plt.figure(figsize=(10,4))
az.plot_posterior(trace[0::4])

plt.figure(figsize=(10,4))
az.plot_forest(trace[0::4], var_names=['mu'], combined=True)

#%% Train vs. Test Data

dataTest = pd.read_excel('data/1hr/trn_test/test4.xlsx')
T = dataTest['Arrivals'].values

s = 10000; t = len(T);
r = np.shape(ppc['y_like'])[0]; c = np.shape(ppc['y_like'])[1];
ppc_Smpl = pd.DataFrame(np.reshape(ppc['y_like'], (r*c,1)))
ppc_Smpl['Hour'] = np.tile(hrs_idx,r)
ppc_Smpl = ppc_Smpl.sample(s)

# Setup PPC
ppcTest = pd.DataFrame(np.zeros((s+t,3)), columns=['Hr', 'Departures', 'Src'])
ppcTest.Hr[0:s] = ppc_Smpl['Hour']
ppcTest.Departures[0:s] = ppc_Smpl[0]
ppcTest.Src[0:s] = 'PPC'
aggTrn = ppcTest[0:s]

# Setup test
ppcTest.Hr[s:s+t] = dataTest.Hour
ppcTest.Departures[s:s+t] = T
ppcTest.Src[s:s+t] = 'Test'
aggTest = ppcTest[s:s+t]

#% SNS Relationship Plot

import seaborn as sns

plt.style.use('default')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

g_Rel = sns.relplot(x='Hr', y='Departures', kind='line',
                 hue='Src', ci='sd', 
                 data=ppcTest)

g_Rel.fig.set_size_inches(12,6)
plt.xticks(np.arange(0,28,4))
plt.yticks(np.arange(0,18,2))

g_Data = pd.DataFrame(np.zeros((24,3)), columns=['Hr','Trn','Test'])

#%% Error Measure
import arviz as az
    
def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

font = {'family': 'Times New Roman', 'weight': 'light', 'size': 1}
plt.rc('font', **font)
g_Trn = sns.relplot(x='Hr', y='Departures', kind='line',
                 hue='Src', ci='sd', data=aggTrn)
g_Trn.fig.set_size_inches(0.1,0.1)

for ax in g_Trn.axes.flat:    
    for line in ax.lines:
        if len(line.get_xdata()) == 24:
            g_Data.Hr = line.get_xdata();
            g_Data.Trn = line.get_ydata();
    
g_Test = sns.relplot(x='Hr', y='Departures', kind='line',
                 hue='Src', ci='sd', data=aggTest)
g_Test.fig.set_size_inches(0.1,0.1)

for ax in g_Test.axes.flat:    
    for line in ax.lines:
        if len(line.get_xdata()) == 24:
            g_Data.Hr = line.get_xdata();
            g_Data.Test = line.get_ydata();

print('SMAPE: ', SMAPE(g_Data.Trn, g_Data.Test))

#print('r2 Trn: ', az.r2_score(X, np.array(ppc_Smpl[0].sample(len(X))))[0])
#print('r2 Test: ', az.r2_score(T, np.array(ppc_Smpl[0].sample(len(T))))[0])

