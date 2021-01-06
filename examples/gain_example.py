import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fireman_imputation.src import utils
from fireman_imputation.gain_training import gain_train
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load and scale the data
data_orig = pd.read_csv('#datasets/Tennessee_Event-Driven/tep_train_extended.csv')
data_orig = data_orig[data_orig['simulationRun'] == 1].drop(columns=['faultNumber',
                                                                     'simulationRun',
                                                                     'sample']).values
data = data_orig.values

# create missing data
data_missing, mask = utils.mcar_gen(data, 0.1)
# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_missing)
data_missing = scaler.transform(data_missing)

# divide the data to train/test
# by default shuffles data, if pandas is passed the index shows shuffle result
data_missing_train, data_missing_test, data_train, data_test = train_test_split(data_missing, data, train_size=0.9)

# set hyper-parameters
gain_params = {'batch_size': 100,
               'hint_rate': 0.9,
               'alpha': 100,
               'epochs': 10,
               'learning_rate': 0.001}

gen, disc = gain_train(gain_params, data_missing_train, cont=False)

data_missing_test_torch, mask_test_torch = utils.gain_data_prep(data_missing_test)
data_imputed_test = gen(data_missing_test_torch, mask_test_torch)
data_imputed_test = data_imputed_test.detach().numpy()

# merge the imputed data(zero out rest in imputed data) and data with missing values
inv_mask_test = 1 - mask_test_torch.numpy()
data_missing_test_0 = data_missing_test.copy()
data_missing_test_0[np.isnan(data_missing_test_0)] = 0
# data_missing_test contains nan values so we need to either =0 or *mask
data_imputed_test = inv_mask_test*data_imputed_test + data_missing_test_0

# rescale the imputed data
data_imputed_test = scaler.inverse_transform(data_imputed_test)
data_missing_test = scaler.inverse_transform(data_missing_test)

# compute error
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
RMSE = mean_squared_error(data_test, data_imputed_test, squared=True)
print('RMSE of test dataset is {}'.format(RMSE))

# create pandas for easier visualization
data_test_pd = pd.DataFrame(data_test, columns=data_orig.columns)
data_missing_test_pd = pd.DataFrame(data_missing_test, columns=data_orig.columns)
data_imputed_test_pd = pd.DataFrame(data_imputed_test, columns=data_orig.columns)

# matplotlib
column = 'Cont 57'
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
data_test_pd[column].plot(ax=ax, style='--', alpha=0.6, color='red')
data_imputed_test_pd[column].plot(ax=ax, alpha=0.8, color='black')

x = data_test_pd[data_missing_test_pd[column].isna()][column].index.values
y = data_test_pd[data_missing_test_pd[column].isna()][column].values
plt.scatter(x, y);
