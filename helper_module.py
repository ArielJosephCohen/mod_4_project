"""
Import all necessary libraries
"""
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import RFE
from numpy.polynomial.polynomial import polyfit
from pandas import Series
import matplotlib.pylab as plt
import datetime
from dateutil.relativedelta import relativedelta
from matplotlib.pylab import rcParams
from statsmodels.compat import lzip

def clear_null_features(dataframe,features,value):
    """
    Input a dataframe and filter out problematic values by imputing a dummy value for later deletion
    """
    for feature in features:
        dataframe[feature].fillna(value,inplace=True)
        dataframe = dataframe[dataframe[feature]!=value]
    return dataframe

def add_dummy_columns(dataframe,feature):
    """
    Add dummy columns to a data frame based on a feature
    """
    dummy_df = pd.get_dummies(dataframe[feature])
    dummy_df = dummy_df.iloc[:,1:]
    dataframe_atg = dataframe.copy()
    for col in dummy_df.columns:
        dataframe_atg[col]=dummy_df[col]
    dataframe_atg.drop(feature,axis=1,inplace=True)
    return dataframe_atg

def filter_outliers(dataframe,subset,threshold):
    """
    Filter outliers based on a certain number of standard deviations
    """
    dataframe = dataframe[(np.abs(stats.zscore(subset)) <= threshold).all(axis=1)]
    return dataframe

def normalize_data(dataframe,value):
    """
    Normalize data using Box-Cox with the option to add a constant to the transform to make things easier
    """
    for feature in dataframe.columns:
        dataframe[feature] = list(stats.boxcox(abs(dataframe[feature]+value)))[0]
    return dataframe

def scale_data(dataframe):
    """
    Enter a data frame and scale all features to be between 0 and 1
    """
    mm = MinMaxScaler()
    for feature in dataframe.columns:
        if (dataframe[feature]>=1).sum()>0:
            dataframe[feature] = mm.fit_transform(dataframe[[feature]])
    return dataframe


def show_correlation(dataframe,threshold):
    """
    Input a data frame and check correlation among custom features based on a custom threshold
    """
    plt.figure(figsize=(5,3))
    plt.tight_layout()
    sns.heatmap(dataframe.corr()>=threshold)
    plt.show()

def messy_model(X_tr,y_tr,intercept=True):
    """
    Check a simple training model with the ability to use od not use intercept
    """
    if intercept == True:
        X_int_train = sm.add_constant(X_tr)
        mod = sm.OLS(y_tr,X_int_train).fit()
        result = mod.summary()
    else:
        mod=sm.OLS(y_tr,X_tr)
        res=feat_mod.fit()
        result = res.summary()
    return result

def ridge_reg_score(X_tr,y_tr,X_val,y_val,alpha):
    """
    Check scores after applying ridge regularization
    """
    ridge_reg = Ridge(alpha=0.05)
    ridge_reg.fit(X_tr,y_tr)
    ridge_coef = pd.DataFrame(ridge_reg.coef_).T
    ridge_coef.columns = X_tr.columns
    ridge_score = ridge_reg.score(X_val,y_val)
    return ridge_score,ridge_coef