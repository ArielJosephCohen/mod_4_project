def import_libraries():
    '''
    This function imports all necessary libraries for the Jupyter Notebook
    '''
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
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
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    import datetime
    from dateutil.relativedelta import relativedelta
    from statsmodels.graphics.tsaplots import plot_pacf
    from statsmodels.graphics.tsaplots import plot_acf
    from matplotlib.pylab import rcParams
    from statsmodels.tsa.arima_model import ARMA
    from statsmodels.compat import lzip
    
def import_data(csv):
    '''
    This function imports the data set to be used
    '''
    combine_data = pd.read_csv(csv)
    print(combine_data.shape)

def drop_and_fill_na(columns):
    '''
    This function will fill all null values and drop the ones that are equal to zero
    '''
    pass
    
def drop_columns(columns):
    '''
    This function will drop columns not important for the analysis
    '''
    pass
    
def create_categorical_data(columns):
    '''
    This function will create dummy variables and add them to the data set
    '''
    pass
    
def drop_correlation(columns):
    '''
    This function will dorp highly correlated columns
    '''
    pass
    
def check_messy_model(data,target)
    '''
    This function will check the model before train-test-split and feature engineering
    '''
    pass