### SALARY PREDICTION MODEL - MACHINE LEARNING

## Veri Seti

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından
# yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

## Değişkenler

# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import anderson
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.exceptions import ConvergenceWarning

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 170)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.SettingWithCopyWarning)

df=pd.read_csv('hitters.csv')
# TARGET="Salary"

### Overall Picture

def check_def(dataframe, head=5):
    print("## Shape ##")
    print(dataframe.shape)

    print("## Types ##")
    print(dataframe.dtypes)

    print("## Head ##")
    print(dataframe.head(head))

    print("## NA ##")
    print(dataframe.isnull().sum())

    print("## Quantiles ##")
    print(dataframe.describe([0.25, 0.5, 0.75]).T)

check_def(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    #cat_cols, cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=="O"]
    num_but_cat= [col for col in dataframe.columns if dataframe[col].nunique() <cat_th and
                  dataframe[col].dtypes != "O"]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique() >car_th and
                 dataframe[col].dtypes == "O"]
    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

### Analysis of Categorical Variables

def cat_summary(dataframe, col_name, plot=False):
    print("###")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("###")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

### Analysis of Numerical Variables

def num_summary(dataframe, numerical_col, plot=False):
     quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

     print(dataframe[numerical_col].describe(quantiles).T)

     if plot:
         dataframe[numerical_col].hist(bins=20)
         plt.xlabel(numerical_col)
         plt.title(numerical_col)
         plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


### Analysis of Target Variable

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"Salary", col)

### Analysis of Correlation

print(df[num_cols].corr(method="spearman"))

fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(df[num_cols].corr(), annot=True, linewidths=.5, ax=ax)
plt.show(block=True)

# Correlation with the final state of the variables
plt.figure(figsize=(45,45))
corr=df[num_cols].corr()
sns.set_context("talk", font_scale=1.5)
mask=np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(df[num_cols].corr(),mask=mask, cmap="coolwarm", vmax=1, center=0,
            square=True, linewidths=.5, annot=True, annot_kws={"size": 20})
plt.show(block=True)

def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlation=[]
    low_correlation=[]
    for col in numeric_cols:
        if col == "Salary":
            pass
        else:
            correlation=dataframe[[col, "Salary"]].corr().loc[col, "Salary"]
            print(col,correlation)
            if abs(correlation) > corr_limit:
                high_correlation.append(col + ": " + str(correlation))
            else:
                low_correlation.append(col + ": " + str(correlation))
    return low_correlation, high_correlation

low_corrs, high_corrs = find_correlation(df, num_cols)


















