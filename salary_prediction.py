### SALARY PREDICTION MODEL - MACHINE LEARNING

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
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
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
mask=np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(df[num_cols].corr(),mask=mask, cmap="coolwarm", vmax=1, center=0,
            square=True, linewidths=.5, annot=True, annot_kws={"size": 20})
plt.show(block=True)

def find_correlation(dataframe, numeric_cols, corr_limit=0.50):
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

### Outliers

sns.boxplot(x=df["Salary"], data=df)
plt.show(block=True)

for col in df[num_cols]:
    sns.boxplot(x=df[col], data=df)
    plt.show(block=True)

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1=dataframe[col_name].quantile(q1)
    quartile3=dataframe[col_name].quantile(q3)
    interquantile_range=quartile3-quartile1
    up_limit=quartile3 + 1.5*interquantile_range
    low_limit=quartile1 - 1.5*interquantile_range
    return low_limit, up_limit

def check_outliers(dataframe, col_name):
    low_limit, up_limit=outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit=outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outliers(df, col))

for col in num_cols:
    if check_outliers(df, col):
        replace_with_thresholds(df, col)

### Missing Values

def missing_values_table(dataframe, na_name=False):
    na_columns=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss=dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missinf_df=pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss', 'ratio'])
    print(missinf_df, end="\n\n")
    if na_name:
        return na_columns

missing_values_table(df)

from sklearn.impute import KNNImputer

def impute_missing_values(dataframe):
    df1 = df.copy()
    df1.head()
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    dff=pd.get_dummies(df1[cat_cols + num_cols], drop_first=True)
    scaler= RobustScaler()
    dff=pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    imputer=KNNImputer(n_neighbors=5)
    dff=pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
    dff=pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
    return dff

df1=impute_missing_values(df)

df1.head()

print(df1.head())
print(df1.isnull().sum())

### Feature Extraction

#new_num_cols=[col for col in num_cols if col != "Salary"]
#df1[new_num_cols]=df1[new_num_cols]+0.0000000001


#df1["Hits_Success"] = (df1["Hits_Success"]/df1["AtBat"])*100
df1["NEW_RBI"] = df1["RBI"] / df1["CRBI"]
df1['NEW_Walks'] = df1['Walks'] / df1['CWalks']
df1 ['NEW_PutOuts'] = df1['PutOuts'] * df1['Years']
df1 ['NEW_Hits'] = df1['Hits'] / df1['CHits'] + df1['Hits']
df1 ["NEW_CRBI*CATBAT"] = df1['CRBI'] * df1['CAtBat']
df1["NEW_Chits"] = df1["CHits"] / df1 ["Years"]
df1 ["NEW_CHmRun"] = df1 ["CHmRun"] * df1 ["Years"]
df1 ["NEW_CRuns"] = df1["CRuns"] / df1["Years"]
df1 ["CHits"] = df1["CHits"] * df1["Years"]
df1 ["NEW_RW"] = df1["RBI"] * df1["Walks"]
df1 ["NEW_RBWALK"] = df1["RBI"] / df1["Walks"]
df1 ["NEW_CH_CB"] = df1["CHits"] / df1["CAtBat"]
df1 ["NEW_CHm_CAT"] = df1["CHmRun"] / df1["CAtBat"]
df1 ["NEW_Diff_AtBat"] = df1["AtBat"] - (df1["CAtBat"] / df["Years"])
df1 ["NEW_Diff_Hits"] = df1["Hits"] - (df1["CHits"] / df["Years"])
df1 ["NEW_Diff_HmRun"] = df1["HmRun"] - (df1["CHmRun"] / df["Years"])
df1 ["NEW_Diff_Runs"] = df1["Runs"] - (df1["CRuns"] / df["Years"])
df1 ["NEW_Diff_RBI"] = df1["RBI"] - (df1["CRBI"] / df["Years"])
df1 ["NEW_Diff_Walks"] = df1["Walks"] - (df1["CWalks"] / df["Years"])


def feature_ext(df1):
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    new_num_cols = [col for col in num_cols if col != "Salary"]
    df1[new_num_cols]=df1[new_num_cols]+0.0000000001



### One_Hot Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df1)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe=pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool':
            dataframe[col] = dataframe[col].astype(int)
    return dataframe

df1 = one_hot_encoder(df1, cat_cols, drop_first=True)

### Feature Scaling

cat_cols, num_cols, cat_but_car = grab_col_names(df1)

from scipy import stats

#def check_normality(dataframe, num_cols, plot=False):

def feature_scaling(dataframe, num_cols):
    scaler = StandardScaler()
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols= [col for col in num_cols if col not in ["Salary"]]
    dataframe[num_cols] = scaler.fit_transform(df1[num_cols])
    return dataframe

### Correlation Analysis

df1[num_cols].corr(method="spearman")

fig.ax = plt.subplots(figsize=(25,10))
sns.heatmap(df1.corr(), annot=True, linewidths=.5, ax=ax)
plt.show(block=True)

corr_matrix = df1.corr()
print(corr_matrix)

high_corr_threshold=0.95
high_corr_features=set()
for i in range(len(corr_matrix.columns)):
    for j in range (i):
        if abs(corr_matrix.iloc[i,j])>high_corr_threshold:
            colname=corr_matrix.columns[i]
            high_corr_features.add(colname)

print(high_corr_features)




















