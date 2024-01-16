
#############################################################################
#############################################################################

Notes = '''




'''

#############################################################################
#############################################################################

import numpy as np 
import pandas as pd

from random import shuffle, choice
import random,time,os,io,requests,datetime
import json,hmac,hashlib,base64,pickle,math 
from collections import defaultdict as defd
from heapq import nlargest
from copy import deepcopy

from scipy import signal
from scipy.stats import entropy 
from scipy.constants import convert_temperature
from scipy.interpolate import interp1d 
#from scipy.ndimage import uniform_filter1d
#from scipy.spatial.transform import Rotation

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
from sklearn.ensemble       import ExtraTreesClassifier       as ETC
from sklearn.ensemble       import ExtraTreesRegressor        as ETR
from sklearn.ensemble       import BaggingClassifier          as BGC 
from sklearn.ensemble       import GradientBoostingClassifier as GBC 
from sklearn.ensemble       import GradientBoostingRegressor  as GBR 
from sklearn.neural_network import MLPRegressor               as MLP 
from sklearn.linear_model   import LinearRegression           as OLS
from sklearn.preprocessing  import LabelBinarizer             as LBZ 
from sklearn.decomposition  import PCA                        as PCA 

from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline 
from sklearn.utils import check_array
from sklearn.preprocessing import * 
from sklearn.metrics import *

from sympy.solvers import solve
from sympy import Symbol,Eq,sympify
from sympy import log,ln,exp          #,Wild,Mul,Add,sin,cos,tan

import statsmodels.formula.api as smf
from pygam import LinearGAM, GAM #, s, f, l, te

# XGBoost Python Wrapper for SKLearn: 
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
import xgboost as xgb 
from xgboost import XGBRegressor  as XGR
from xgboost import XGBClassifier as XGC 


#############################################################################
#############################################################################


# Set print options: suppress scientific notation and set precision
np.set_printoptions(suppress=True, precision=8)
# Set Numpy error conditions: 
old_set = np.seterr(divide = 'ignore',invalid='ignore') 


#############################################################################
#############################################################################

### FILE I/O OPERATIONS: 

def get_files(folder_path):
    """
    List all non-hidden files in a given folder.

    Parameters:
    folder_path (str): Path to the folder.

    Returns:
    list: A list of filenames in the folder.
    """
    if folder_path[-1] != '/':
        folder_path += '/'
    return [file for file in os.listdir(folder_path) if not file.startswith('.')]

def get_path(string):
    """
    Extract the path from a string representing a file or folder.

    Parameters:
    string (str): The input string.

    Returns:
    str: The extracted path.
    """
    if '/' not in string:
        return '' if '.' in string else string
    parts = string.split('/')
    if '.' not in parts[-1]:
        return string if string.endswith('/') else string + '/'
    return '/'.join(parts[:-1]) + '/'

def ensure_path(path):
    """
    Create a path if it doesn't already exist.

    Parameters:
    path (str): The path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(filename):
    """
    Read a JSON file and return its content.

    Parameters:
    filename (str): The name of the JSON file.

    Returns:
    dict: The content of the JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)

def write_json(filename, obj, pretty=True):
    """
    Write an object to a JSON file.

    Parameters:
    filename (str): The name of the JSON file.
    obj (object): The Python object to write.
    pretty (bool): Whether to write the JSON in a pretty format.
    """
    path = get_path(filename)
    if path:
        ensure_path(path)
    with open(filename, 'w') as file:
        if pretty:
            json.dump(obj, file, sort_keys=True, indent=2, separators=(',', ': '))
        else:
            json.dump(obj, file, sort_keys=True)

def export_model(filename, model_object):
    """
    Export a fitted model to a file.

    Parameters:
    filename (str): The name of the file to save the model to.
    model_object (object): The model object to save.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model_object, file)

def import_model(filename):
    """
    Import a fitted model from a file.

    Parameters:
    filename (str): The name of the file to load the model from.

    Returns:
    object: The loaded model object.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def read_github_csv(csv_url):
    """
    Read a CSV file from a GitHub URL.

    Parameters:
    csv_url (str): The URL of the CSV file.

    Returns:
    DataFrame: The content of the CSV file as a pandas DataFrame.
    """
    response = requests.get(csv_url)
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

def try_get_default(dictionary, key, default=0.0):
    """
    Try to get a value from a dictionary; return a default value if the key is not found.

    Parameters:
    dictionary (dict): The dictionary to search.
    key (object): The key to look for.
    default (object): The default value to return if the key is not found.

    Returns:
    object: The value associated with the key or the default value.
    """
    return dictionary.get(key, default)


#############################################################################
#############################################################################

def calc_rmse(actual, predictions):
    """
    Calculate the Root Mean Square Error (RMSE) between actual and predicted values.

    Parameters:
    actual (numpy array): The actual values.
    predictions (numpy array): The predicted values.

    Returns:
    float: The RMSE value.
    """
    # Calculate the square of differences
    differences = np.subtract(actual, predictions)
    squared_differences = np.square(differences)

    # Calculate the mean of squared differences
    mean_squared_differences = np.mean(squared_differences)

    # Calculate the square root of the mean squared differences (RMSE)
    rmse = np.sqrt(mean_squared_differences)
    return rmse


def calc_r2(actual, predictions):
    """
    Calculate the R-squared value between actual and predicted values.

    Parameters:
    actual (numpy array): The actual values.
    predictions (numpy array): The predicted values.

    Returns:
    float: The R-squared value.
    """
    # Calculate the mean of actual values
    mean_actual = np.mean(actual)

    # Calculate the total sum of squares (SST)
    sst = np.sum(np.square(np.subtract(actual, mean_actual)))

    # Calculate the residual sum of squares (SSR)
    ssr = np.sum(np.square(np.subtract(actual, predictions)))

    # Calculate R-squared
    r_squared = 1 - (ssr / sst)
    return r_squared


def robust_mean(distribution, center=0.7):
    """
    Calculate the mean of a distribution, excluding outliers.

    Parameters:
    distribution (array-like): The input distribution from which the mean is calculated.
    center (float): The central percentage of the distribution to consider. 
                    Default is 0.7, meaning the middle 70% is considered.

    Returns:
    float: The mean of the distribution after excluding outliers.
    """
    if not isinstance(distribution, np.ndarray):
        distribution = np.array(distribution)

    if distribution.size == 0 or not np.issubdtype(distribution.dtype, np.number):
        return np.nan

    margin = 100.0 * (1 - center) / 2.0
    min_val = np.percentile(distribution, margin)
    max_val = np.percentile(distribution, 100.0 - margin)

    filtered_dist = distribution[(distribution >= min_val) & (distribution <= max_val)]

    return np.mean(filtered_dist) if filtered_dist.size > 0 else np.nan


#############################################################################
#############################################################################


# Interpolation Model: 
class InterpModel:
    def __init__(self):  
        pass
    
    def fit(self,x_train,y_train): 
        x = np.array(x_train)
        y = np.array(y_train)

        if x.shape == (len(x),1): x = x.reshape([len(x),]) 
        if y.shape == (len(y),1): y = y.reshape([len(y),]) 
              
        self.lin_predict = interp1d(x,y)
        #self.cub_predict = interp1d(x,y,kind='cubic') 
        self.xmin = x.min()
        self.xmax = x.max() 
        
    def predict(self,x_test,kind='linear'):
        x = np.array(x_test) 
        if x.shape == (len(x),1): x = x.reshape([len(x),]) 
        x2 = np.clip(x,self.xmin,self.xmax)
        if kind=='linear': preds = self.lin_predict(x2)
        #if kind=='cubic':  preds = self.cub_predict(x2) 
        return preds 


class QuantileModel:
    def __init__(self):
        self.v2q_model = InterpModel()
        self.q2v_model = InterpModel()
    
    def fit(self,arr): 
        ys = np.sort(np.array(arr))
        xs = np.linspace(0.0,1.0,len(ys))
        self.v2q_model.fit(ys,xs)
        self.q2v_model.fit(xs,ys) 
        
    def v2q(self,value):
        return self.v2q_model.predict(value)
    
    def q2v(self,quant):
        return self.q2v_model.predict(quant) 
    
    def predict(self,value):
        return self.v2q(value)
        
#############################################################################
#############################################################################
        
def VecDist(v1,v2):
    return np.abs((v1-v2)**2).sum(1)**0.5   

#############################################################################
#############################################################################

#----------------------------------------------------

# Simulation Variables:
N_DIMS  = 6
N_USERS = 1000
N_ITEMS = 500 
MIN_REV = 10
MAX_REV = 60

#----------------------------------------------------

UserPrefMatrix = np.random.uniform(-9.99,+9.99,(N_USERS,N_DIMS))
UserPrefMatrix = np.around(UserPrefMatrix,7)

UserDF = pd.DataFrame(UserPrefMatrix) 
DimCols = ["D"+str(a) for a in range(1,N_DIMS+1)]
UserDF.columns = DimCols

UserIDs = np.arange(1,N_USERS+1) 
UserIDs = ["U"+(('00000'+str(a))[-4:]) for a in UserIDs]
UserDF['UID'] = UserIDs
old_cols = list(UserDF.columns)
new_cols = [old_cols[-1]]+old_cols[:-1]
UserDF = UserDF[new_cols] 

user2vec = {} 
for r in UserDF.values:
    uid = r[0]
    vec = r[1:] 
    user2vec[uid] = vec

#----------------------------------------------------

ItemAttrMatrix = np.random.uniform(-9.99,+9.99,(N_ITEMS,N_DIMS))
ItemAttrMatrix = np.around(ItemAttrMatrix,7)

ItemDF = pd.DataFrame(ItemAttrMatrix) 
DimCols = ["D"+str(a) for a in range(1,N_DIMS+1)]
ItemDF.columns = DimCols

ItemIDs = np.arange(1,N_ITEMS+1) 
ItemIDs = ["I"+(('00000'+str(a))[-4:]) for a in ItemIDs]
ItemDF['IID'] = ItemIDs
old_cols = list(ItemDF.columns)
new_cols = [old_cols[-1]]+old_cols[:-1]
ItemDF = ItemDF[new_cols] 

item2vec = {} 
for r in ItemDF.values:
    iid = r[0]
    vec = r[1:] 
    item2vec[iid] = vec

#----------------------------------------------------

rows = []
for uid in UserIDs:
    uvec = user2vec[uid]
    uvec = np.array([uvec])
    
    n_revs = np.random.randint(MIN_REV,MAX_REV)
    items = [choice(ItemIDs) for _ in range(n_revs)] 
    items = sorted(set(items)) 
    for iid in items:
        ivec = item2vec[iid]
        ivec = np.array([ivec])
        
        dist = VecDist(uvec,ivec)[0] 
        rows.append([uid,iid,dist])
    
RevDF = pd.DataFrame(rows) 
RevDF.columns = ['UID','IID','DIST'] 

#----------------------------------------------------

qm = QuantileModel()
qm.fit(RevDF['DIST'].values) 
dper = qm.v2q(RevDF['DIST'].values) 
RevDF['DPER'] = dper
RevDF['AFFIN'] = (1-RevDF['DPER'])**2
RevDF['RATING'] = np.around((RevDF['AFFIN']*9)+1).astype(int) 

#----------------------------------------------------


rand_arr = np.random.uniform(0,1,len(RevDF))*0.7
TF = rand_arr < RevDF['AFFIN'].values 
RevDF['SALE'] = TF.astype(int) 

#----------------------------------------------------






































































































#############################################################################
#############################################################################




