import joblib
MODEL_PATH = 'models/logistic/logistic_model.pkl'
SCALER_PATH = 'models/logistic/logistic_scaler.pkl'

def load_logistic_model ():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler

## models --> JOBLIB / Pickle

'''
import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib   # to conver ML models to binary format

df = pd.read_csv('data/Cardiovascular_Disease.csv')

# cleaning
age_data = df['age']//365
df['age'] = age_data


data_filter = df[(df['height'].between(125, 200)) & 
                (df['weight'].between(40, 120)) & 
                (df['ap_hi'].between(100, 200)) & 
                (df['ap_lo'].between(50, 90))
    ]
df = data_filter

MODEL_PATH = 'models/logistic/logistic_model.pkl'
SCALER_PATH = 'models/logistic/logistic_scaler.pkl'


## Model
def cardio():
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'

    X = df[features]
    Y = df[target]

    ## train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver = 'lbfgs',
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_scale, Y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    
    return scaler, model

'''

