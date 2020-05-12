import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

import os
dir = (os.path.abspath(os.path.dirname(__file__)))

dftrain_raw = pd.read_csv(dir+'/data/train.csv')
dftest_raw = pd.read_csv(dir+'/data/test.csv')
dftrain_raw.head(10)


def preprocessing(dfdata):

    dfresult = pd.DataFrame()

    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return(dfresult)


x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape =", x_train.shape)
print("x_test.shape =", x_test.shape)


# ================

model_loaded = tf.keras.models.load_model(dir + '/tf_model')
model_loaded.evaluate(x_test, y_test)

# 预测概率
print(model_loaded.predict(x_test[0:10]))
# model(tf.constant(x_test[0:10].values,dtype = tf.float32)) #等价写法

# 预测类别
print(model_loaded.predict_classes(x_test[0:10]))
print(y_test[0:10])
