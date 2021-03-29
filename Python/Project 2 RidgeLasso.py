# coding: utf-8
"""
@author: Jacob Hobbs
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1, l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Returns the neural net regModel with l1 or l2 regularization based on given number of predictors,
# a model type, and regularizer.
#
# @param num_predictors - The number of predictor variables in the regModel.
# @param model_type - The type of neural net (Perceptron, NN3L, or NN4L).
# @param regularizer - The type of regularizer for the model (l1 (lasso) or l2 (ridge)).
def makeModel(num_predictors, model_type, regularizer):
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False, name = "Adam")
    
    model = Sequential()
    if regularizer == 'l1':
        if (model_type == "NeuralNet3L") or (model_type == "NeuralNetXL"):
            model.add(Dense(num_predictors, input_dim = num_predictors, kernel_initializer = 'normal',
                            activation = 'relu', kernel_regularizer = l1(0.01)))
        if model_type == "NeuralNetXL":
            model.add(Dense(num_predictors, input_dim = num_predictors, kernel_initializer = 'normal',
                            activation = 'relu', kernel_regularizer = l1(0.01)))    
        model.add(Dense(1, kernel_initializer = 'normal', kernel_regularizer = l1(0.01)))
        model.compile(loss = 'mean_squared_error', optimizer = opt)
    elif regularizer == 'l2':
        if (model_type == "NeuralNet3L") or (model_type == "NeuralNetXL"):
            model.add(Dense(num_predictors, input_dim = num_predictors, kernel_initializer = 'normal',
                            activation = 'relu', kernel_regularizer = l1(0.01)))
        if model_type == "NeuralNetXL":
            model.add(Dense(num_predictors, input_dim = num_predictors, kernel_initializer = 'normal',
                            activation = 'relu', kernel_regularizer = l1(0.01)))    
        model.add(Dense(1, kernel_initializer = 'normal', kernel_regularizer = l1(0.01)))
        model.compile(loss = 'mean_squared_error', optimizer = opt)
    return model

# Returns various measures of model fit for regularized neural nets. This method calls
# makeModel using the X and y dataset parameters.
#
# @param X - The matrix of predictors.
# @param y - The response vector.
# @param model_type - The type of neural net (perceptron, NN3L, or NN4L).
# @param measure - The measure of model fit.
# @param regularizer - The type of regularizer for the model (l1 (lasso) or l2 (ridge)).
def measureModel(X, y, model_type, measure, regularizer):
  
    if (measure == "rsquared") or (measure == "rsquared_adj"):
        
        model = makeModel(X.shape[1], model_type, regularizer)
        model.fit(X, y, epochs = 100, batch_size = 10, verbose = 0)
        y_hat = np.reshape(model.predict(X), len(y))
        sst = np.sum(np.square(y - np.mean(y)))
        sse = np.sum(np.square(y - y_hat))
        rsquared = (1 - sse/sst)
        
        if measure == "rsquared":
            return rsquared
        elif measure == "rsquared_adj":
            return (1 - ((len(y) - 1)/(len(y) - model.count_params())) * (1 - rsquared))
        
    elif measure == "rsquared_cv":
        estimator = KerasRegressor(build_fn = makeModel, num_predictors = X.shape[1], model_type = model_type, regularizer = regularizer,
                                   epochs = 100, batch_size = 5, verbose = 0)
        kfold = KFold(n_splits = 10)
        sst = np.sum(np.square(y - np.mean(y)))
        sse = np.mean(np.abs(cross_val_score(estimator, X, y, cv = kfold))) * len(y)
        return (1 - sse/sst)

# Prints the results of l1 and l2 regularization in the form of various measures
# of model fit.
#
# @param dataframe - The dataframe containing predictors and response variable.
# @param response - The name of the response variable.
def summariesPrinter(dataframe, response):
    print(dataframe.name)
    predictors = dataframe.columns.tolist()
    predictors.remove(response)
    X = dataframe[predictors].values
    y = dataframe[response].values
    
    for model_type in ['Perceptron', 'NeuralNet3L', 'NeuralNetXL']:
        print('----------' + model_type + '----------', '\n', '\n')
        for measure in ['rsquared', 'rsquared_adj', 'rsquared_cv']:
            print('----------' + measure + '----------', '\n')
            print('LASSO REGRESSION (L1 REGULARIZATION)')
            print(measureModel(X, y, model_type, measure, 'l1'))
            print('\n', 'RIDGE REGRESSION (L2 REGULARIZATION)')
            print(measureModel(X, y, model_type, measure, 'l2'))
            print('\n')
        print('\n', '\n')
    print('\n', '\n', '\n')

autompg = pd.read_csv('auto-mpg.data', delim_whitespace = True, 
                      names = ['mpg', 'cylinders','displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'], 
                      index_col = 'car_name').replace(to_replace = '?', value = np.NaN)
autompg['horsepower'] = autompg['horsepower'].astype(float)
autompg = autompg.fillna(autompg.mean())
autompg.name = 'autompg'
summariesPrinter(autompg, 'mpg')

concrete = pd.read_csv('slump_test.data', index_col = 'No')
concrete.name = 'concrete'
summariesPrinter(concrete, 'Compressive_Strength')

aquatic = pd.read_csv('qsar_aquatic_toxicity.csv')
aquatic.name = 'aquatic'
summariesPrinter(aquatic, 'LC50')

wine = pd.read_csv('wine.csv')
wine.name = 'wine'
summariesPrinter(wine, 'quality')

concrete_strength = pd.read_csv('Concrete_Data.csv')
concrete_strength.name = 'concrete_strength'
summariesPrinter(concrete_strength, 'Concrete_compressive_strength')

