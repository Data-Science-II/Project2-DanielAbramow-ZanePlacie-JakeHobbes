# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:58:09 2021

@author: zgp21
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# This method returns the structure of a neural net based on the given number of
# predictors and model type given. If the model type is 'Perceptron', the method
# returns a perceptron (a neural net with no hidden layer). If the model type is
# 'NeuralNet3L', the method returns a net with one fully connected hidden layer.
# If the model type is 'NeuralNetXL', the method returns a net with two fully 
# connected hidden layers. All layers use the relu activation function.
#
# @param num_predictors - the number of predictor variables in the model; this
# sets the number of neurons in each fully connected layer
# @param model_type - the type of neural net
def makeModel(num_predictors, model_type):
    
    # Change learning rate here
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False, name = "Adam")
    
    model = Sequential()
    if (model_type == "NeuralNet3L") or (model_type == "NeuralNetXL"):
        
        # Change activation function or model architecture here
        model.add(Dense(2 * num_predictors, input_dim = num_predictors, kernel_initializer = 'normal', activation = 'relu'))
        
    if model_type == "NeuralNetXL":
        
        # Change activation function or model architecture here
        model.add(Dense(num_predictors, input_dim = num_predictors, kernel_initializer = 'normal', activation = 'relu'))    
        
    model.add(Dense(1, kernel_initializer = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer = opt)
    return model

# This method can return various measures of model fit for various types of 
# neural nets. First, the method calls makeModel to fit a net based on the model
# type specified and the number of columns in the predictor matrix X. Then, it
# calculates either r-squared, adjusted r-squared, or cross-validated r-squared
# for that net and the given data depending on the measure specified.
#
# @param X - the matrix of predictor values for each observation
# @param y - the response vector
# @param model_type - the type of neural net
# @param measure - the measure of model fit returned
def measureModel(X, y, model_type, measure):
  
    if (measure == "rsquared") or (measure == "rsquared_adj"):
        
        model = makeModel(X.shape[1], model_type)
        
        # Change number of epochs or batch size here
        model.fit(X, y, epochs = 100, batch_size = 5, verbose = 0)
        
        y_hat = np.reshape(model.predict(X), len(y))
        sst = np.sum(np.square(y - np.mean(y)))
        sse = np.sum(np.square(y - y_hat))
        rsquared = (1 - sse/sst)
        
        if measure == "rsquared":
            return rsquared
        elif measure == "rsquared_adj":
            return (1 - ((len(y) - 1)/(len(y) - model.count_params())) * (1 - rsquared))
        
    elif measure == "rsquared_cv":
        estimator = KerasRegressor(build_fn = makeModel, num_predictors = X.shape[1], model_type = model_type, epochs = 100, batch_size = 5, verbose = 0)
        kfold = KFold(n_splits = 10)
        sst = np.sum(np.square(y - np.mean(y)))
        sse = np.mean(np.abs(cross_val_score(estimator, X, y, cv = kfold))) * len(y)
        return (1 - sse/sst)

# This method returns the model with one fewer predictor than the given model 
# that results in the largest improvement in the given measure of model fit.
# If there is no way to improve model fit by removing a predictor, this method
# returns the model with no predictors removed. Specifically, the method returns
# the list of predictors used in the new model and the given measure for the 
# new model.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param predictors - the list of variables in the dataframe used as predictors
# in the given model
# @param response - the name of the response variable
# @param model_type - the type of neural net
# @param measure - the measure of model fit to optimize
def backwardElim(dataframe, predictors, response, model_type, measure):
 
    final_predictors = predictors
    base_measure = 0
    
    # A predictor can only be removed if there is at least one to remove
    if len(predictors) > 0:
        
        X = dataframe[predictors].values
        y = dataframe[response].values
        base_measure = measureModel(X, y, model_type, measure) # make base model
            
        # For each predictor, make a new set of predictors including each
        # current predictor except that one, then fit a new model and 
        # assess its measure.
        for predictor in predictors:
            new_predictors = []
            for p in predictors:
                if p != predictor:
                    new_predictors.append(p)
                
            X = dataframe[new_predictors].values        
            new_measure = 0 # A model with no predictors has an r squared of 0
            if len(new_predictors) > 0:
                new_measure = measureModel(X, y, model_type, measure)
                
            # Use this model if its r squared exceeds the previous best model
            if new_measure > base_measure:
                base_measure = new_measure
                final_predictors = new_predictors
            
    return final_predictors, base_measure

# This method returns the model with one more predictor than the given model 
# that results in the largest improvement in the given measure of model fit.
# If there is no way to improve model fit by adding a predictor, this method
# returns the model with no predictors added. Specifically, the method returns
# the list of predictors used in the new model and the given measure for the 
# new model.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param predictors - the list of variables in the dataframe used as predictors
# in the given model
# @param response - the name of the response variable
# @param model_type - the type of neural net
# @param measure - the measure of model fit to optimize
def forwardSel(dataframe, predictors, response, model_type, measure):
    
    X = dataframe[predictors].values
    y = dataframe[response].values
    base_measure = measureModel(X, y, model_type, measure) # make base model
    
    # Get list of variables which are not currently used as predictors
    unused_predictors = dataframe.columns.tolist()
    unused_predictors.remove(response)
    for predictor in predictors:
        unused_predictors.remove(predictor)
        
    original_predictors = []
    for predictor in predictors:
        original_predictors.append(predictor)

    final_predictors = predictors
        
    # For each unused predictor, add this predictor to the list of predictors
    # and fit another model
    for unused_predictor in unused_predictors:
            
        predictors = []
        for original_predictor in original_predictors:
            predictors.append(original_predictor)   
        predictors.append(unused_predictor)
            
        X = dataframe[predictors].values
        new_measure = measureModel(X, y, model_type, measure)
        
        # Replace the previous best model with the current model if it has 
        # a higher r squared.
        if new_measure > base_measure:
            base_measure = new_measure
            final_predictors = predictors

    return final_predictors, base_measure

# This method returns the model with either one less or one more predictor than the given model 
# that results in the largest improvement in the given measure of model fit.
# If there is no way to improve model fit by removing or adding a predictor, this method
# returns the model with no predictors removed or added. Specifically, the method returns
# the list of predictors used in the new model and the given measure for the 
# new model.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param predictors - the list of variables in the dataframe used as predictors
# in the given model
# @param response - the name of the response variable
# @param model_type - the type of neural net
# @param measure - the measure of model fit to optimize
def stepRegression(dataframe, predictors, response, model_type, measure):
    
    X = dataframe[predictors].values
    y = dataframe[response].values
    base_measure = measureModel(X, y, model_type, measure)
    
    final_predictors = predictors # The predictors to return if no other model is better
    
    # Sets the backward elimination model to be returned if that model has a 
    # better measure of model fit than the original model
    backward_predictors, backward_measure = backwardElim(dataframe, predictors, response, model_type, measure)
    if backward_measure > base_measure:
        final_predictors = backward_predictors
        base_measure = backward_measure
   
    # Sets the forward selection model to be returned if that model has a 
    # better measure of model fit than the original model or backward elimination
    # model
    forward_predictors, forward_measure = forwardSel(dataframe, predictors, response, model_type, measure)
    if forward_measure > base_measure:
        final_predictors = forward_predictors
        base_measure = forward_measure

    return final_predictors, base_measure

# This method constructs a model for the given response variable including all
# of the possible predictors in the dataframe, then repeatedly calls backwardElim
# until the specified measure of model fit cannot be improved by removing
# another predictor. Specifically, the method returns the list of predictors used 
# in the best model and the given measure for this model.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param response - the name of the response variable
# @param model_type - the type of neural net
# @param measure - the measure of model fit to optimize
def backwardElimAll(dataframe, response, model_type, measure):
    
    # Get all possible predictors
    predictors = dataframe.columns.tolist()
    predictors.remove(response)
    
    X = dataframe[predictors].values
    y = dataframe[response].values
    base_measure = measureModel(X, y, model_type, measure) # make base model
    
    while True:
       
        # Gets best backwards elimination model
        new_predictors, new_measure = backwardElim(dataframe, predictors, response, model_type, measure)
        
        # Either replaces the input model with the backwards elimination model 
        # or (if there is no improvement in model fit) ends the method
        if new_measure > base_measure:
            predictors = new_predictors
            base_measure = new_measure
        else:
            break

    return predictors, base_measure

# This method constructs the null model for the given response variable, then 
# repeatedly calls forwardSel until the specified measure of model fit cannot 
# be improved by adding another predictor. Specifically, the method returns the 
# list of predictors used in the best model and the given measure for this model.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param response - the name of the response variable
# @param model_type - the type of neural net
# @param measure - the measure of model fit to optimize
def forwardSelAll(dataframe, response, model_type, measure): 
    
    predictors = []
    base_measure = 0 # r-squared of null model is 0
    
    while True:
        
        # Get best forward selection model
        new_predictors, new_measure = forwardSel(dataframe, predictors, response, model_type, measure)
        
        # Either replaces the input model with the forward selection model or 
        # (if there is no improvement in model fit) ends the method
        if new_measure > base_measure:
            predictors = new_predictors
            base_measure = new_measure
        else:
            break

    return predictors, base_measure

# This method constructs the null model for the given response variable, then 
# repeatedly calls forwardSel and backwardElim until the specified measure of 
# model fit cannot be improved by adding or removing another predictor. 
# Specifically, the method returns the list of predictors used in the best model 
# and the given measure for this model.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param response - the name of the response variable
# @param model_type - the type of neural net
# @param measure - the measure of model fit to optimize
def stepRegressionAll(dataframe, response, model_type, measure):
    
    predictors = []
    base_measure = 0 # r-squared of null model is 0
    
    while True:
        
        # Get best stepwise regression model
        new_predictors, new_measure = stepRegression(dataframe, predictors, response, model_type, measure)
        
        # Either replaces the input model with the forward selection model or 
        # (if there is no improvement in model fit) ends the method
        if new_measure > base_measure:
            predictors = new_predictors
            base_measure = new_measure
        else:
            break

    return predictors, base_measure

# This method constructs the null model for the response variable, then
# repeatedly adds the variable to the model which results in the greatest
# increase in r-squared, until the model includes all possible predictors. The 
# method records the r-squared, adjusted r-squared, and cross-validated r-squared
# for each model and plots them.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param response - the name of the response variable
# @param model_type - the type of neural net
def backwardElimPlot(dataframe, response, model_type):
    
    number_of_predictors = []
    rsquared = []
    rsquared_adj = []
    rsquared_cv = []
    
    # Get all possible predictors
    predictors = dataframe.columns.tolist()
    predictors.remove(response)
    y = dataframe[response].values
    
    while len(predictors) > 0:
        
        X = dataframe[predictors].values
        
        # Records the various measures of model fit
        number_of_predictors.append(len(predictors))
        rsquared.append(measureModel(X, y, model_type, 'rsquared'))
        rsquared_adj.append(measureModel(X, y, model_type, 'rsquared_adj'))
        rsquared_cv.append(measureModel(X, y, model_type, 'rsquared_cv'))

        # Ends the method if there is only one predictor left
        if len(predictors) == 1:
            break

        base_rsquared = -100
        
        # Of the models constructed from removing one predictor from the previous
        # model, selects the model with the highest r-squared    
        original_predictors = predictors
        for original_predictor in original_predictors:
            new_predictors = []
            for p in original_predictors:
                if p != original_predictor:
                    new_predictors.append(p)
            X = dataframe[new_predictors].values
            new_rsquared = measureModel(X, y, model_type, 'rsquared')
            if new_rsquared > base_rsquared:
                base_rsquared = new_rsquared
                predictors = new_predictors
    
    # Plots the measures of model fit against the number of predictors
    measures = pd.DataFrame(data = {'rsquared': rsquared, 'rsquared_adj': rsquared_adj, 'rsquared_cv': rsquared_cv}, 
                                     index = number_of_predictors)
    measures.plot()
    
# This method constructs a model for the response variable including all possible predictors, then
# repeatedly removes the variable from the model which results in the least decrease
# in r-squared, until the model includes only one predictor. The method records the r-squared,
# adjusted r-squared, and cross-validated r-squared for each model and 
# plots them.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param response - the name of the response variable
# @param model_type - the type of neural net
def forwardSelPlot(dataframe, response, model_type):
    
    number_of_predictors = []
    rsquared = []
    rsquared_adj = []
    rsquared_cv = []
    
    predictors = []
    y = dataframe[response].values
    
    while True:
        
        # If the model is no longer the null model, records the various measures
        # of model fit
        if len(predictors) > 0:
            
            X = dataframe[predictors].values
            number_of_predictors.append(len(predictors))
            rsquared.append(measureModel(X, y, model_type, 'rsquared'))
            rsquared_adj.append(measureModel(X, y, model_type, 'rsquared_adj'))
            rsquared_cv.append(measureModel(X, y, model_type, 'rsquared_cv'))
 
        # Get list of variables which are not currently used as predictors
        unused_predictors = dataframe.columns.tolist()
        unused_predictors.remove(response)
        for predictor in predictors:
            unused_predictors.remove(predictor)
    
        # Ends the method if there are no more predictors to add
        if len(unused_predictors) == 0:
            break
        
        base_rsquared = -100
        original_predictors = predictors.copy()
        
        for unused_predictor in unused_predictors:
            new_predictors = original_predictors.copy()
            new_predictors.append(unused_predictor)
            X = dataframe[new_predictors].values
            
            new_rsquared = measureModel(X, y, model_type, 'rsquared')

            if new_rsquared > base_rsquared:
                base_rsquared = new_rsquared
                predictors = new_predictors
                
    # Plots the measures of model fit against the number of predictors
    measures = pd.DataFrame(data = {'rsquared': rsquared, 'rsquared_adj': rsquared_adj, 'rsquared_cv': rsquared_cv}, 
                                     index = number_of_predictors)
    measures.plot()
    
# For each type of neural net, this method prints the results of repeated backwards 
# elimination, forward selection, and stepwise regression for the specified dataframe and response
# variable, using the various measures of model fit, and prints plots of measures 
# of model fit against the number of predictors used.
#
# @param dataframe - the dataframe containing the values of predictor and 
# response variables
# @param response - the name of the response variable
def summariesPrinter(dataframe, response):
    print(dataframe.name)
    for model_type in ['Perceptron', 'NeuralNet3L', 'NeuralNetXL']:
        print('----------' + model_type + '----------', '\n', '\n')
        for measure in ['rsquared', 'rsquared_adj', 'rsquared_cv']:
            print('----------' + measure + '----------', '\n')
            print('BACKWARD ELIMINATION')
            print(backwardElimAll(dataframe, response, model_type, measure))
            print('\n', 'FORWARD SELECTION')
            print(forwardSelAll(dataframe, response, model_type, measure))
            print('\n', 'STEPWISE REGRESSION')
            print(stepRegressionAll(dataframe, response, model_type, measure))
            print('\n')
        print('\n', '\n')
        backwardElimPlot(dataframe, response, model_type)
        plt.title(dataframe.name + ", " + model_type + ", Backward Elimination")
        forwardSelPlot(dataframe, response, model_type)
        plt.title(dataframe.name + ", " + model_type + ", Forward Selection")
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