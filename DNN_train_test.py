
# coding: utf-8

# In[51]:



#!pip install --upgrade pip
#!pip install keras
#!pip install numpy
#!pip install tensorflow


import numpy as np
import keras
# from keras.models import Sequential
from keras.models import *
from keras.layers import *

import random

from sklearn.model_selection import train_test_split
#this one will be used for normalization and standardization
from sklearn import preprocessing

# We use pandas for easiness of use and representation of data
import pandas as pd




# Trains a DNN model, returns the trained model

# def nn_train(x_train, y_train, input_dimension, output_dimension, nb_epochs=5, nb_batch=1, nb_deeplayers=13, nb_neurons=100):
def nn_train(x_train, y_train, input_dimension, output_dimension, nb_epochs=35, nb_batch=1, nb_deeplayers=6, nb_neurons=50):
#50 nb_neurons and 6 nb_deeplayers
        # Creating the keras model
        model = Sequential()

        model.add(Dense(nb_neurons, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))

        for _ in range(nb_deeplayers - 1):
            model.add(Dense(nb_neurons, kernel_initializer='normal', activation='relu'))
            # model.add(BatchNormalization())
            # model.add(LeakyReLU(alpha=0.001))
            # model.add(Dropout(0.2))

        model.add(Dense(output_dimension, kernel_initializer='normal', activation='linear'))


        # Configuring the learning process with .compile() function
        # This is where we decide the type of loss function (aka error function)
        # (in our case : mean squared error (MSE))
        # and also the optimizer (here it's s modified stochastic gradient descent)


        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape', 'accuracy'])
        model.fit(x_train, y_train, epochs=nb_epochs, batch_size=nb_batch)

        return model


# Splits data into testing and training sets (randomly shuffles it as well)
def generate_training_testing_sets(X, y, ratio_testing=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio_testing)
    return X_train, X_test, y_train, y_test


def open_csv(filepath='simulation_data.csv'):
    data = pd.read_csv(filepath)
    print('data loaded from %s'%filepath)
    return data

def system_dynamics(x_k, u_k, d_k):
    A = np.array([[0.8511, 0],[0, 1]])
    B = np.array([[0.0035, 0, 0],[0, -5, 0]])
    E = (1e-03)*np.array([[22.217, 1.7912, 42.212],[0, 0, 0]])

    # system : x_k+1 = A*x_k + B*u_k + E*d_k
    x_k_plus = np.dot(A, x_k.reshape((2,1))) + np.dot(B, u_k.reshape((3,1))) + np.dot(E, d_k.reshape((3,1)))

    print('u_k = ')
    print(u_k)
    print('x_k_plus = ')
    print(x_k_plus)

    return x_k_plus

def csv_dump_test(X_test, y_test, filepath='test_data.csv'):
    import pandas as pd
    temp = np.concatenate((X_test, y_test), axis=1)

    # df = pd.DataFrame(temp, columns=['Tr0','Ebat0','dT','dsr','dint','Phvac','Pbat','Pgrid'])
    df = pd.DataFrame(temp)
    print(df.head())
    try:
        df.to_csv(filepath)
        print('csv test data file successfully written to %s'%filepath)
    except IOError as e:
        print(e)


if __name__ == '__main__':
    data = open_csv('Varying_disturbance_simulation_data10000lines.csv')
    print('A little overview of the loaded data: ')
    print(data.head())

    data_np = data.values
    print(data_np.shape)


    X = data_np[:, 1:18]
    y = data_np[:, 18:21]

    X_train, X_test, y_train, y_test = generate_training_testing_sets(X, y)


    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    csv_dump_test(X_test_scaled, y_test, filepath='test_data_scaled.csv')
    csv_dump_test(X_test, y_test, filepath='test_data_not_scaled.csv')


    model = nn_train(X_train_scaled, y_train, input_dimension=X.shape[1], output_dimension=y.shape[1], nb_epochs=100)

    filepath_trained_model = 'Final_model_varDist_20epochs_100000lines.h5'

    model.save(filepath_trained_model)
    print('model saved under :%s'%filepath_trained_model)

    from keras.models import *
    trained_model = load_model(filepath_trained_model)
    print('trained model loaded from :%s'%filepath_trained_model)

    X_test = X_test_scaled

    predictions = np.array([])
    for i in range(X_test.shape[0]):
        temp = np.array([])
        for j in range(X_test.shape[1]):
            temp = np.append(temp, X_test[i, j])
        temp = temp.reshape((1, X_test.shape[1]))
        predictions = np.append(predictions, trained_model.predict(temp))

    predictions = predictions.reshape((X_test.shape[0], y_test.shape[1]))

    print('X_train_scaled :')
    print(X_train_scaled[0:5, :])

    print('X_test_scaled :')
    print(X_test_scaled[0:5, :])

    print('Prediction matrix :')
    print(predictions[0:5, :])

    print('Compare it to label matrix y_test :')
    print(y_test[0:5, :])
