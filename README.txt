The project consists of 3 python files that should be executed in a certain order (using python3): 

1 - python3 data_generator.py : will generate the need data (in csv format) for later training/tetsing of the DNN.

2 - python3 DNN_train_test.py : creates a DeepNeuralNetwork model using keras then trains it. Once the model is trained, it is saved in .h5 format.

3 - python3 simulate_DL_controller : simulates the DeepLearning based controller and the mpc controller at the same time and plots all of it on the same graph for comparison.


REQUIREMENTS :
please install keras, numpy, matplotlib, scipy, pandas, and scikit-learn as well as casadi of course



