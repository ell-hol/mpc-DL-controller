import numpy as np
import keras
# from keras.models import Sequential
from keras.models import *
from keras.layers import *

import random

from sklearn.model_selection import train_test_split
#this one will be used for normalization and standardization
from sklearn import preprocessing

import scipy.io as sio

# We use pandas for easiness of use and representation of data
import pandas as pd

from casadi import *

# Simulates the system's dynamics using the system's equation
# Inputs : x, u and disturbance at step k
# output : x at step k+1
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

# Calculates the mixed constraints of the whole simulation
# Inputs : the input matrix generated from a controller (MPC or DL) simulation and the disturbance vectors
# Output : the computed mixed constraints vector g
def generate_mixed_constraints(mpc_u, d_full):
    D = np.array([[-1, 1, 1], [1, 1, 1]])
    G = np.array([[0, 0.5, 0], [0, 0.5, 0]])

    mpc_g = np.array([])

    for i in range(mpc_u.shape[0]):
        temp = np.dot(D, mpc_u[i,:]) + np.dot(G, d_full[i,:])
        mpc_g = np.append(mpc_g, temp)

    mpc_g = mpc_g.reshape((mpc_u.shape[0], 2))
    return mpc_g

# Simulates a trained DNN model as fully functionning DL based controller
# Inputs : The DNN already trained model, the disturbances and the number of simulation steps
# Outputs : Simulated input u and simulated state x
def simulate_DLcontroller(trained_model, d_full, X_test, S=100):

    X_test_scaled = preprocessing.scale(X_test[:,0:2])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    print(scaler.fit(X_test[:,0:2]))

    # X_init_test_scaled = X_test_scaled[0,0:2]
    # X_init_test = X_test[0,0:2]

    X_init_test_scaled = scaler.transform(np.array([[20, 50000]]))
    X_init_test = np.array([[20, 50000]])

    d_full_scaled = preprocessing.scale(d_full)


    Sim_u = np.array([])
    Sim_x = np.array([])

    Sim_x_scaled = np.array(X_init_test_scaled)


    Sim_x = np.append(Sim_x, X_init_test)

    Sim_x = Sim_x.reshape((1,2))
    Sim_x_scaled = Sim_x_scaled.reshape((1,2))


    for i in range(S):

        temp = np.array([])
        temp = np.append(temp, (Sim_x_scaled[i, :]).reshape((1,2)))
        temp = np.append(temp, (d_full_scaled[i:i+5, :]).reshape((1,15)))

        temp = temp.reshape((1, 17))
        prediction_u = trained_model.predict(temp)
        prediction_u = prediction_u.reshape((1,3))

        x_plus = system_dynamics(Sim_x[i,:], prediction_u, d_full[i,:])
        Sim_x = np.append(Sim_x, x_plus)
        Sim_x = Sim_x.reshape((i+2, 2))

        print('x_plus.shape= ')
        print(x_plus.shape)

        x_plus_scaled = scaler.transform(x_plus.reshape((1,2))).reshape((2,1))

        Sim_x_scaled = np.append(Sim_x_scaled, x_plus_scaled)
        Sim_x_scaled = Sim_x_scaled.reshape((i+2, 2))

        Sim_u = np.append(Sim_u, prediction_u)

    print(Sim_u)
    print(Sim_x)

    Sim_u = Sim_u.reshape((S,3))

    return Sim_u, Sim_x

# Creates new disturbances by adding gaussian (normal) noise
# Inputs : original disturbance vector
# Outputs : disturbance with additive gaussian noise
def create_new_disturbance(d_full, noise_level=10):
    noise = noise_level*np.random.normal(0, 1, d_full.shape)
    d_full_withNoise = d_full + noise
    #print('Original disturbance')
    #plot_disturbance(d_full)
    return d_full_withNoise

def import_disturbance(filepath='external_disturbances.mat'):
    mat_disturbance = sio.loadmat(filepath)
    print('disturbance vector loaded')
    d_full = np.column_stack((mat_disturbance['room_temp'], mat_disturbance['sol_rad'], mat_disturbance['int_gains']))

    print('peek into d_full (First 5 elements) :')
    print(d_full[0:5, :])
    return d_full


def open_test_csv(filepath='test_data.csv'):
    data = pd.read_csv(filepath)
    print('test data loaded from %s'%filepath)
    return data.values

def plot_mpc(mpc_u, mpc_x):
    """### Plot the results"""

    # matplotlib to plot the results
    import matplotlib.pyplot as plt

    print('*As a reminder, x_init = %s*'%mpc_x[0, :])

    # plot the states
    plt.figure(1)
    plt.hold = True;
    plt.plot(mpc_x[:,0])
    plt.title('state x[0] (room temp Tr)')
    plt.xlabel('t')

    plt.figure(2)
    plt.hold = True;
    plt.plot(mpc_x[:,1])
    plt.title('state x[1] (Energy in battery Ebat)')
    plt.xlabel('t')

    # plot the inputs
    plt.figure(3)
    plt.hold = True;
    for k in range(mpc_u.shape[1]):
      plt.plot(mpc_u[:,k])
    plt.title('inputs')
    plt.xlabel('t')
    # show the plots
    plt.show()

def plot_compare(mpc_u, mpc_x, mpc_g_mixed, Sim_u, Sim_x, Sim_g_mixed):
    """### Plot the results"""

    # matplotlib to plot both the mpc simulation and the DL simulation on the same graphs
    import matplotlib.pyplot as plt

    print('*As a reminder, x_init = %s*'%mpc_x[0, :])

    # plot the states
    plt.figure(1)
    plt.hold = True;
    plt.plot(mpc_x[:,0], '--')
    plt.plot(Sim_x[:,0])
    plt.title('state x[0] (room temp Tr)')
    plt.xlabel('t')
    plt.legend(('mpc', 'DL'))

    plt.figure(2)
    plt.hold = True;
    plt.plot(mpc_x[:,1], '--')
    plt.plot(Sim_x[:,1])
    plt.title('state x[1] (Energy in battery Ebat)')
    plt.xlabel('t')
    plt.legend(('mpc', 'DL'))


    # plot the inputs
    plt.figure(3)
    plt.hold = True;
    for k in range(mpc_u.shape[1]):
      plt.plot(mpc_u[:, k], '--')
      plt.plot(Sim_u[:, k])
    plt.title('MPC inputs')
    plt.xlabel('t')
    plt.legend(('mpc', 'DL','mpc', 'DL','mpc', 'DL'))

    # plot the constraints
    plt.figure(4)
    plt.hold = True
    for k in range(mpc_g_mixed.shape[1]):
        plt.plot(mpc_g_mixed[:, k], '--')
        plt.plot(Sim_g_mixed[:, k])

    plt.title('mixed constraints')
    plt.xlabel('t')
    plt.legend(('mpc', 'DL','mpc', 'DL'))


    # show the plots
    plt.show()
    # show the plots

    # plt.figure(4)
    # plt.hold = True;
    # for k in range(mpc_u.shape[1]):
    #   plt.plot(Sim_u[:, k])
    # plt.title('DL controller inputs')
    # plt.xlabel('t')

    plt.show()


def plot_disturbance(d_full, title='Disturbances'):
    print('Plotting the disturbances')

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.hold = True;
    plt.plot(d_full[:,0])

    plt.figure(1)
    plt.hold = True;
    plt.plot(d_full[:,1])

    plt.figure(1)
    plt.hold = True;
    plt.plot(d_full[:,2])

    plt.xlabel('t')
    plt.title(title)
    plt.legend(('Room temp', 'Solar Radiation', 'Internal Gains'))

    plt.show()

# Just added this one in order to plot the mpc controller and compare it to the DL controller
def simulate_MPC(d_full, S = 100, N=10, x_init = np.array([[20],[50000]])):
    ##Define a linear system as a CasADi function"""
    A = np.array([[0.8511, 0],[0, 1]])
    B = np.array([[0.0035, 0, 0],[0, -5, 0]])

    E = (1e-03)*np.array([[22.217, 1.7912, 42.212],[0, 0, 0]])

    D = np.array([[-1, 1, 1], [1, 1, 1]])
    G_mixed = np.array([[0, 0.5, 0], [0, 0.5, 0]])

    ## Define the optimization variables for MPC

    nx = A.shape[1]
    nu = B.shape[1]
    nm = D.shape[1] # this is for the mixed variables
    nd = E.shape[1] # this is for the disturbance variable

    x = SX.sym("x",nx,1)
    u = SX.sym("u",nu,1)
    m = SX.sym("m",nm,1) # Mixed variable
    d = SX.sym("d",nd,1) # Disturbance variable

    print('nx=%s'%nx)
    print('nu=%s'%nu)
    print('nm=%s'%nm)
    print('nd=%s'%nd)

    """## Choose the reference battery energy """
    #@title choose Ebat_ref
    Ebat_ref = 50000 #@param {type:"slider", min:0, max:200000, step:1000}

    """## Choose the tuning of MPC"""

    #@title Choose prediction horizon N
    #N = 7 #@param {type:"slider", min:1, max:15, step:1}

    #@title Choose number of steps S
    # S = 100 #@param {type:"slider", min:1, max:144, step:1}

    #@title Choose the penalty parameter gamma
    gamma = 4.322 #@param {type:"slider", min:0, max:10, step:0.0001}

    """# Define the dynamics as a CasADi expression"""
    # Fill d here from the .mat disturbance file

    # For collab only
    #!wget -O external_disturbances.mat https://www.dropbox.com/s/57ta25v9pg94lbw/external_disturbances.mat?dl=0
    #!ls


    #mat_disturbance = sio.loadmat('external_disturbances.mat')

    #d_full = np.column_stack((mat_disturbance['room_temp'], mat_disturbance['sol_rad'], mat_disturbance['int_gains']))

    #print('disturbance vector successfully loaded in vector d_full')
    print('length of d_full:%i'%(d_full.shape[0]))

    d_0 = d_full[0, 0]
    d_1 = d_full[0, 1]
    d_2 = d_full[0, 2]
    print('first line of d (3 columns)')
    print('d[0,0] = %f'%d_0)
    print('d[0,1] = %f'%d_1)
    print('d[0,2] = %f'%d_2)

    # Definition of the system, and the mixed constraint equations
    output_sys = mtimes(A,x) + mtimes(B,u) + mtimes(E, d)
    output_mixed = mtimes(D,u) + mtimes(G_mixed,d)

    system = Function("sys", [x,u,d], [output_sys])
    mixed = Function("sys", [u,d], [output_mixed])

    """### Construct CasADi objective function"""


    ### state cost
    J_stage_exp = u[2] + gamma*mtimes((x[1]-Ebat_ref),(x[1]-Ebat_ref))
    J_stage = Function('J_stage',[x,u],[J_stage_exp])

    # ### terminal cost ?? How ?
    # Suggestion : Terminal cost is stage cost function at last x_k (x_k[N])
    J_terminal_exp = gamma*mtimes((x[1]-Ebat_ref),(x[1]-Ebat_ref))
    J_terminal = Function('J_terminal',[x],[J_terminal_exp])
    # J_terminal = Function('J_terminal',[x],[J_terminal_exp])

    """## Define optimization variables"""

    X = SX.sym("X",(N+1)*nx,1)
    U = SX.sym("U",N*nu,1)

    # Added by me : Mixed constraints optimization variable M
    M = SX.sym("M",N*nu,1)

    """## Define constraints"""

    # state constraints :  20.0<=Tr<=23 and 0.0 ≤ SoC ≤ 200000
    lbx = np.array([[20],[0]])
    ubx = np.array([[23],[200000]])
    # input constraints
    lbu = np.array([[-1000],[-500],[-500]])
    ubu = np.array([[1000],[500],[500]])
    # mixed constraints ?
    lbm = np.array([[0], [0]])
    ubm = np.array([[inf], [inf]])

    """## Initialize vectors and matrices"""

    # Initializing the vectors
    # initial state vector has to be initialize with a feasible solution

    ############### Commented out to modularize the code ########
    # x_init = np.array([[21],[150000]]) #Arbitrary (random) feasible solution
    # #############################################################

    # Storing u_k and x_k in history matrices mpc_x and mpc_u
    mpc_x = np.zeros((S+1,nx))
    mpc_x[0,:] = x_init.T
    mpc_u = np.zeros((S,nu))

    #added by me to store mixed constraints values at each step
    mpc_g_mixed = np.zeros((S, G_mixed.shape[0]))

    """## MPC loop"""

    for step in range(S):

        ### formulate optimization problem
        J = 0
        lb_X = []
        ub_X = []
        lb_U = []
        ub_U = []
        # Added by me : bound vectors for mixed constraints
        lb_M = []
        ub_M = []
        #####################
        G = []
        lbg = []
        ubg = []

        ###
        for k in range(N):
            d_k = d_full[step + k,:] # check correct index!
            x_k = X[k*nx:(k+1)*nx,:]
            x_k_next = X[(k+1)*nx:(k+2)*nx,:]
            u_k = U[k*nu:(k+1)*nu,:]

            # objective
            J += J_stage(x_k,u_k)

            # equality constraints (system equation)
            x_next = system(x_k,u_k,d_k)

            # mixed constraints vector calculation
            g_mixed = mixed(u_k, d_k)

            if k == 0:
                G.append(x_k)
                lbg.append(x_init)
                ubg.append(x_init)

            G.append(x_next - x_k_next)
            lbg.append(np.zeros((nx,1)))
            ubg.append(np.zeros((nx,1)))

            # Added by me : mixed constraints with their bounds
            G.append(g_mixed)
            lbg.append(lbm)
            ubg.append(ubm)

            # inequality constraints
            lb_X.append(lbx)
            ub_X.append(ubx)
            lb_U.append(lbu)
            ub_U.append(ubu)
            # added by me
            #lb_M.append(lbm)
            #ub_M.append(ubm)
            ####################

        ## Terminal cost and constraints
        x_k = X[N*nx:(N+1)*nx,:]
        J += J_terminal(x_k)
        lb_X.append(lbx)
        ub_X.append(ubx)

        ### solve optimization problem
        lb = vertcat(vertcat(*lb_X),vertcat(*lb_U))
        ub = vertcat(vertcat(*ub_X),vertcat(*ub_U))
        prob = {'f':J,'x':vertcat(X,U),'g':vertcat(*G)}
        solver = nlpsol('solver','ipopt',prob)
        res = solver(lbx=lb,ubx=ub,lbg=vertcat(*lbg),ubg=vertcat(*ubg))
        u_opt = res['x'][(N+1)*nx:(N+1)*nx+nu,:]

        # Ignore this
        # g_constrained = res['g'][N*2]
        # print('res["x"] = %s'%res['x'])
        # print('u_opt = %s'%u_opt)
        # print('res["g"] = : %s'%g_constrained)
        ####################################

        ### simulate the system
        x_plus = system(x_init.T,u_opt, d_full[step,:])
        mpc_x[step+1,:] = x_plus.T
        mpc_u[step,:] = u_opt.T
        x_init = x_plus
        # added by me
        g_plus = mixed(u_opt, d_full[step,:])
        mpc_g_mixed[step, :] = g_plus.T
        # print(mpc_g_mixed)
        ######################
    return mpc_u, mpc_x, mpc_g_mixed, d_full



if __name__ == '__main__':
    filepath_trained_model = 'Final_model_varDist_20epochs_100000lines.h5'
    trained_model = load_model(filepath_trained_model)
    print('loaded trained model loaded from :%s'%filepath_trained_model)

    # Saving the model to a png representation
    from keras.utils import plot_model
    plot_model(trained_model, show_shapes=True, to_file='trained_model.png')

    # we should load the test data :
    test_data = open_test_csv(filepath='test_data_not_scaled.csv')
    print('test_data shape :')
    print(test_data.shape)

    X_test = test_data[:, 1:18]
    y_test = test_data[:, 18:21]

    print('X_test =')
    print(X_test.shape)
    print('y_test =')
    print(y_test)


    # Making simple predictions now

    predictions = np.array([])
    for i in range(X_test.shape[0]):
        # predictions = np.append(predictions, trained_model.predict(np.array([[X_test[i,0], X_test[i,1], X_test[i,2], X_test[i,3], X_test[i,4]]])))
        temp = np.array([])
        for j in range(X_test.shape[1]):
            temp = np.append(temp, X_test[i, j])
        temp = temp.reshape((1, X_test.shape[1]))
        predictions = np.append(predictions, trained_model.predict(temp))

    predictions = predictions.reshape((X_test.shape[0], y_test.shape[1]))


    print('X_test:')
    print(X_test[10:20, :])

    print('Prediction matrix :')
    print(predictions[10:20, :])

    print('Compare it to label matrix y_test :')
    print(y_test[10:20, :])

    test_data_scaled = open_test_csv(filepath='test_data_scaled.csv')

    X_test_scaled = test_data_scaled[:, 1:18]
    y_test_scaled = test_data_scaled[:, 18:21]

    print('Metrics of evaluation')
    print(trained_model.metrics_names)
    print('Model evaluation : ')
    print(trained_model.evaluate(X_test_scaled, y_test_scaled))


    d_full = import_disturbance()
    print(d_full.shape)

    d_full_withNoise = create_new_disturbance(d_full, noise_level=10)

    plot_disturbance(d_full_withNoise, title='disturbance with additive noise level=10')

    Sim_u, Sim_x = simulate_DLcontroller(trained_model, d_full_withNoise, X_test, S = 100)

    print('Sim_u.shape=')
    print(Sim_u.shape)
    print('Sim_x.shape=')
    print(Sim_x.shape)
    print('\n')
    print('Sim_x = ')
    print(Sim_x)
    print('\n')
    print('Sim_u = ')
    print(Sim_u)


    mpc_u, mpc_x, mpc_g_mixed, _ = simulate_MPC(d_full_withNoise, S = 100, N=5, x_init = np.array([[20],[50000]]))

    Sim_g_mixed = generate_mixed_constraints(Sim_u, d_full_withNoise)

    plot_compare(mpc_u, mpc_x, mpc_g_mixed, Sim_u, Sim_x, Sim_g_mixed)
