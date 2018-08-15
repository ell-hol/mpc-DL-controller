
# coding: utf-8

# In[2]:


#!pip install --upgrade pip
#!pip install casadi


# In[3]:


# Import casadi
from casadi import *
# Import Numpy
import numpy as np
# Import matplotlib
import matplotlib.pyplot as plt
# Import Scipy to load .mat file
import scipy.io as sio
import pdb


# In[4]:


def simulate_MPC(d_full, S = 100, N=10, x_init = np.array([[21],[150000]])):
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



def import_disturbance(filepath='external_disturbances.mat'):
    mat_disturbance = sio.loadmat(filepath)
    print('disturbance vector loaded')
    d_full = np.column_stack((mat_disturbance['room_temp'], mat_disturbance['sol_rad'], mat_disturbance['int_gains']))

    print('peek into d_full (First 5 elements) :')
    print(d_full[0:5, :])
    return d_full


# Creates new disturbances by adding gaussian (normal) noise
def create_new_disturbance(d_full, noise_level=10):
    noise = noise_level*np.random.normal(0, 1, d_full.shape)
    d_full_withNoise = d_full + noise
    print('Original disturbance')
    #plot_disturbance(d_full)
    return d_full_withNoise




def plot_mpc(mpc_u, mpc_x, mpc_g_mixed):
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

    # plot the constraints
    plt.figure(4)
    plt.hold = True
    for k in range(mpc_g_mixed.shape[1]):
        plt.plot(mpc_g_mixed[:, k])
    plt.title('mixed constraints')
    plt.xlabel('t')

    # show the plots
    plt.show()


# Generates nb_x0 possible allowed combinations of the initial state vector x0
# Returns the array of combinations
def generate_list_x0(nb_x0 = 1000):
    x0_Tr = np.linspace(20, 23, num=int(sqrt(nb_x0)))
    x0_Ebat = np.linspace(0, 200000, num=int(sqrt(nb_x0)))

    x0_combinations = []

    import itertools
    counter = 0
    for i in itertools.product(x0_Tr, x0_Ebat):
        x0_combinations.append([i[0], i[1]])
        # print(i)
        counter += 1

    return np.array(x0_combinations)

# This will be used to save training/testing data in csv format
def csv_dump(X_data, y_data, filepath='last_simulation_data100000lines.csv'):
    temp = np.concatenate((X_data, y_data), axis=1)
    import pandas as pd
    # df = pd.DataFrame(temp, columns=['Tr0','Ebat0','dT','dsr','dint','Phvac','Pbat','Pgrid'])
    df = pd.DataFrame(temp)
    print(df.head())
    try:
        df.to_csv(filepath)
        print('csv data file successfully written to %s'%filepath)
    except IOError as e:
        print(e)



# Core function of this script :
# Takes the list of combinations of x0
# Then simulates MPC optimization for each different x0 for N=5 and S=100
# Returns the simulation data : x and u
def generate_data(list_x0, d_training, N=5, S=100):


    data = np.array([])

    mpc_x_all = np.array([])
    mpc_u_all = np.array([])

    # d_matrix contains (N*3 unrolled disturbance vectors) and should be of size S
    d_matrix = np.array([])


    for i in range(S):
        d_temp = create_new_disturbance(d_training[i:(i+N), :], noise_level=10)
        d_matrix = np.append(d_matrix, d_temp.reshape((1,N*3)))

    d_matrix = d_matrix.reshape((S, N*3))

    for i in range(list_x0.shape[0]):
        mpc_u, mpc_x, _, _ = simulate_MPC(d_training, x_init=list_x0[i,:], N=N, S=S)

        mpc_x_all = np.append(mpc_x_all, mpc_x[0:(mpc_x.shape[0]-1),:])
        mpc_u_all = np.append(mpc_u_all, mpc_u)

    mpc_x_all = mpc_x_all.reshape((list_x0.shape[0]*(mpc_x.shape[0]-1), 2))
    mpc_u_all = mpc_u_all.reshape((list_x0.shape[0]*mpc_u.shape[0], 3))

    data_x_full = np.zeros((mpc_x_all.shape[0], mpc_x_all.shape[1]+d_matrix.shape[1]))
    # duplicating disturbance list_x0.shape[0] times :
    d_final = np.array([])
    for i in range(list_x0.shape[0]):
        d_final = np.append(d_final, d_matrix)
    d_final = d_final.reshape((mpc_x_all.shape[0], d_matrix.shape[1]))

    data_x_full[:, 0:2] = mpc_x_all
    data_x_full[:, 2:] = d_final

    return data_x_full, mpc_u_all


if __name__ == '__main__':
    d_full = import_disturbance()
    list_x0 = generate_list_x0()
    data_x, data_y = generate_data(list_x0, d_full)
    csv_dump(data_x, data_y, filepath='Varying_disturbance_simulation_data10000lines.csv')
