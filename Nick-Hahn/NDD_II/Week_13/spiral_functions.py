import random
from sdtf import StreamDecisionForest
from proglearn.forest import LifelongClassificationForest
from proglearn.sims import generate_spirals
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_spirals(spiral1, y_spiral1, num_spirals1, spiral2, y_spiral2, num_spirals2):
    '''
    plots spiral 1 and spiral 2
    '''
    colors = sns.color_palette("Dark2", n_colors=5)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    clr = [colors[i] for i in y_spiral1]
    ax[0].scatter(spiral1[:, 0], spiral1[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(str(num_spirals1) + " spirals", fontsize=30)
    ax[0].axis("off")

    clr = [colors[i] for i in y_spiral2]
    ax[1].scatter(spiral2[:, 0], spiral2[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title(str(num_spirals2) + " spirals", fontsize=30)
    ax[1].axis("off")

def load_results():
    '''
    loads results from csv files
    '''
    pass

def plot_results(t1_error, t2_error, ble, fle):
    '''
    plots generalization error (3 spirals), generalization error (5 spirals), and log FLE/BLE
    '''
    pass

def run(n_spiral3, n_spiral5, mc_rep, n_test, n_trees, n_update):
    '''
    runs spiral experiment and saves results to csv
    '''
    for i in range(mc_rep):
        t1_error, t2_error, ble, fle = experiment()

def experiment(n_task1, n_task2, n_test=1000, n_trees =10, n_update, max_depth=None, random_state = None)
    '''
    one rep of spiral experiment
    '''
    #arrays for storing results 
    n = int((n_task1+n_task2)/n_update)
    t1_error = 0.5*np.ones((2,n))
    t2_error = 0.5*np.ones((2,n))
    ble = np.zeros((2, n))
    fle = np.zeros((2, n))

    #instantiate classifiers
    synf_single_t1 = LifelongClassificationForest(default_n_estimators=n_trees)
    synf_multi_t1 = LifelongClassificationForest(default_n_estimators=n_trees)
    synf_single_t2 = LifelongClassificationForest(default_n_estimators=n_trees)
    synf_multi_t2 = LifelongClassificationForest(default_n_estimators=n_trees)
    sdf_single_t1 = StreamDecisionForest(n_estimators=n_trees)
    sdf_multi_t1 = StreamDecisionForest(n_estimators=n_trees)
    sdf_single_t2 = StreamDecisionForest(n_estimators=n_trees)
    sdf_multi_t2 = StreamDecisionForest(n_estimators=n_trees)

    # Generate initial/test spirals
    X_t1, y_t1 = generate_spirals(n_update, 3, noise=0.8)
    test_t1, test_t1_y = generate_spirals(n_test, 3, noise=0.8)

    X_t2, y_t2 = generate_spirals(n_update, 5, noise=0.4)
    test_t2, test_t2_y = generate_spirals(n_test, 5, noise=0.4)

    # add task 1
    synf_single_t1.add_task(X_t1, y_t1, task_id=0, classes = [0,1,2])
    synf_multi_t1.add_task(X_t1, y_t1, task_id=0, classes = [0,1,2])
    synf_multi_t2.add_task(X_t1, y_t1, task_id=0, classes = [0,1,2])
    sdf_single_t1.partial_fit(X_t1, y_t1, classes=[0,1,2])
    sdf_multi_t1.partial_fit(X_t1, y_t1, classes=[0,1,2])
    sdf_multi_t2.partial_fit(X_t1, y_t1, classes=[0,1,2])
    
    # update task 1
    for i in range(int(n_task1/n_update)-1):
        X_t1, y_t1 = generate_spirals(n_update, 3, noise=0.8)
        synf_single_t1.update_task(X_t1, y_t1, task_id=0)



    # add task 2 


    #update task 2 

    # calculate learning efficiencies 





    return t1_error, t2_error, ble, fle

