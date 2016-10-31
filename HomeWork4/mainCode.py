__author__ = 'deepika'

FILENAME = 'MiniBooNE_PID.txt'
import hw_utils as hw
import time
import json
import Data as d

def displayJson(results):
    json_string = json.dumps([ob.__dict__ for ob in results])
    print json_string

def partd_a(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    archs = [[50,2],[50,50,2],[50,50,50,2],[50,50,50,50,2]]
    results = hw.testmodels(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, archs=archs,
                        actfn='linear', last_act='softmax', reg_coeffs=[0.0],
				        num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
					    sgd_Nesterov=False, EStop=False, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

def partd_b(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    archs = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
    results = hw.testmodels(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, archs=archs,
                        actfn='linear', last_act='softmax', reg_coeffs=[0.0],
				        num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
					    sgd_Nesterov=False, EStop=False, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

def parte(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    archs = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
    results = hw.testmodels(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, archs=archs,
                        actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0],
				        num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
					    sgd_Nesterov=False, EStop=False, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

def partf(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    archs = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
    results = hw.testmodels(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, archs=archs,
                        actfn='relu', last_act='softmax', reg_coeffs=[0.0],
				        num_epoch=30, batch_size=1000, sgd_lr=5*pow(10, -4), sgd_decays=[0.0], sgd_moms=[0.0],
					    sgd_Nesterov=False, EStop=False, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

#L2- Regularization
def partg(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    regularization_architecture = [ [50, 800, 500, 300, 2] ]
    regularization = [pow(10,-7), 5*pow(10,-7), pow(10, -6), 5*pow(10, -6), pow(10, -5)]
    results = hw.testmodels(X_tr,y_tr,X_te,y_te, regularization_architecture, actfn='relu', last_act='softmax', reg_coeffs=regularization,
				num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)
    print " Time Taken = ", time.time() - start_time
    displayJson(results)

#Early Stopping and L2-regularization
def parth(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    regularization_architecture = [ [50, 800, 500, 300, 2] ]
    regularization = [pow(10,-7), 5*pow(10,-7), pow(10, -6), 5*pow(10, -6), pow(10, -5)]

    results = hw.testmodels(X_tr,y_tr,X_te,y_te, regularization_architecture, actfn='relu', last_act='softmax', reg_coeffs=regularization,
				num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=True, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

#SGD with weight decay
def parti(X_tr,y_tr,X_te,y_te):
    start_time = time.time()

    regularization = [5* pow(10, -7)]
    decay_list = [pow(10,-5), 5*pow(10,-5), pow(10,-4), 3*pow(10,-4), 7*pow(10,-4), pow(10,-3)]

    results = hw.testmodels(X_tr,y_tr,X_te,y_te, [[50,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=regularization,
				num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=decay_list, sgd_moms=[0.0],
					sgd_Nesterov=False, EStop=False, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

#Momentum
def partj(X_tr,y_tr,X_te,y_te):
    start_time = time.time()
    archs = [[50, 800, 500, 300, 2]]
    optimal_decay = 5*pow(10,-5)

    results = hw.testmodels(X_tr,y_tr,X_te,y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
				num_epoch=50, batch_size=1000, sgd_lr=0.00001, sgd_decays=[optimal_decay], sgd_moms=[0.99, 0.98, 0.95, 0.9, 0.85],
					sgd_Nesterov=True, EStop=False, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

#combining
def partk(X_tr,y_tr,X_te,y_te):
    start_time = time.time()

    optimal_regu = 5*pow(10,-6)
    optimal_decay = 5*pow(10,-5)
    optimal_momentum = 0.99

    results = hw.testmodels(X_tr,y_tr,X_te,y_te, [[50, 800, 500, 300, 2]], actfn='relu', last_act='softmax', reg_coeffs=[optimal_regu],
				num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=[optimal_decay], sgd_moms=[optimal_momentum],
					sgd_Nesterov=True, EStop=True, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

#Grid search with cross validation
def partl(X_tr,y_tr,X_te,y_te):
    start_time = time.time()

    decay_list = [pow(10,-5), 5*pow(10,-5), pow(10,-4)]
    archs = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
    regularization = [pow(10,-7), 5*pow(10,-7), pow(10, -6), 5*pow(10, -6), pow(10, -5)]

    results = hw.testmodels(X_tr,y_tr,X_te,y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=regularization,
				num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=decay_list, sgd_moms=[0.99],
					sgd_Nesterov=True, EStop=True, verbose=0)

    print " Time Taken = ", time.time() - start_time
    displayJson(results)

if __name__=="__main__":

    #Part A) Load and normalize
    start_time=time.time()
    X_tr,y_tr,X_te,y_te = hw.loaddata(FILENAME)
    print "Time taken to load data = ", time.time() - start_time
    start_time=time.time()
    X_tr, X_te = hw.normalize(X_tr, X_te)
    print "Time taken to normalize data = ", time.time() - start_time

    #Part D)
    #partd_a(X_tr,y_tr,X_te,y_te)
    #partd_b(X_tr,y_tr,X_te,y_te)

    #parte(X_tr,y_tr,X_te,y_te)

    #partf(X_tr,y_tr,X_te,y_te)

    #partg(X_tr,y_tr,X_te,y_te)

    #parth(X_tr,y_tr,X_te,y_te)

    #parti(X_tr,y_tr,X_te,y_te)

    #partj(X_tr,y_tr,X_te,y_te)

    #print " Combining optimal values"
    #partk(X_tr,y_tr,X_te,y_te)

    partl(X_tr,y_tr,X_te,y_te)

