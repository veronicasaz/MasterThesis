#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import time

import AstroLibraries.AstroLib_Basic as AL_BF 


###############################################
# EVOLUTIONARY ALGORITHM
###############################################
# Random initialization--> fitness calculation--> offspring creation --> immigration, mutation --> convergence criterion
def EvolAlgorithm(f, bounds, *args, **kwargs):
    """
    EvolAlgorithm: evolutionary algorithm
    INPUTS:
        f: function to be analyzed
        x: decision variables
        bounds: bounds of x to initialize the random function
        x_add: additional parameters for the function. As a vector
        ind: number of individuals. 
        cuts: number of cuts to the variable
        tol: tolerance for convergence
        max_iter: maximum number of iterations (generations)
        max_iter_success
        elitism: percentage of population elitism
        bulk_fitness: if True, the data has to be passed to the function all 
                    at once as a matrix with each row being an individual
    """
    x_add = kwargs.get('x_add', False)
    ind = kwargs.get('ind', 100)
    cuts = kwargs.get('cuts', 1)
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1e3)
    max_iter_success = kwargs.get('max_iter_success', 1e2)
    elitism = kwargs.get('elitism',0.1)
    mut = kwargs.get('mutation',0.01)
    cons = kwargs.get('cons', None)
    bulk = kwargs.get('bulk_fitness', False)

    
    def f_evaluate(pop_0):
        if bulk == True:
            if x_add == False:
                pop_0[:,0] = f(pop_0[:,1:])
            else:
                pop_0[:,0] = f(pop_0[:,1:], x_add)
        else:
            if x_add == False: # No additional arguments needed
                for i in range(ind):
                    pop_0[i,0] = f(pop_0[i,1:])
            else:
                for i in range(ind):
                    pop_0[i,0] = f(pop_0[i,1:], x_add)

        return pop_0

    ###############################################
    ###### GENERATION OF INITIAL POPULATION #######
    ###############################################
    pop_0 = np.zeros([ind, len(bounds)+1])
    for i in range(len(bounds)):
        pop_0[:,i+1] = np.random.rand(ind) * (bounds[i][1]-bounds[i][0]) + bounds[i][0]

    ###############################################
    ###### FITNESS EVALUATION               #######
    ###############################################
    pop_0 = f_evaluate(pop_0)
    
    Sol = pop_0[pop_0[:,0].argsort()]
    minVal = min(Sol[:,0])
    x_minVal = Sol[0,:]
    
    ###############################################
    ###### NEXT GENERATION                  #######
    ###############################################
    noImprove = 0
    counter = 0
    lastMin = minVal
    
    Best = np.zeros([max_iter+1,len(bounds)+1])
    while noImprove <= max_iter_success and counter <= max_iter :
        
        ###############################################
        #Generate descendents

        #Elitism
        ind_elit = int(round(elitism*ind))

        children = np.zeros(np.shape(pop_0))
        children[:,1:] = Sol[:,1:]

        #Separate into the number of parents  
        pop = np.zeros(np.shape(pop_0))
        pop[:ind_elit,:] = children[:ind_elit,:] #Keep best ones
        np.random.shuffle(children[:,:]) #shuffle the others
        

        for j in range ( (len(children)-ind_elit) //2 ):
            if len(bounds) == 2:
                cut = 1
            else:
                cut = np.random.randint(1,len(bounds)-1)

            pop[ind_elit +2*j,1:] = np.concatenate((children[2*j,1:cut+1],children[2*j +1,cut+1:]),axis = 0)
            pop[ind_elit+ 2*j + 1,1:] = np.concatenate((children[2*j+1,1:cut+1],children[2*j ,cut+1:]),axis = 0)
        
        if (len(children)-ind_elit) %2 != 0:
            pop[-1,:] = children[-ind_elit,:]

        #Mutation
        for i in range(ind):
            for j in range(len(bounds)):
                if np.random.rand(1) < mut: #probability of mut                    
                    pop[i,j+1] =  np.random.rand(1) * (bounds[j][1]-bounds[j][0]) + bounds[j][0]
        
        ###############################################
        # Fitness
        pop = f_evaluate(pop)

        Sol = pop[pop[:,0].argsort()]
        minVal = min(Sol[:,0])

        ###############################################
        #Check convergence        
        if  minVal >= lastMin: 
            noImprove += 1
#         elif abs(lastMin-minVal)/lastMin > tol:
#             noImprove += 1
        else:
#             print('here')
            lastMin = minVal 
            x_minVal = Sol[0,1:]
            noImprove = 0
            Best[counter,:] = Sol[0,:] 
        
        print(counter, "Minimum: ", minVal)
        counter += 1 #Count generations 
        if counter % 20 == 0:
            AL_BF.writeData(x_minVal, 'w', 'SolutionEA.txt')
        
        # print(counter)
    print("minimum:", lastMin)
    print("Iterations:", counter)
    print("Iterations with no improvement:", noImprove)
    
    return x_minVal, lastMin


# # Test
# bnds = (bnd_tflight,bnd_vE)
# for i in range(Nimp):
#     bnds += (bnd_deltavmag,)
# for i in range(Nimp):
#     bnds += (bnd_deltavang,)
# bnd_1 = (1,4)
# bnds = (bnd_1,bnd_1,bnd_1)

# def f(x,y):
#     return (2*x[0]+x[1]+x[2])

# x = EvolAlgorithm(f, [2,3,4], bnds, [],ind=100, max_iter = 1000, mutation = 0.01, elitism = 0.1)
# x

#######################################3
# EVOLUTIONARY ALGORITHM WITH CONSTRAINT PENALIZATION
#######################################
      
def EvolAlgorithm_cons(f, bounds, *args, **kwargs):
    """
    EvolAlgorithm: evolutionary algorithm
    INPUTS:
        f: function to be analyzed
        x: decision variables
        bounds: bounds of x to initialize the random function
        x_add: additional parameters for the function. As a vector
        ind: number of individuals. 
        cuts: number of cuts to the variable
        tol: tolerance for convergence
        max_iter: maximum number of iterations (generations)
        max_iter_success
        elitism: percentage of population elitism
        mut: mutation rate
        immig: migration rate (new individuals)
    """
    x_add = kwargs.get('x_add', False)
    ind = kwargs.get('ind', 100)
    cuts = kwargs.get('cuts', 1)
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1e3)
    max_iter_success = kwargs.get('max_iter_success', 1e2)
    elitism = kwargs.get('elitism',0.1)
    mut = kwargs.get('mutation',0.01)
    immig = kwargs.get('immig',0.01)
    cons = kwargs.get('cons', None)

    
    ###############################################
    ###### GENERATION OF INITIAL POPULATION #######
    ###############################################
    pop_0 = np.zeros([ind, len(bounds)+1])
    for i in range(len(bounds)):
        pop_0[:,i+1] = np.random.rand(ind) * (bounds[i][1]-bounds[i][0]) + bounds[i][0]
    
 
    
    ###############################################
    ###### FITNESS EVALUATION               #######
    ###############################################
    if x_add == False: # No additional arguments needed
        for i in range(ind):
            pop_0[i,0] = f(pop_0[i,1:])
    else:
        for i in range(ind):
            pop_0[i,0] = f(pop_0[i,1:], x_add)
    
    feas =  cons[0](pop_0[:,1:])
    print('Number feasible', np.count_nonzero(feas==0))
    pop_0[:, 0] += feas*cons[1] # Add penalty to unfeasible ones

    Sol = pop_0[pop_0[:,0].argsort()]
    minVal = min(Sol[:,0])
    x_minVal = Sol[0,:]
    
    ###############################################
    ###### NEXT GENERATION                  #######
    ###############################################
    noImprove = 0
    counter = 0
    lastMin = minVal
    
    Best = np.zeros([max_iter+1,len(bounds)+1])
    while noImprove <= max_iter_success and counter <= max_iter :
        
        ###############################################
        #Generate descendents

        #Elitism
        ind_elit = int(round(elitism*ind))

        children = np.zeros(np.shape(pop_0))
        children[:,1:] = Sol[:,1:]

        #Separate into the number of parents  
        pop = np.zeros(np.shape(pop_0))
        pop[:ind_elit,:] = children[:ind_elit,:] #Keep best ones
        np.random.shuffle(children[:,:]) #shuffle the others
        

        for j in range ( (len(children)-ind_elit) //2 ):
            if len(bounds) == 2:
                cut = 1
            else:
                cut = np.random.randint(1,len(bounds)-1)

            pop[ind_elit +2*j,1:] = np.concatenate((children[2*j,1:cut+1],children[2*j +1,cut+1:]),axis = 0)
            pop[ind_elit+ 2*j + 1,1:] = np.concatenate((children[2*j+1,1:cut+1],children[2*j ,cut+1:]),axis = 0)
        if (len(children)-ind_elit) %2 != 0:
            pop[-1,:] = children[-ind_elit,:]

        
        #Mutation
        for i in range(ind):
            for j in range(len(bounds)):
                if np.random.rand(1) < mut: #probability of mut                    
                    pop[i,j+1] =  np.random.rand(1) * (bounds[j][1]-bounds[j][0]) + bounds[j][0]
        
        #Immigration
        ind_immig = int(round(immig*ind))
        for i in range(len(bounds)):
            pop[-ind_immig:, i+1] = np.random.rand(ind_immig) * (bounds[i][1]-bounds[i][0]) + bounds[i][0]
        

        ###############################################
        # Fitness
        if x_add == False: # No additional arguments needed
            for i in range(ind):
                pop[i,0] = f(pop[i,1:])
        else:
            for i in range(ind):
                pop[i,0] = f(pop[i,1:], x_add)

        feas =  cons[0](pop[:,1:])
        print('Number feasible', np.count_nonzero(feas==0))
        pop[:, 0] += feas*cons[1] # Add penalty to unfeasible ones

        Sol = pop[pop[:,0].argsort()]
        minVal = min(Sol[:,0])

        ###############################################
        #Check convergence        
        if  minVal >= lastMin: 
            noImprove += 1
#         elif abs(lastMin-minVal)/lastMin > tol:
#             noImprove += 1
        else:
#             print('here')
            lastMin = minVal 
            x_minVal = Sol[0,1:]
            noImprove = 0
            Best[counter,:] = Sol[0,:] 
        
        print(counter, "Minimum: ", minVal)
        counter += 1 #Count generations 
        if counter % 20 == 0:
            AL_BF.writeData(x_minVal, 'w', 'SolutionEA.txt')
        
        # print(counter)
    print("minimum:", lastMin)
    print("Iterations:", counter)
    print("Iterations with no improvement:", noImprove)
    
    return x_minVal, lastMin

###############################################
# OTHERS
###############################################

def randomJump(f,x,bounds,*args, **kwargs):
    x_add = kwargs.get('x_add', None)
    npoints = kwargs.get('npoints', 100)
    
    x0 = np.zeros([npoints,len(x)])
    for i in range(len(x)):
        x0[:,i] = np.random.rand(npoints) * (bounds[i][1]-bounds[i][0]) + bounds[i][0]
    
    fx = np.zeros([npoints,len(x)+1])
    for i in range(npoints):
        fx[i,0] = f(x0[i,:],x_add)
        fx[i,1:] = x0[i,:]
    
    Sol = fx[fx[:,0].argsort()]
    
    return Sol[0,:]


# In[2]:


def coordinateSearch(f,x,bounds,h,*args, **kwargs):
    """
    h: vector of length x, with the magnitude of the step
    """
    x_add = kwargs.get('x_add', None)
    tol = kwargs.get('tol', 1e-5)
    max_iter = kwargs.get('max_iter', 100)
    max_iter_success = kwargs.get('max_iter_success', 20)
    percentage = kwargs.get('percentage', 0.8)
    
    if np.all(x) == None:
        #Initial point
        x0 = np.zeros(len(x))
        for i in range(len(x)):
            x0[i] = np.random.rand(1) * (bounds[i][1]-bounds[i][0]) + bounds[i][0]
    
    x0 = x
    step = h
    improvement = 2*tol
    Prev_Min = 0
    counter = 0
    counter_success = 0

    while  improvement > tol and counter< max_iter and counter_success< max_iter_success:
        for i in range(len(x)): #Go through all variables
            x_plus = np.copy(x0)
            x_plus[i] += step[i]
            x_minus = np.copy(x0)
            x_minus[i] -= step[i]

            #Adjust to bounds
            if x_plus[i] >= bounds[i][1]:
                x_plus[i] = x0[i]
            if x_minus[i] <= bounds[i][0]:
                x_minus[i] = x0[i]

            #Evaluate
            if x_add == False: # No additional arguments needed
                fx = f(x0)
                fx_plus = f(x_plus)
                fx_minus = f(x_minus)
            else:
                fx = f(x0,x_add)
                fx_plus = f(x_plus,x_add)
                fx_minus = f(x_minus,x_add)

            Min = min([fx,fx_plus,fx_minus])
            if Min == fx_plus:
                x0 = x_plus
            elif Min == fx_minus:
                x0 = x_minus

        improvement = abs(Prev_Min - Min)/Min 
        if improvement == 0:
            improvement = 2*tol
            counter_success += 1
        else:
            counter_success = 0
        
        counter += 1
        print("Iteration", counter)
        step *= percentage
#         step = abs(Prev_Min-Min)/Prev_Min * step
        
        Prev_Min = Min        
        
        
    print('Iterations', counter)
    print('Iterations no improvement', counter_success)
#     print('Step Size',step)
    print('Min',Min)
    return x0, Min


# In[ ]:


def coordinateSearch_plot(f,x,bounds,h,*args, **kwargs):
    """
    h: vector of length x, with the magnitude of the step
    """
    x_add = kwargs.get('x_add', None)
    tol = kwargs.get('tol', 1e-5)
    max_iter = kwargs.get('max_iter', 100)
    max_iter_success = kwargs.get('max_iter_success', 20)
    percentage = kwargs.get('percentage', 0.8)
    
    if np.all(x) == None:
        #Initial point
        x0 = np.zeros(len(x))
        for i in range(len(x)):
            x0[i] = np.random.rand(1) * (bounds[i][1]-bounds[i][0]) + bounds[i][0]
    
    x0 = x
    step = h
    improvement = 2*tol
    Prev_Min = 0
    counter = 0
    counter_success = 0
    
    Best = np.zeros([100,len(x)+1])
    while  improvement > tol and counter< max_iter and counter_success< max_iter_success:
        for i in range(len(x)): #Go through all variables
                x_plus = np.copy(x0)
                x_plus[i] += step[i]
                x_minus = np.copy(x0)
                x_minus[i] -= step[i]

                #Adjust to bounds
                if x_plus[i] >= bounds[i][1]:
                    x_plus[i] = x0[i]
                if x_minus[i] <= bounds[i][0]:
                    x_minus[i] = x0[i]

                #Evaluate
                fx = f(x0,x_add)
                fx_plus = f(x_plus,x_add)
                fx_minus = f(x_minus,x_add)

                Min = min([fx,fx_plus,fx_minus])
                if Min == fx_plus:
                    x0 = x_plus
                elif Min == fx_minus:
                    x0 = x_minus

        improvement = abs(Prev_Min - Min)/Min 
        if improvement == 0:
            improvement = 2*tol
            counter_success += 1
        else:
            counter_success = 0
            Best[counter,0] = Min
            Best[counter,1:] = x0
        
        counter += 1
        step *= percentage
#         step = abs(Prev_Min-Min)/Prev_Min * step
        
        Prev_Min = Min        
        
        
    print('Iterations', counter)
    print('Iterations no improvement', counter_success)
#     print('Step Size',step)
    print('Min',Min)
    return x0, Min, Best


# In[13]:


def lineSearch(alpha,args):
    [f,x,x_add,sq] = args
    x_new = x + alpha*sq
    return f(x_new,x_add)

def SteepestDescentDirection(f,x,stepSize,alpha, *args, **kwargs):
    """
    stepSize: vector in the lenght of x
    """
    x_add = kwargs.get('x_add', None)
    tol = kwargs.get('tol', 1e1)
    max_iter = kwargs.get('max_iter', 100)
    
    
    fx = f(x,x_add)
    improvement = 2*tol
    counter = 0
    while improvement > tol or counter< max_iter:
        fx_0 = fx
        
        fx1plush = np.zeros(len(x))
        dfdx = np.zeros(len(x))
        for i in range(len(x)):
            h = stepSize[i]
            fx1plush[i] = f(x+h, x_add)
            dfdx[i] = (fx1plush[i] - fx_0) / h

        sq = - dfdx 
        
        x_new = x + alpha*sq
        fx = f(x_new,x_add)

        x = x_new
        improvement = abs(fx-fx_0)/fx_0 
        
        for i in range(len(alpha)):
            alpha[i] *= 0.9
            stepSize[i] *= 0.9
            
        counter += 1
        
    return x


###############################################
# MONOTONIC BASIN HOPPING
###############################################
# Function to change step in basin hoping
class MyTakeStep(object):
    def __init__(self, Nimp, bnds):
        self.Nimp = Nimp
        self.bnds = bnds
    def __call__(self, x):
        self.call(x)
        return x

    def call(self, x, magnitude):
        x2 = np.zeros(len(x))
        for j in range(len(x)):
            x2[j] = x[j] + magnitude* np.random.uniform(self.bnds[j][0] - x[j], \
                                    self.bnds[j][1] - x[j], 1)[0]

        # x[0] += np.random.normal(-1e3, 1e3, 1)[0] # velocity magnitude
        # x[1:3] += np.random.normal(-0.2, 0.2, 1)[0] # angle
        # x[3] += np.random.normal(-1e3, 1e3, 1)[0] # velocity magnitude
        # x[4:6] += np.random.normal(-0.2, 0.2, 1)[0] # angle
        # x[6] += np.random.normal(-30, 30, 1)[0] # time in days
        # x[7] += np.random.normal(-30, 30, 1)[0] # initial mass
        # x[8] += np.random.normal(- AL_BF.days2sec(30), AL_BF.days2sec(30), 1)[0] # transfer time 
        # for i in range(self.Nimp):
        #     x[9+i*3] += np.random.normal(-0.1, 0.1, 1)[0]
        #     x[9+i*3+1 : 9+i*3+3] += np.random.normal(-0.5, 0.5, 2)
        
        return x2

def print_fun(f, x, accepted):
    """
    print_fun: choose what to print after each iteration
    INPUTS: 
        x: current value of the decision vector
        f: function value
        accepted: iteration improves the function = TRUE. Otherwise FALSE.
    OUTPUTS: none
    """
    print("Change iteration",accepted)
    print("#####################################################################")
    # print('t',time.time() - start_time)


def print_sol(Best, bestMin, n_itercounter, niter_success):
    """
    print_sol: 
    INPUTS: 
        
    OUTPUTS: none
    """
    print("Number of iterations", n_itercounter)
    print("Number of iterations with no success", niter_success)
    print("Minimum", bestMin)
    print("x", Best)
    print("#####################################################################")


def check_feasibility(x, bnds):
    feasible = True
    for j in range(len(x)): 
        if bnds[j][0] <0:
            tol = 1.05
        else:
            tol = 0.95
        if ( x[j] < tol*bnds[j][0] ) or ( x[j] > 1.05*bnds[j][1] ): # 1.05* to have tolerance
            feasible = False
            print(j, "Within bounds?", "min", bnds[j][0], "value",x[j], "max",bnds[j][1])
    # print(feasible)

    return feasible

def check_constraints(x, cons):
    value = cons['fun'](x)
    if value == 0:
        return True
    else:
        return False

def MonotonicBasinHopping(f, x, take_step, *args, **kwargs):
    """
    Step for jump is small, as minimum cluster together.
    Jump from current min until n_no improve reaches the lim, then the jump is random again.
    """

    niter = kwargs.get('niter', 100)
    niter_success = kwargs.get('niter_success', 50)
    niter_local = kwargs.get('niter_local', 50)
    bnds = kwargs.get('bnds', None)
    cons =  kwargs.get('cons', 0)
    jumpMagnitude_default =  kwargs.get('jumpMagnitude', 0.1) # Small jumps to search around the minimum
    tolLocal = kwargs.get('tolLocal', 1e2)
    tolGlobal = kwargs.get('tolGobal', 1e-5)
    
    n_itercounter = 1
    n_noimprove = 0

    Best = x
    bestMin =  f(x)
    previousMin = f(x)
    jumpMagnitude = jumpMagnitude_default

    while n_itercounter < niter:
        n_itercounter += 1
        
        # Change decision vector to find one within the bounds
        feasible = False
        while feasible == False and bnds != None:
            x_test = take_step.call(x, jumpMagnitude)
            feasible = check_feasibility(x_test, bnds)
            # if feasible == True:
            #     feasible = check_constraints(x, cons)

        
        # Local optimization 
        # solutionLocal = spy.minimize(f, x, method = 'COBYLA', constraints = cons, options = {'maxiter': niter_local} )
        if type(cons) == int:
            solutionLocal = spy.minimize(f, x_test, method = 'SLSQP', \
                tol = tolLocal, bounds = bnds, options = {'maxiter': niter_local} )
        else:
            solutionLocal = spy.minimize(f, x_test, method = 'SLSQP', \
                tol = tolLocal, bounds = bnds, options = {'maxiter': niter_local},\
                constraints = cons )
        currentMin = f( solutionLocal.x )
        feasible = check_feasibility(solutionLocal.x, bnds) 

        # if feasible == True: # jump from current point even if it is not optimum
        #     x = solutionLocal.x

        # Check te current point from which to jump: after doing a long jump or
        # when the solution is improved        
        if jumpMagnitude == 1 or currentMin < previousMin:
            x = solutionLocal.x
            
        # Check improvement      
        if currentMin < bestMin and feasible == True: # Improvement            
            Best = x
            bestMin = currentMin
            accepted = True

            # If the improvement is not large, assume it is not improvement
            if (previousMin - currentMin ) < tolGlobal: 
                n_noimprove += 1
            else:
                n_noimprove = 0
            jumpMagnitude = jumpMagnitude_default

        elif n_noimprove == niter_success: # Not much improvement
            accepted = False
            jumpMagnitude = 1
            n_noimprove = 0 # Restart count so that it performs jumps around a point
        else:
            accepted = False
            jumpMagnitude = jumpMagnitude_default
            n_noimprove += 1        

        previousMin = currentMin

        # Save results every 5 iter
        if n_itercounter % 20 == 0:
            AL_BF.writeData(Best, 'w', 'SolutionMBH_self.txt')
        
        print("iter", n_itercounter)
        print("Current min vs best one", currentMin, bestMin)
        print_fun(f, x, accepted)

    # Print solution 
    # print_sol(Best, bestMin, n_itercounter, niter_success)
    return Best, bestMin
        
