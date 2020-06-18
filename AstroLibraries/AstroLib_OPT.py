#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import time

# # Monotonic Basin hopping 
# Configuration for scipy.optimize.basinhopping()

# In[2]:


# Function to change step in basin hoping
class MyTakeStep(object):
    def __init__(self, stepsize):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        x[0] += np.random.normal(0, 0.1e4, 1)[0]
        x[1] += np.random.normal(0, 1e3, 2)[0]
        x[2:2+Nimp] += np.random.normal(0, 50, np.shape(x[5:5+Nimp]))
        x[2+Nimp:] += np.random.normal(0, 0.3, np.shape(x[5+Nimp:]))

        
#         for i in range(len(x)):
#             x[i] += x[i]*p[i]
#         print('aft',x)
        return x

mytakestep = MyTakeStep(0.5)


# In[3]:


def print_fun(x, f, accepted):
    """
    print_fun: choose what to print after each iteration
    INPUTS: 
        x: current value of the decision vector
        f: function value
        accepted: iteration improves the function = TRUE. Otherwise FALSE.
    OUTPUTS: none
    """
    print("#####################################################################")
    print("Change iteration",accepted)
    print('t',time.time() - start_time)
#     print(x)
#     print("#####################################################################")


# # Initial problem investigation

# # Evolutionary algorithm
# 
# Random initialization--> fitness calculation--> offspring creation --> immigration, mutation --> convergence criterion

# In[6]:


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
    """
    x_add = kwargs.get('x_add', None)
    ind = kwargs.get('ind', 100)
    cuts = kwargs.get('cuts', 1)
    tol = kwargs.get('tol', 1e-4)
    max_iter = kwargs.get('max_iter', 1e3)
    max_iter_success = kwargs.get('max_iter_success', 1e2)
    elitism = kwargs.get('elitism',0.1)
    mut = kwargs.get('mutation',0.01)
    
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
            aaa = f(pop_0[i,1:])
            pop_0[i,0] = f(pop_0[i,1:])
    else:
        for i in range(ind):
            pop_0[i,0] = f(pop_0[i,1:], x_add)
    
    Sol = pop_0[pop_0[:,0].argsort()]
    minVal = min(Sol[:,0])
    x_minVal = Sol[0,:]
    
    ###############################################
    ###### NEXT GENERATION                  #######
    ###############################################
    noImprove = 0
    counter = 0
    lastMin = minVal
    
    Best = np.zeros([100,len(bounds)+1])
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
        np.random.shuffle(children[ind_elit:,:]) #shuffle the others
        

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
        if x_add == False: # No additional arguments needed
            for i in range(ind):
                pop[i,0] = f(pop[i,1:])
        else:
            for i in range(ind):
                pop[i,0] = f(pop[i,1:], x_add)

        Sol = pop[pop[:,0].argsort()]
        minVal = min(Sol[:,0])

        ###############################################
        #Check convergence
        counter += 1 #Count generations 
        
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
        
    print("minimum:", lastMin)
    print("Iterations:", counter)
    print("Iterations with no improvement:", noImprove)
    
    return x_minVal, Best


# In[7]:


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


# In[8]:


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


# In[14]:


# def f(x,y):
#     return x[1]**2+x[2]**2+x[0]**2

# x = np.array([1.,1.,1])

# stepSize = np.array([0.1,0.1,0.1])
# alpha = np.copy(stepSize)
# SteepestDescentDirection(f,x,stepSize,alpha,np.array([1,1,1]))

