#!/usr/bin/env python
# coding: utf-8

# # MVBC Alg

# ## Functions and Libraries
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import copy


# # Functions in MVBC
# 
# Note: 
# 
# - matmul is * in MATLAB
# - \* is .* in MATLAB

# Something wrong in the update_u function double check

# In[2]:


################ For testing purposes, use the following metrics: ################
# M = M[0]
# z
# u =  U[:,0]
# v = V[0]
# gamma = gamma_u


################################################################################

def update_u(M, z, u, v, gamma):
    # Given z, v, update u
    grad = np.matmul(((z.T*u*v[0]).T - M),v[0]) * z

    # tmp
    h1 = np.matmul (np.transpose (v[0]), v[0])
    h2 = np.matmul (z,h1)
    tmp = h2 * z

    # lip
    lip = np.sqrt (np.matmul (np.transpose (tmp), tmp))

    if (lip == 0):
        u = np.zeros ((len (u),1)).shape
    
    # u 
    output = []
    output = u-np.transpose ((1/(gamma*lip) * grad))

    return output


# In[3]:


################ For testing purposes, use the following metrics: ################

# iView = 1
# M = M[iView]
# z
# u = U[:, iView]
# v = V[iView]
# gamma = gamma_v
# s = sv[iView][0]

################################################################################

def update_v(M, z, u, v, gamma, s):
    # grad
    h1 = z.T*u
    h2 = h1.T * v[0].T    
    h3 = h2 - M
    p1 = (z.T*u).T
    grad = np.matmul (h3.T , p1)

    # tmp
    tmp = (z.T*u).T
    
    # lip
    lip = np.matmul(tmp.T, tmp)
    
    
    if (lip == 0):
        np.zeros ((len(v[0]),1))
        # return

    # v
    output = []
    output = v[0]-1/(lip*gamma)*grad
    output = map_python (output,s)
    
    return output
    
# In[]:

################ For testing purposes, use the following metrics: ################
    
    # M = M
    # z = z
    # u = U
    # v = V
    # gamma = gamma_z 
    # s = sz

################################################################################


def update_z (M,z,u,v,gamma,s):
    
    grad = np.zeros((len(z),1))
    tmp = np.zeros ((len(z),1))
    
    for i in range (len(M)):
            
            ## GRAD
            
            # z .* u(:,i) * v{i}' - M{i}
            h1 = (z.T * u[:,i])
            h2 = h1.T * (v[i][0]).T
            h3 = h2 - M[i] 
            
            # h3 * v{i} 
            h4 = np.matmul (h3 , v[i][0])
            h5 = (h4.T*u[:,i]).T 
            
            grad += h5
        
            ## TMP
            
            t1 = np.matmul (v[i][0].T,v[i][0])
            t2 = t1 * u[:,i]
            t3 = t2*u[:,i]
            
            tmp += t3.T
            
            
            
    lip = np.sqrt (np.matmul (tmp.T, tmp))
    
    if (lip == 0):
        z = np.zeros ((len (z),1))
        
    z = z - 1/(lip*gamma) * grad
    
    z = map_python (z,s)

    
    return (z)



# In[]:
    
################ OBJECTIVE FUNCTION ################

####################################################

    
def objective (M,z,U,V):
    obj = 0;
    for i in range (len(M)):
        
        # MAT
        h1 = U[:,i]*z.T
        h2 = (h1.T*V[i][0].T)        

        mat = M[i] - h2 
        
        # OBJ
        self_mat = np.matmul (mat.T,mat)
        
        obj += np.sum(np.diag (self_mat))
        
        
    return obj


# In[4]:


################ For testing purposes, use the following metrics: ################

# Map u to its closest that has <= s none zeros 
# u = U[:,0]
# s = sz1

################################################################################

def map_python(u, s):

    u_a = abs(u)
    h1 = np.argsort (u_a, axis =0)
    I = h1[::-1]
    v = u_a[I]
    
    output = []
    output = copy.copy(u)
    output[I[s:len(I)]]=0
    
    return output


# In[5]:


# NORMALIZATION

def normalization (x):
    return ((x-min(x))/(max(x)-min(x)))

# def objective (M,z,U,V):
    
#     obj = 0;
    
#     for i in range (len (M)):
#         mat =  M{i} - (U(:, i) .* z) * V{i}';            
#         obj = obj + trace(mat' * mat)
                          


# # Code

# In[6]:


# What is set-up in the main fct of the MATLAB file

path = '../'
filename = 'M.mat'
signal = loadmat(path+filename)
M = signal['M'][0]

'''
signal['M'][0] # M Var
signal['M'][0][0] # M Var First Cell
signal['M'][0][0][0] # M Var First Cell First Row
'''

sz1 = 9
sv1 = np.array([[5, 5, 5]]).T
ini_v1 = 1

sz2 = 7
sv2 = np.array([[5, 5, 5]]).T
ini_v2 = 1


# # mvlrrl0

# In[7]:

# DEBUG FCT
debug = 0


# Inside the mvlrrl0 function
sz = sz1
sv = sv1
iSeedV1 = ini_v1


# line 1-50

maxIter = 1e03
threshold = 1e-05

n = M[0].shape[0]
nView = len(M)

d = np.zeros((nView, 1))
z = np.ones((n,1)); # initialize z with all ones
U = np.ones((n, nView)); # initialize u with all ones
V = np.full([nView,1], None) 
# V = [None]*nView

gamma_z = 1.2;
gamma_u = 1.2;
gamma_v = 1.2;


# In[8]:


# line 51-60

for iView in range (nView):
    d[iView] = M[iView].shape [1]
    V[iView] = [np.ones ((int(d[iView]),1))]


# In[9]:


# line 62-68

# Arguments > 3
# Initialize v1 with all zeros except 1 for the seed feature

V[0][0][0:len(V[0][0])] = 0
V[0][0][0] = 1


# In[10]:


# line 70-74

# Did not do update v because we expect 3 arguments

U[:,0] = update_u(M[0], z, U[:,0], V[0], gamma_u)

tmp = map_python (U[:,0], sz)

z[tmp == 0] = 0


# line 76 - 83

if (nView > 1):
    for iView in range  (1,nView):
        V[iView] = [update_v (M[iView], z, U[:, iView], V[iView], gamma_v, sv[iView][0])]
        U[:, iView] = update_u(M[iView], z, U[:, iView], V[iView], gamma_u)

    # Update Z
    z = update_z (M,z,U,V,gamma_z,sz)     


# line 85 

obj = objective (M,z,U,V)


# line 90

for iTer in range (int (maxIter)):
    pre_obj =  copy.copy(obj)
    
    pre_z =  copy.copy(z)
    
    # Given z, update U and V
    for iView in range (nView):
        V[iView] = [update_v (M[iView], z, U[:, iView], V[iView], gamma_v, sv[iView][0])]
        U[:, iView] = update_u(M[iView], z, U[:, iView], V[iView], gamma_u)
        
    
    # Given U, V, update z
    z = update_z (M,z,U,V,gamma_z,sz)     
    
    # Calc the objective fct
    obj = objective (M,z,U,V)
    
    if (debug == 1):
        print ('Iteration: %d \n' %iTer)
        print ('Objective value %f \n' %obj)
        
    if (np.linalg.norm(pre_z-z) < threshold):
        break
    
    


    
    













































