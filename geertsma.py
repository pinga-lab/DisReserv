# coding: utf-8

"""
Forward modelling of elastic reservoir deformation and stresses produced by a disk-shaped reservoir, using Geertsma (1973a, b) methodology. The equations are valid OUTSIDE  de reservoir.
Implementation from "Petroleum Related Rock Mechanics (2008)", Appendix D-5.
"""

import numpy as np
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc

def geertsma(coordinates, model, DP, cm, poisson, R, h):
    '''Calculus of displacement using Geertsma methodology for a disk-shaped model. Implementation 
    based on Equations in Appendix D.5 from Petroleum Related Rock Mechanics (2008).'''

    Np = coordinates.shape[1] # number of computation points
    ur = np.zeros(Np) # radial component of the displacement
    uz = np.zeros(Np) # vertical component of the displacement

#     aux = 1e-6
    
    for obs in range(Np):
        #Observation points
        yp = coordinates[0,obs]
        xp = coordinates[1,obs]
        zp = coordinates[2,obs]
        
        # avoiding division by zero
#         if np.abs(xp) < aux and np.abs(yp) < aux:
#                 xp = aux
#                 yp = aux
        
        for mod in range(len(model)):
            # center of the prisms
            yc = (model[mod,1] + model[mod,0])/2.
            xc = (model[mod,3] + model[mod,2])/2.
            zc = (model[mod,4] + model[mod,5])/2.
            
            r = np.sqrt((yp-yc)**2 + (xp-xc)**2 + (zp-zc)**2)
            
            # avoiding division by zero
#             if r < aux:
#                 r = aux
            
            #radial displacement
            ur[obs] -= DP[mod]*(Int1(np.abs(zp-zc),r,R) + (3-4*poisson)*Int1(zp+zc,r,R) - 2*zp*Int2(zp+zc,r,R))
            
            #vertical displacement
            uz[obs] += DP[mod]*((zp-zc)*Int3(np.abs(zp-zc),r,R)/np.abs(zp-zc) - (3-4*poisson)*Int3(
                zp+zc,r,R) - 2*zp*Int4(zp+zc,r,R))
            
    ur *= cm*R*h/2.
    uz *= cm*R*h/2.
    
    return ur, uz


def geertsma_stress(coordinates, model, DP, cm, poisson, R, h, G):
    '''Calculus of stress using Geertsma methodology for a disk-shaped model. Implementation 
    based on Equations in Appendix D.5 from Petroleum Related Rock Mechanics (2008).'''

    Np = coordinates.shape[1] #number of computation points
    sr = np.zeros(Np) # radial component of the stress
    sz = np.zeros(Np) # vertical component of the stress
    st = np.zeros(Np) # tangential component of the stress

    for obs in range(Np):
        #Observation points
        yp = coordinates[0,obs]
        xp = coordinates[1,obs]
        zp = coordinates[2,obs]
                
        for mod in range(len(model)):
            # center of the prisms
            yc = (model[mod,1] + model[mod,0])/2.
            xc = (model[mod,3] + model[mod,2])/2.
            zc = (model[mod,4] + model[mod,5])/2.
            
            r = np.sqrt((yp-yc)**2 + (xp-xc)**2 + (zp-zc)**2)
                  
            #radial stress
            sr[obs] += DP[mod]*(Int4(np.abs(zp-zc),r,R) + 3*Int4(zp+zc,r,R) - 2*zp*Int6(zp+zc,r,R) - (
                Int1(np.abs(zp-zc),r,R) + (3-4*poisson)*Int1(zp+zc,r,R) - 2*zp*Int2(zp+zc,r,R))/r)
            
            #tangential stress
            st[obs] += DP[mod]*(4*poisson*Int4(zp+zc,r,R) + (Int1(np.abs(zp-zc),r,R) + (3-4*poisson)*Int1(
                zp+zc,r,R) - 2*zp*Int2(zp+zc,r,R))/r)
            
            #vertical stress
            sz[obs] -= DP[mod]*(-Int4(np.abs(zp-zc),r,R) + Int4(zp+zc,r,R) + 2*zp*Int6(zp+zc,r,R))
                  
    sr *= G*cm*R*h
    st *= G*cm*R*h
    sz *= G*cm*R*h
    
    return sr, st, sz


def Int1(q, r , R):
    m = 4*R*r/(q**2 + (r+R)**2)
    
    K = ellipk(m) #Complete elliptic integral of the first kind
    E0 = ellipe(m) #Complete elliptic integral of the second kind
    I1 = 2*((1-(m/2))*K - E0)/(np.pi*np.sqrt(m*r*R))
    
    return I1


def Int2(q, r , R):
    m = 4*R*r/(q**2 + (r+R)**2)
    
    K = ellipk(m) #Complete elliptic integral of the first kind
    E0 = ellipe(m) #Complete elliptic integral of the second kind
    I2 = q*np.sqrt(m)*((1-m/2)*E0/(1-m) - K)/(2*np.pi*np.sqrt(r*R)**3)
    
    return I2


def Int3(q, r , R):
    m = 4*R*r/(q**2 + (r+R)**2)
    
    K0 = ellipk(m) #Complete elliptic integral of the first kind
    K1 = ellipk(1-m)
    E0 = ellipe(m) #Complete elliptic integral of the second kind
    E1 = ellipe(1-m)
    
    beta = np.arcsin(q/np.sqrt(q**2 + (R-r)**2))
    K2 = ellipkinc(beta,1-m) #Incomplete elliptic integral of the first kind
    E2 = ellipeinc(beta,1-m) #Incomplete elliptic integral of the second kind
    
    Z = E2-E1*K2/K1  #Jacobi zeta function
    
    lamb = K2/K1 +2*K0*Z/np.pi #Heuman’s lambda function
    
    I3 = -q*np.sqrt(m)*K0/(2*np.pi*R*np.sqrt(r*R)) + (np.heaviside(r-R, 0.5)-np.heaviside(R-r, 0.5))*lamb/(
        2*R) + np.heaviside(R-r, 0.5)/R
    
    return I3


def Int4(q, r , R):
    m = 4*R*r/(q**2 + (r+R)**2)
    
    K0 = ellipk(m) #Complete elliptic integral of the first kind
    E0 = ellipe(m) #Complete elliptic integral of the second kind
    
    I4 = np.sqrt(m)**3*(R**2-r**2-q**2)*E0/(8*np.pi*np.sqrt(r*R)**3*R*(1-m)) + np.sqrt(m)*K0/(
        2*np.pi*R*np.sqrt(r*R))
    
    return I4


def Int6(q, r , R):
    m = 4*R*r/(q**2 + (r+R)**2)
    
    K0 = ellipk(m) #Complete elliptic integral of the first kind
    E0 = ellipe(m) #Complete elliptic integral of the second kind
    
    I6 = q*np.sqrt(m)**3*(3*E0 + m*(R**2-r**2-q**2)*((1-m/2)*E0/(1-m) - K0/4)/(r*R))/(
        8*np.pi*np.sqrt(r*R)**3*R*(1-m)) 
    return I6