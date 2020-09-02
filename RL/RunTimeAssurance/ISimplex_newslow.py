#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mark Mote (marklmote@gmail.com)

A template for creating RTA filters, passes desired input through unaltered

The main loop will call the RTA's "main" class and expect
a control signal of appropriate size and type to be returned

In order for this script to work correctly with the main script, avoid
changing the name of the "RTA" class, or the inputs and outputs of
the "main" function.

"""

import numpy as np

class RTA():
    def __init__(self, env):
        self.mass_chaser = env.mass_deputy
        self.mean_motion = env.n
        self.max_available_thrust = env.force_magnitude
        self.controller_sample_period = env.tau
        self.controller_sample_rate = 1 / self.controller_sample_period
        self.filter_sample_period = env.tau
        self.filter_sample_rate = 1 / self.filter_sample_period
        
        self.zero_input = np.zeros([3,1])

        # Define Backup controller parameters 
        self.T_backup = 750 #  2100 # [s] length of time in backup trajectory horizon 
        self.Nsteps = 500 #    # number steps in horizon of backup trajectory 
        self.kappa = .6     # higher values of this make actuation stonger  
        
        # Define safety set and backup set parameters 
        self.eta_b = .1   # acceptable error magnitude from NMT plane for reachability constraint
        self.K1_s = 2.0*self.mean_motion # slope of safety boundary for speed limit constraint (must be >= 2n)
        self.K2_s = 0.97*0.2  # 2*self.eta_b # max allowable speed at origin (must be > eta_b)
        
        # # Other 
        # self.dt_first_step = 1
        
        self.timevec = np.linspace(0, self.T_backup, self.Nsteps)
        self.dt = self.timevec[1]-self.timevec[0]
        
        # Define in-plane CWH Dynamics 
        self.A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [3*self.mean_motion**2, 0, 0, 2*self.mean_motion],
                  [0, 0, -2*self.mean_motion, 0]])
        self.B = np.array([[0, 0], 
                           [0, 0],
                           [1/self.mass_chaser, 0],
                           [0, 1/self.mass_chaser]])
        
        self.phi  = np.zeros([4, self.Nsteps])


    ##########################################################################    
    def main(self, x0, u_des):
        """
        Parameters
        ----------
        x : numpy array with 6 elements 
            x = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
            where x,y,z are hill frame coordinates 
        u: 3x1 numpy array
        u_des = [[Fx], [Fy], [Fz]]
            desired control input 

        Returns
        -------
        u : 3x1 numpy array
            u = [[Fx], [Fy], [Fz]]
            returned control input  
        """
        
        # Cut out "z" dimension in x0 
        x0 = np.array([x0[0], x0[1], x0[3], x0[4]])
        
        # Take one step under "u_des" 
        x1 = x0 + self.xdot(x0, u_des[0:2,0])*self.dt
        
        # Integrate trajectory from x1
        self.integrate(x1)
        
        udes_safe = True 
        
        # Backwards Reachability 
        if self.h_b(self.phi[:,-1]) < 0: 
            udes_safe = False 
            
        # Invariance
        for i in range(0,self.Nsteps):     
            if self.h_s(self.phi[:,i]) < 0 :
                udes_safe = False 
                break 
            
        if udes_safe:
            return u_des
        else: 
            ub = self.u_b(x0)
            u = np.array([ [ub[0,0]], [ub[1,0]], [u_des[2,0]] ])
            return u
      
    ########################################################################    
    def h_s(self, x):
        """
        h_s(x) >= 0 defines the set of all "safe states". The goal of the ASIF 
        is to ensure that this constraint remains satisfied 
        
        """
        r2 = x[0]**2 + x[1]**2
        return self.K2_s**2 + (self.K1_s**2)*(r2) + self.K1_s*self.K2_s*np.sqrt(r2) - (x[2]**2 + x[3]**2) 

    # ##########################################################################        
    def grad_hs(self, x):
        ghs = np.matmul( self.Hs, x )
        rn1 = 1/np.sqrt(x[0]**2 + x[1]**2) # 1/r 
        ghs[0] = ghs[0] + 2*self.K1_s*self.K2_s*rn1*x[0] 
        ghs[1] = ghs[1] + 2*self.K1_s*self.K2_s*rn1*x[1] 
        # print(ghs)
        return ghs 

    ##########################################################################  
    def h_b(self, x):
        """
        h_b(x) >= 0 defines the "terminal set".
        
        """
        sx = x[0]
        sy = x[1]
        vx = x[2]
        vy = x[3]
        n = self.mean_motion
        
        return self.eta_b**2 - vx**2 - vy**2 - 4*(n**2)*sx**2 - 0.25*(n**2)*sy**2 - 4*n*sx*vy + n*sy*vx 
    
    ##########################################################################        
    def grad_hb(self, x):
        return np.matmul( self.Hb, x )
    
    ##########################################################################            
    def u_b(self, x0):
                
        e1 = x0[2]-(self.mean_motion/2)*x0[1]
        e2 = x0[3]+(2*self.mean_motion)*x0[0]
        
        umax = self.max_available_thrust
        
        u = np.array([ [umax*np.tanh((-1.5*self.mean_motion*e2 - self.kappa*e1)/umax)],
                       [umax*np.tanh((-self.kappa*e2)/umax)]])
        
        return u
        
    # ##########################################################################         
    def xdot(self, x, u): 
        return (np.matmul(self.A, x.reshape(4,1)) + np.matmul(self.B, u.reshape(2,1))).reshape(4)
    
    ##########################################################################    
    def integrate(self, x0):
        # integrate backup controller to get phi, Dphi 
        
        self.phi[:,0] = x0 
        
        for i in range(1,self.Nsteps): 
            # Dynamics 
            self.phi[:,i] = self.phi[:,i-1] + self.xdot(self.phi[:,i-1], self.u_b(self.phi[:,i-1]))*self.dt 
        
    # ##########################################################################

        
     