from . import attitude_utils as attu
from . import env_utils as envu 
import numpy as np
from time import time

class Dynamics_model(object):

    """
        The dynamics model take a lander model object (and later an obstacle object) and modifies  
        the state of the lander.

        The lander object instantiates an engine model, that maps body frame thrust and torque to
        the inertial frame.  Note that each lander can have its own intertial frame which can be 
        centered on the lander's target. 

        Currentlly this model does not model environmental dynamics, will be added later
 
        The lander model maintains a state vector: 
            position                                [0:3]
            velocity                                [3:6]
            body frame rotational velocity (w_bn)   [6:9]
            mass                                    [9]     
            attitude in target frame                [10:]  (size depends on attitude parameterization)
                 

    """

    def __init__(self, h=5e-2, noise_u=np.zeros(3), noise_sd=np.zeros(3), l_offset=np.zeros(3), adjust_inertia_tensor=True):
        self.h = h 
        self.Isp = 210.0
        self.g_o = 9.81
        self.g = np.asarray([0.0,0.0,-3.7114])
        self.noise_sd = noise_sd
        self.noise_u =  noise_u
        self.l_offset = l_offset
        self.adjust_inertia_tensor = adjust_inertia_tensor
 
        print('6dof dynamics model')

    def next(self,t,thrust_command,lander):
        #t0 = time()
        if self.adjust_inertia_tensor:
            J = lander.inertia_tensor * lander.state['mass'] / lander.nominal_mass
        else:
            J = lander.inertia_tensor
        w = lander.state['w']
        x = lander.get_state_dynamics()
        #
        # get force and torque in body frame
        # 


        F,L,mdot = lander.thruster_model.thrust(thrust_command)
        L += self.l_offset

        #
        # convert force to acceleration
        #

        acc_body_frame = F / lander.state['mass']

        #
        # Find acceleration to inertial frame
        # Since the attitude is BN (body with respect to inertial) the associated DCM 
        # is BN and maps from inertial to body, so we need to invert it (transpose)
        # to map pfrom body to inertial (I think)
        # 

        #noise = (self.noise_u + self.noise_sd * np.random.normal(size=3)) /  lander.state['mass']
        noise = (self.noise_u + np.clip(self.noise_sd * np.random.normal(size=3), 0, 3*self.noise_sd)) /  lander.state['mass']

        dcm_NB = lander.attitude_parameterization.get_body_to_inertial_DCM(lander.state['attitude'])
        acc_inertial_frame = dcm_NB.dot(acc_body_frame) 
        thrust = acc_inertial_frame * lander.state['mass']
        acc_inertial_frame += self.g + noise 
        
        #
        # Here we use the Euler rotational equations of motion to find wdot
        #

        Jinv = np.linalg.inv(J)
        w_tilde = attu.skew(w)
        wdot = -Jinv.dot(w_tilde).dot(J).dot(w) + Jinv.dot(L)
        #print('DEBUG: ',L,wdot)
        #
        # differential kinematic equation for derivative of attitude
        #
        # integrate w_bt (body frame lander rotation relative to target frame) to get 
        # lander attitude in target frame
        # w_bn is stored in lander (rotation in inertial frame, which is caused by thruster torque)
        # reward function will try to make  w_bt zero
        #

        w_bt = w
        qdot = lander.attitude_parameterization.qdot(lander.state['attitude'], w_bt)

        #
        # Use 4th order Runge Kutta to integrate equations of motion
        #

        ode = lambda t,x : self.eqom(t, x, acc_inertial_frame, qdot, wdot, mdot)
        x_next = envu.rk4(t, x, ode, self.h )
        attitude = x_next[10:]
        attitude = lander.attitude_parameterization.fix_attitude(attitude) # normalize quaternions
        assert np.all(attitude < np.pi + 1e-9)
        # integrate w_bt (lander_body to targeta to get lander attitude in target frame)
        # w_bn is stored in lander (rotation in inertial frame, which is caused by thruster torque)

        #print(thrust_command, w, x_next[6:9])
        lander.state['position'] = x_next[0:3]
        lander.state['velocity'] = x_next[3:6]
        lander.state['w']        = x_next[6:9]
        lander.state['mass']     = x_next[9]
        lander.state['attitude'] = attitude 
        lander.state['attitude_321'] = lander.attitude_parameterization.q2Euler321(attitude) 

        #if not  np.all(lander.state['attitude'] < 4):
        #    print(lander.state['attitude'] , lander.state['w'])
        #assert np.all(lander.state['attitude'] < 4)

        lander.state['thrust'] = thrust 
        lander.state['bf_thrust'] = F
        lander.state['torque'] = L

        _, t_go = lander.track_func(lander.state['position'],lander.state['velocity'])
        lander.state['t_go'] = t_go

        #print('DEBUG3: ',lander.state['w']) 
        return x_next

    
     
           
    def eqom(self,t, x, acc, qdot, wdot, mdot):

        r = x[0:3]
        v = x[3:6]
        w = x[6:9]

        rdot = v
        vdot = acc

        xdot = np.zeros(10+qdot.shape[0])
        xdot[0:3] = v
        xdot[3:6] = acc
        xdot[6:9] = wdot
        xdot[9] = mdot
        xdot[10:] = qdot

        return xdot
 
         
       
        
        
