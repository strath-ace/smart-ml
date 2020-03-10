import numpy as np

class Thruster_model(object):

    """
        Thruster model for spacecraft computes force and torque in the body frame and converts
        to inertial frame

        Commanded thrust is clipped to lie between zero and one, and then scaled based off of 
        thrust capability

        ellipsoid c = 1m, a = b = 2m
    """
    def __init__(self):
        #                   dvec                      body position
        config = [   
                      [ 0.0,  0.0,    1.0,    0.0,   -2.0,   -1.0 ],  # rotate around X (roll)  
                      [ 0.0,  0.0,    1.0,    0.0,    2.0,   -1.0 ],  # rotate around X (roll)
                      [ 0.0,  0.0,    1.0,   -2.0,    0.0,   -1.0 ],  # rotate around Y (pitch) 
                      [ 0.0,  0.0,    1.0,    2.0,    0.0,   -1.0 ]  # rotate around Y (pitch) 
                 ]
        # no yaw . note that yaw rotates around z-axis, which we don't want (or need) to do

        config = np.asarray(config)
        self.dvec = config[:,0:3]

        self.position = config[:,3:6]

        self.num_thrusters = self.position.shape[0]
 
        self.max_thrust = 4000.0
        self.min_thrust =  800.0 

        self.pulsed = False
   
        self.Isp = 210.0 
        self.g_o = 9.81
 
        self.eps = 1e-8

        self.mdot = None 
                
              
        print('Thruster Config Shape: ',config.shape, self.num_thrusters)
 
    def thrust(self,commanded_thrust):

        assert commanded_thrust.shape[0] == self.num_thrusters

        if self.pulsed:
            commanded_thrust = commanded_thrust > self.eps

        commanded_thrust = np.clip(commanded_thrust, 0.0, 1.0) * self.max_thrust
        commanded_thrust = np.clip(commanded_thrust, self.min_thrust, self.max_thrust) 
 
        force = np.expand_dims(commanded_thrust,axis=1) * self.dvec
 
        torque = np.cross(self.position, force)
        force = np.sum(force,axis=0)
        torque = np.sum(torque,axis=0)

        mdot = -np.sum(np.abs(commanded_thrust)) / (self.Isp * self.g_o)
        self.mdot = mdot # for rewards
        return force, torque, mdot

 
        
