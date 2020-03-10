import numpy as np

class Flat_constraint(object):

    def  __init__(self, terminate_on_violation=True):
        self.terminate_on_violation = terminate_on_violation
        print('Flat Constraint')

    def get_margin(self,state,debug=False):
        return 100.0 

    def get_reward(self,state):
        return 0.0, 100.0 

    def get_term_reward(self,state):
        return 0.0 

    def get_apf(self,pos,vel):
        return np.zeros(3) 
        
