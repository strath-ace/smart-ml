import numpy as np

class Glideslope_constraint(object):
    def __init__(self, delta=2, debug=False, gs_coeff=-1.0, gs_margin=0.1, gs_limit= 0.1, gs_tau=0.05, gs_penalty=-50.0, terminate_on_violation=True):
        self.delta = delta
        self.debug = debug

        self.gs_coeff = gs_coeff
        self.gs_margin = gs_margin
        self.gs_limit = gs_limit
        self.gs_tau = gs_tau
        self.gs_penalty = gs_penalty

        self.terminate_on_violation = terminate_on_violation

        self.gs = None

        print('queue fixed')
 
    def reset(self, state):
        pos = state['position']
        self.positions = []
        self.positions.append(pos.copy())
        self.gs = 100.

    def calculate(self, state):
        pos = state['position']
        self.positions.append(pos.copy())
        z = pos[2]
        pos_array = np.asarray(self.positions)
        z_array = pos_array[:,2] 
        indices = np.where( np.abs(z_array - z) > self.delta)[0]
        if indices.shape[0] > 0:        
            idx = indices[-1] 
            last_position = pos_array[idx]
            dz = last_position[2] - pos[2] 
            dy = last_position[1] - pos[1]
            dx = last_position[0] - pos[0]
            assert np.abs(dz) >= self.delta
            gs = np.abs(dz) / np.sqrt(dx**2 + dy**2)
            if self.debug:
                print('GS: ',gs, dx,dy,dz)
            self.gs = gs
        else:
            gs = 100.
        self.gs = gs
        return gs

    def get_reward(self):
        r_gs = 0.0
        gs_err = np.maximum(0.0, self.gs - self.gs_limit)
        if self.gs < self.gs_limit + self.gs_margin:
            r_gs = self.gs_coeff * np.exp(-gs_err / self.gs_tau)
        return r_gs

    def get_term_reward(self):
        r_gs = 0.0
        if self.gs - self.gs_limit < 0:
            r_gs = self.gs_penalty
        return r_gs

    def get_margin(self):
        return self.gs - self.gs_limit
    
    def get(self):
        return self.gs

#
#delta = 3
#for z in za:
#    idx = np.where(za - z > delta)[0]
#    if idx.shape[0] > 0:
#        foo = idx[-1]
#        print(z,foo,za[foo],za[foo]-z)
#         

