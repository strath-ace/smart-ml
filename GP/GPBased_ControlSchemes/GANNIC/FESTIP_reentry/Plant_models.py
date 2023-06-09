# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2023 University of Strathclyde and Author ------
# ---------------- Author: Francesco Marchetti ------------------------
# ----------- e-mail: francesco.marchetti@strath.ac.uk ----------------

# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 3.0, as described below:

# This file is free software: you may copy, redistribute and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3.0 of the License, or (at your
# option) any later version.

# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import numpy as np
import surrogate_models as surr_mods
from deap import gp
from copy import copy
import scipy.io as sio
from scipy.interpolate import PchipInterpolator
import GP.GPBased_ControlSchemes.GANNIC.utils as utils
import GP.GPBased_ControlSchemes.GANNIC.utils_NN as utils_NN

class Spaceplane:
    """
    FESTIP FSS5 model
    """
    def __init__(self):

        # environment's parameters
        self.g0 = 9.80665  # m/s2
        self.Re = 6371000  # Earth Radius [m]

        # plant's parameters
        self.n_states = 6  # number of states
        self.n_controls = 2
        self.states_names = ['v', 'chi', 'gamma', 'teta', 'lam', 'h']
        self.controls_names = ['alfa', 'sigma']
        self.wingSurf = 500.0  # m2
        self.m = 45040  # kg  starting mass
        self.Cfin = 9.12e-4  # (kg ^ 0.5 * m ^ 1.5 * s ^ -3)
        self.Lam = np.deg2rad(45)  # deg
        self.MaxAz = 25
        self.Maxq = 40000  # Pa
        self.MaxQ = 4e6  # W/m^2

        # trajectory's parameters
        self.vmin = 1.0
        self.vmax = 5200
        self.chimin = np.deg2rad(-180)
        self.chimax = np.deg2rad(180)
        self.gammamin = np.deg2rad(-89)
        self.gammamax = np.deg2rad(89)
        self.tetamin = np.deg2rad(-87)
        self.tetamax = np.deg2rad(-80)
        self.lammin = np.deg2rad(27)
        self.lammax = np.deg2rad(30)
        self.hmin = 1.0
        self.hmax = 60000
        self.alfamin = np.deg2rad(-2)
        self.alfamax = np.deg2rad(40)
        self.sigmamin = np.deg2rad(-90)
        self.sigmamax = np.deg2rad(90)
        self.tetaend = np.deg2rad(-80.7112)
        self.tetaend_tol = np.deg2rad(0.0014)
        self.lamend = np.deg2rad(28.6439)
        self.lamend_tol = np.deg2rad(0.0014)
        self.hend = 609.60  # m
        self.hend_tol = 121.92  # m

        # simulation's parameters
        self.Npoints = 0 # number of propagation points
        self.t0 = 345 # initial time simulation
        self.tf = 0 # end time simulation, updated by initial conditions load function
        self.dt = 5 # time step for propagation
        self.t_points = 0 # time points in propagation
        self.change_steps = [0, 0, 0, 0, 0]
        self.n_train = 5
        self.n_test = 100
        self.traj_to_simulate = 100 # last 100 seconds

        # initial_conditions
        self.states_refs = {}
        self.controls_refs = {}
        self.states_refs_funs = {}
        self.controls_refs_funs = {}
        self.init_conds = []

        # NN settings
        self.n_hidden_layers = 1
        self.nodes_per_hidden_layer = [3]
        self.activation = 'tanh'
        self.NNoutput = 2
        self.NNseed = 42
        self.n_inputNN = 0

        # Other settings
        self.n_inputGP = 0
        self.n_inputTOT = 0

        # uncertainty's parameters
        self.lbc = 0.01
        self.initial_ubc = 0.2
        self.lbd = 0.01
        self.initial_ubd = 0.2
        self.lba = 0.01
        self.uba = 0.2
        self.ubd = 0.2
        self.ubc = 0.2
        self.hc = self.hmax
        self.Mc = 10

        # scaler's parameters
        self.lb_contr = 0.0
        self.ub_contr = 100.0
        self.tanh_param_out = 2000
        self.tanh_param_in = 0.0015

        # GP fitness's parameters
        self.fit_weights = np.array(([1, 1, 1, 10, 10, 1]))



    def load_init_conds(self, filepath):
        """
        Function used to load the initial conditions

        Attributes:
            filepath: str
                path to save folder
        """
        ref_traj = sio.loadmat(filepath) # loaded from matlab, change if different
        tref = ref_traj['timetot'][0] # reference time points
        self.tf = tref[-1] # final time
        self.t0 = self.tf-self.traj_to_simulate
        self.Npoints = int((self.tf - self.t0) / self.dt) + 1 # number of propagation points
        self.t_points = np.linspace(self.t0, self.tf, self.Npoints) # time points

        indexes = np.where(np.diff(tref) == 0)
        tref = np.delete(tref, indexes)
        self.tref = tref
        for name in self.states_names:
            ref = copy(ref_traj['{}tot'.format(name)][0])
            updated_ref = np.delete(ref, indexes)
            self.states_refs['{}ref'.format(name)] = updated_ref
            self.states_refs_funs['{}fun'.format(name)] = PchipInterpolator(tref, updated_ref)
            self.init_conds.append(self.states_refs_funs['{}fun'.format(name)](self.t0))

        for name in self.controls_names:
            ref = copy(ref_traj['{}tot'.format(name)][0])
            updated_ref = np.delete(ref, indexes)
            self.controls_refs['{}ref'.format(name)] = updated_ref
            self.controls_refs_funs['{}fun'.format(name)] = PchipInterpolator(tref, updated_ref)




def forward_dynamicsNNUncert(t, states, obj, uncertain, uncertain_atmo, uncertain_aero, uncertFuns, NNmodel, GPind,
                             pset, n_weights, ubd, ubc, weight_index=None):
    """
    Function used to propagate the dynamics

    Attributes:
        t: float
            time
        states: array
            state variables
        obj: class
            plant's model
        uncertain: bool
            whether to use the uncertainty
        uncertain_atmo: bool
            whether to use uncertainty on atmospheric model
        uncertain_aero: bool
            whether to use uncertainty on aerodynamic model
        uncertFuns: array
            interpolating function for the uncertainties
        NNmodel: tensorflow model
            NN model
        GPind: list
            GP individual
        pset: class
            primitive set
        n_weights: int
            number of weights in NN model
        ubd: float
            upper bound on uncertainty applied to rho
        ubc: float
            upper bound on uncertainty applied to c
        weight_index: int
            index of NN weight

    Return:
        dx: array
            propagated values of state variables
    """
    states = np.nan_to_num(states)

    v = states[0]
    chi = states[1]
    gamma = states[2]
    lam = states[4]
    h = states[5]
    NNweights = states[6:]

    sts = states[:obj.n_states]
    refs = np.array(([obj.states_refs_funs['vfun'](t), obj.states_refs_funs['chifun'](t),
                      obj.states_refs_funs['gammafun'](t), obj.states_refs_funs['tetafun'](t),
                      obj.states_refs_funs['lamfun'](t), obj.states_refs_funs['hfun'](t)]))
    LB = np.array(([obj.vmin, obj.chimin, obj.gammamin, obj.tetamin, obj.lammin, obj.hmin]))
    UB = np.array(([obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax]))
    norm_refs = (refs - LB) / (UB - LB)
    norm_states = (sts - LB) / (UB - LB)

    all_errs = norm_states - norm_refs

    errs = copy(all_errs[:3])

    rho, c = utils.check_uncertainty('atmo', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj, h, v, ubd,
                                     ubc, 0, 0)
    if NNmodel:
        NNmodel = utils_NN.update_NN_weights(NNmodel, NNweights, n_weights)
        inputNN = np.array([[*errs]])
        NNoutput = np.ravel(NNmodel(inputNN).numpy())
        NNoutput = np.nan_to_num(NNoutput)
        eps_cont = surr_mods.control_variation(abs(all_errs), obj.lb_contr, obj.ub_contr)
        alfa = obj.controls_refs_funs['alfafun'](t) * (1 - eps_cont + (eps_cont / obj.tanh_param_out) * (
                    obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[0]) + obj.tanh_param_out))
        sigma = obj.controls_refs_funs['sigmafun'](t) * (1 - eps_cont + (eps_cont / obj.tanh_param_out) * (
                    obj.tanh_param_out * np.tanh(obj.tanh_param_in * NNoutput[1]) + obj.tanh_param_out))

    else:
        alfa = obj.controls_refs_funs['alfafun'](t)
        sigma = obj.controls_refs_funs['sigmafun'](t)


    if sigma > obj.sigmamax or np.isinf(sigma):
        sigma = obj.sigmamax
    elif sigma < obj.sigmamin or np.isnan(sigma):
        sigma = obj.sigmamin
    if alfa > obj.alfamax or np.isinf(alfa):
        alfa = obj.alfamax
    elif alfa < obj.alfamin or np.isnan(alfa):
        alfa = obj.alfamin

    if np.isnan(v):
        M = 0
    elif np.isinf(v):
        M = 1e6 / c
    else:
        M = v / c

    cL, cD = utils.check_uncertainty('aero', uncertain, uncertain_atmo, uncertain_aero, uncertFuns, obj, h, v, ubd,
                                     ubc, M, alfa)

    L = 0.5 * (v ** 2) * obj.wingSurf * rho * cL
    D = 0.5 * (v ** 2) * obj.wingSurf * rho * cD

    g0 = obj.g0

    if h <= 0 or np.isnan(h):
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2

    dx = dynamicsONLY(v, chi, gamma, lam, h, D, L, g, obj, sigma)

    if NNmodel:
        for i in range(len(GPind)):

            if weight_index is not None:
                if i == weight_index:
                    weight_new = 0
                else:
                    compiled_ind = gp.compile(GPind[i], pset=pset)
                    weight_new = compiled_ind(*norm_states, rho, NNweights[i])
            else:
                compiled_ind = gp.compile(GPind[i], pset=pset)
                weight_new = compiled_ind(*norm_states, rho, NNweights[i])
            #if i in [1,16]:   # uncomment to test the absence of multiple NN weights at the same time   # remove comment when using multiple weight_idxs
            #    weight_new = 0
            #else:
            #    compiled_ind = gp.compile(GPind[i], pset=pset)
            #    weight_new = compiled_ind(*norm_states, rho, NNweights[i])

            dx = np.hstack((dx, weight_new))


    del NNmodel, GPind, states, obj, uncertain, uncertain_atmo, uncertain_aero, uncertFuns, pset, sts, refs, LB, UB, \
        norm_refs, norm_states, alfa, sigma, L, D, g, cL, cD, M, rho, c, v, chi, gamma, lam, h
    return dx



def dynamicsONLY(v, chi, gamma, lam, h, D, L, g, obj, sigma):
    """
    Function containing equations of motion

    Attributes:
        v: float
            speed
        chi: float
            heading angle
        gamma: float
            flight path angle
        lam: float
            latitude
        h: float
            altitude
        D: float
            drag
        L: float
            lift
        g: float
            gravitational acceleration
        obj: class
            plant model
        sigma: float
            bank angle
    Return:
        dx: array
            derivative of state variables
    """
    dx = np.array((- D / obj.m - g * np.sin(gamma),
                   (L * np.sin(sigma)) / (obj.m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)),
                   (L * np.cos(sigma)) / (obj.m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma),
                   np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma)))
    return dx







