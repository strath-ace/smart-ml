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

import GP.GPBased_ControlSchemes.GANNIC.FESTIP_reentry.surrogate_models as surr_mods
from scipy.interpolate import PchipInterpolator

def check_uncertainty(mode, uncertain, uncertain_atmo, uncertain_aero, uncertFuns,obj, h, v, ubd, ubc, M, alfa):
    if mode == 'atmo':
        if uncertain and uncertain_atmo:
            r = uncertFuns[1](h)
            eps_dens = surr_mods.atm_uncertainty(h, obj.hc, obj.lbd, ubd)
            rho = surr_mods.rho_surrogate_uncertain(h, r, eps_dens)
            eps_c = surr_mods.atm_uncertainty(h, obj.hc, obj.lbc, ubc)
            c = surr_mods.c_surrogate_uncertain(h, r, eps_c)
        else:
            rho = surr_mods.rho_surrogate(h)
            c = surr_mods.c_surrogate(h)
        return rho, c

    else:
        if uncertain and uncertain_aero:
            r_a = (uncertFuns[4](v) + uncertFuns[5](h)) / 2
            eps_a = surr_mods.aero_uncertainty(h, M, alfa, obj.hc, obj.Mc, obj.lba, obj.uba)
            cL = surr_mods.cl_surrogate_uncertain(M, alfa, r_a, eps_a)
            cD = surr_mods.cd_surrogate_uncertain(M, alfa, r_a, eps_a)
        else:
            cL = surr_mods.cl_surrogate(M, alfa)
            cD = surr_mods.cd_surrogate(M, alfa)
        return cL, cD


def select_uncertainty_profile(v_points, h_points, unc_profiles, l):

    uncert_funDv = PchipInterpolator(v_points, unc_profiles[l][0][:, 0])
    uncert_funDh = PchipInterpolator(h_points, unc_profiles[l][0][:, 1])
    uncert_funCv = PchipInterpolator(v_points, unc_profiles[l][1][:, 0])
    uncert_funCh = PchipInterpolator(h_points, unc_profiles[l][1][:, 1])
    uncert_funAv = PchipInterpolator(v_points, unc_profiles[l][2][:, 0])
    uncert_funAh = PchipInterpolator(h_points, unc_profiles[l][2][:, 1])
    uncertFuns = [uncert_funDv, uncert_funDh, uncert_funCv, uncert_funCh, uncert_funAv, uncert_funAh]

    return uncertFuns