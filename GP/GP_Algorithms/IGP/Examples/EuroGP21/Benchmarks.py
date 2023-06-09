# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2021 University of Strathclyde and Author ------
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

import sys
import os
data_path = os.path.join(os.path.dirname( __file__ ), 'Datasets')
gpfun_path = os.path.join(os.path.dirname( __file__ ), '..', '..')
sys.path.append(gpfun_path)
sys.path.append(data_path)
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def select_testcase(bench):
    if bench == "koza1":
        f = lambda x: x**4 + x**3 + x**2 + x
        input_true = np.random.uniform(-1, 1, 20)
        input_test = np.random.uniform(-1, 1, 100)
        output_true = f(input_true)
        output_test = f(input_test)
    elif bench == "korns11":
        f = lambda x, y, z, v, w: 6.87 + 11*np.cos(7.23*x**3)
        input_true1 = np.random.uniform(-50, 10, 5000)
        input_true2 = np.random.uniform(-50, 10, 5000)
        input_true3 = np.random.uniform(-50, 10, 5000)
        input_true4 = np.random.uniform(-50, 10, 5000)
        input_true5 = np.random.uniform(-50, 10, 5000)
        input_true = np.vstack((input_true1, input_true2, input_true3, input_true4, input_true5))
        input_test1 = np.random.uniform(-50, 10, 5000)
        input_test2 = np.random.uniform(-50, 10, 5000)
        input_test3 = np.random.uniform(-50, 10, 5000)
        input_test4 = np.random.uniform(-50, 10, 5000)
        input_test5 = np.random.uniform(-50, 10, 5000)
        input_test = np.vstack((input_test1, input_test2, input_test3, input_test4, input_test5))
        output_true = f(input_true1, input_true2, input_true3, input_true4, input_true5)
        output_test = f(input_test1, input_test2, input_test3, input_test4, input_test5)
    elif bench == "S1":
        f = lambda x: np.exp(-x) * x**3 * np.sin(x) * np.cos(x) * (np.sin(x)**2*np.cos(x)-1)*(np.sin(x)**2*np.cos(x)-1)
        input_true = np.linspace(-0.5, 10.5, 111)
        input_test = np.linspace(-0.5, 10.5, 222)
        output_true = f(input_true)
        output_test = f(input_test)
    elif bench == "S2":
        f = lambda x, y: (y-5)*(np.exp(-x) * x**3 * np.sin(x) * np.cos(x) * (np.sin(x)**2*np.cos(x)-1)*(np.sin(x)**2*np.cos(x)-1))
        input_true1 = np.linspace(-0.5, 10.5, 111)
        input_true2 = np.linspace(-0.5, 10.5, 111)
        input_test1 = np.linspace(-0.5, 10.5, 222)
        input_test2 = np.linspace(-0.5, 10.5, 222)
        input_true = np.vstack((input_true1, input_true2))
        input_test = np.vstack((input_test1, input_test2))
        output_true = f(input_true1, input_true2)
        output_test = f(input_test1, input_test2)
    elif bench == "UB":
        f = lambda x1, x2, x3, x4, x5: 10/(5+(x1-3)**2+(x2-3)**2+(x3-3)**2+(x4-3)**2+(x5-3)**2)
        input_true1 = np.random.uniform(-0.25, 6.35, 1024)
        input_true2 = np.random.uniform(-0.25, 6.35, 1024)
        input_true3 = np.random.uniform(-0.25, 6.35, 1024)
        input_true4 = np.random.uniform(-0.25, 6.35, 1024)
        input_true5 = np.random.uniform(-0.25, 6.35, 1024)
        input_test1 = np.random.uniform(-0.25, 6.35, 5000)
        input_test2 = np.random.uniform(-0.25, 6.35, 5000)
        input_test3 = np.random.uniform(-0.25, 6.35, 5000)
        input_test4 = np.random.uniform(-0.25, 6.35, 5000)
        input_test5 = np.random.uniform(-0.25, 6.35, 5000)
        input_true = np.vstack((input_true1, input_true2, input_true3, input_true4, input_true5))
        input_test = np.vstack((input_test1, input_test2, input_test3, input_test4, input_test5))
        output_true = f(input_true1, input_true2, input_true3, input_true4, input_true5)
        output_test = f(input_test1, input_test2, input_test3, input_test4, input_test5)
    elif bench == "ENC":
        data = pd.read_excel(r'Datasets/ENB2012_data.xlsx')
        data = shuffle(data)
        input_true1 = pd.DataFrame(data, columns=['X1'])['X1']
        input_true2 = pd.DataFrame(data, columns=['X2'])['X2']
        input_true3 = pd.DataFrame(data, columns=['X3'])['X3']
        input_true4 = pd.DataFrame(data, columns=['X4'])['X4']
        input_true5 = pd.DataFrame(data, columns=['X5'])['X5']
        input_true6 = pd.DataFrame(data, columns=['X6'])['X6']
        input_true7 = pd.DataFrame(data, columns=['X7'])['X7']
        input_true8 = pd.DataFrame(data, columns=['X8'])['X8']
        input_test1 = input_true1[int(len(input_true1)*0.7):].values
        input_true1 = input_true1[:int(len(input_true1)*0.7)].values
        input_test2 = input_true2[int(len(input_true2) * 0.7):].values
        input_true2 = input_true2[:int(len(input_true2) * 0.7)].values
        input_test3 = input_true3[int(len(input_true3) * 0.7):].values
        input_true3 = input_true3[:int(len(input_true3) * 0.7)].values
        input_test4 = input_true4[int(len(input_true4) * 0.7):].values
        input_true4 = input_true4[:int(len(input_true4) * 0.7)].values
        input_test5 = input_true5[int(len(input_true5) * 0.7):].values
        input_true5 = input_true5[:int(len(input_true5) * 0.7)].values
        input_test6 = input_true6[int(len(input_true6) * 0.7):].values
        input_true6 = input_true6[:int(len(input_true6) * 0.7)].values
        input_test7 = input_true7[int(len(input_true7) * 0.7):].values
        input_true7 = input_true7[:int(len(input_true7) * 0.7)].values
        input_test8 = input_true8[int(len(input_true8) * 0.7):].values
        input_true8 = input_true8[:int(len(input_true8) * 0.7)].values
        input_true = np.vstack((input_true1, input_true2, input_true3, input_true4, input_true5, input_true6, input_true7, input_true8))
        input_test = np.vstack((input_test1, input_test2, input_test3, input_test4, input_test5, input_test6, input_test7, input_test8))
        output_true = pd.DataFrame(data, columns=['Y2'])['Y2']
        output_test = output_true[int(len(output_true) * 0.7):].values
        output_true = output_true[:int(len(output_true) * 0.7)].values
        f = []
    elif bench == "ENH":
        data = pd.read_excel(r'Datasets/ENB2012_data.xlsx')
        data = shuffle(data)
        input_true1 = pd.DataFrame(data, columns=['X1'])['X1']
        input_true2 = pd.DataFrame(data, columns=['X2'])['X2']
        input_true3 = pd.DataFrame(data, columns=['X3'])['X3']
        input_true4 = pd.DataFrame(data, columns=['X4'])['X4']
        input_true5 = pd.DataFrame(data, columns=['X5'])['X5']
        input_true6 = pd.DataFrame(data, columns=['X6'])['X6']
        input_true7 = pd.DataFrame(data, columns=['X7'])['X7']
        input_true8 = pd.DataFrame(data, columns=['X8'])['X8']
        input_test1 = input_true1[int(len(input_true1) * 0.7):].values
        input_true1 = input_true1[:int(len(input_true1) * 0.7)].values
        input_test2 = input_true2[int(len(input_true2) * 0.7):].values
        input_true2 = input_true2[:int(len(input_true2) * 0.7)].values
        input_test3 = input_true3[int(len(input_true3) * 0.7):].values
        input_true3 = input_true3[:int(len(input_true3) * 0.7)].values
        input_test4 = input_true4[int(len(input_true4) * 0.7):].values
        input_true4 = input_true4[:int(len(input_true4) * 0.7)].values
        input_test5 = input_true5[int(len(input_true5) * 0.7):].values
        input_true5 = input_true5[:int(len(input_true5) * 0.7)].values
        input_test6 = input_true6[int(len(input_true6) * 0.7):].values
        input_true6 = input_true6[:int(len(input_true6) * 0.7)].values
        input_test7 = input_true7[int(len(input_true7) * 0.7):].values
        input_true7 = input_true7[:int(len(input_true7) * 0.7)].values
        input_test8 = input_true8[int(len(input_true8) * 0.7):].values
        input_true8 = input_true8[:int(len(input_true8) * 0.7)].values
        input_true = np.vstack((input_true1, input_true2, input_true3, input_true4, input_true5, input_true6, input_true7, input_true8))
        input_test = np.vstack((input_test1, input_test2, input_test3, input_test4, input_test5, input_test6, input_test7, input_test8))
        output_true = pd.DataFrame(data, columns=['Y1'])['Y1']
        output_test = output_true[int(len(output_true) * 0.7):].values
        output_true = output_true[:int(len(output_true) * 0.7)].values
        f = []
    elif bench == "CCS":
        data = pd.read_excel(r'Datasets/Concrete_Data.xls')
        data = shuffle(data)
        input_true1 = pd.DataFrame(data, columns=['Cement (component 1)(kg in a m^3 mixture)'])['Cement (component 1)(kg in a m^3 mixture)']
        input_true2 = pd.DataFrame(data, columns=['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']
        input_true3 = pd.DataFrame(data, columns=['Fly Ash (component 3)(kg in a m^3 mixture)'])['Fly Ash (component 3)(kg in a m^3 mixture)']
        input_true4 = pd.DataFrame(data, columns=['Water  (component 4)(kg in a m^3 mixture)'])['Water  (component 4)(kg in a m^3 mixture)']
        input_true5 = pd.DataFrame(data, columns=['Superplasticizer (component 5)(kg in a m^3 mixture)'])['Superplasticizer (component 5)(kg in a m^3 mixture)']
        input_true6 = pd.DataFrame(data, columns=['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']
        input_true7 = pd.DataFrame(data, columns=['Fine Aggregate (component 7)(kg in a m^3 mixture)'])['Fine Aggregate (component 7)(kg in a m^3 mixture)']
        input_true8 = pd.DataFrame(data, columns=['Age (day)'])['Age (day)']
        input_test1 = input_true1[int(len(input_true1) * 0.7):].values
        input_true1 = input_true1[:int(len(input_true1) * 0.7)].values
        input_test2 = input_true2[int(len(input_true2) * 0.7):].values
        input_true2 = input_true2[:int(len(input_true2) * 0.7)].values
        input_test3 = input_true3[int(len(input_true3) * 0.7):].values
        input_true3 = input_true3[:int(len(input_true3) * 0.7)].values
        input_test4 = input_true4[int(len(input_true4) * 0.7):].values
        input_true4 = input_true4[:int(len(input_true4) * 0.7)].values
        input_test5 = input_true5[int(len(input_true5) * 0.7):].values
        input_true5 = input_true5[:int(len(input_true5) * 0.7)].values
        input_test6 = input_true6[int(len(input_true6) * 0.7):].values
        input_true6 = input_true6[:int(len(input_true6) * 0.7)].values
        input_test7 = input_true7[int(len(input_true7) * 0.7):].values
        input_true7 = input_true7[:int(len(input_true7) * 0.7)].values
        input_test8 = input_true8[int(len(input_true8) * 0.7):].values
        input_true8 = input_true8[:int(len(input_true8) * 0.7)].values
        input_true = np.vstack((input_true1, input_true2, input_true3, input_true4, input_true5, input_true6, input_true7, input_true8))
        input_test = np.vstack((input_test1, input_test2, input_test3, input_test4, input_test5, input_test6, input_test7, input_test8))
        output_true = pd.DataFrame(data, columns=['Concrete compressive strength(MPa, megapascals) '])['Concrete compressive strength(MPa, megapascals) ']
        output_test = output_true[int(len(output_true) * 0.7):].values
        output_true = output_true[:int(len(output_true) * 0.7)].values
        f = []
    elif bench == "ASN":
        with open("Datasets/airfoil_self_noise.dat") as f:
            data = []
            for line in f:
                line = line.split()
                if line:
                    line = [float(i) for i in line]
                    data.append(line)
        f.close()
        data = np.asarray(data)
        data = pd.DataFrame(data=data, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
        data = shuffle(data)
        input_true1 = pd.DataFrame(data, columns=['x1'])['x1']
        input_true2 = pd.DataFrame(data, columns=['x2'])['x2']
        input_true3 = pd.DataFrame(data, columns=['x3'])['x3']
        input_true4 = pd.DataFrame(data, columns=['x4'])['x4']
        input_true5 = pd.DataFrame(data, columns=['x5'])['x5']
        input_test1 = input_true1[int(len(input_true1) * 0.7):].values
        input_true1 = input_true1[:int(len(input_true1) * 0.7)].values
        input_test2 = input_true2[int(len(input_true2) * 0.7):].values
        input_true2 = input_true2[:int(len(input_true2) * 0.7)].values
        input_test3 = input_true3[int(len(input_true3) * 0.7):].values
        input_true3 = input_true3[:int(len(input_true3) * 0.7)].values
        input_test4 = input_true4[int(len(input_true4) * 0.7):].values
        input_true4 = input_true4[:int(len(input_true4) * 0.7)].values
        input_test5 = input_true5[int(len(input_true5) * 0.7):].values
        input_true5 = input_true5[:int(len(input_true5) * 0.7)].values
        input_true = np.vstack((input_true1, input_true2, input_true3, input_true4, input_true5))
        input_test = np.vstack((input_test1, input_test2, input_test3, input_test4, input_test5))
        output_true = pd.DataFrame(data, columns=['y'])['y']
        output_test = output_true[int(len(output_true) * 0.7):].values
        output_true = output_true[:int(len(output_true) * 0.7)].values
        f = []
    return f, input_true, output_true, input_test, output_test

def out_terminals(bench):
    if bench == "koza1":
        terminals = 1
        npoints = 20
    elif bench == "korns11":
        terminals = 5
        npoints = 5000
    elif bench == "S1":
        terminals = 1
        npoints = 111
    elif bench == "S2":
        terminals = 2
        npoints = 111
    elif bench == "UB":
        terminals = 5
        npoints = 1024
    elif bench == "ENC" or bench == "ENH":
        terminals = 8
        npoints = 768
    elif bench == "CCS":
        terminals = 8
        npoints = 1030
    elif bench == "ASN":
        terminals = 5
        npoints = 1502
    return terminals, npoints
