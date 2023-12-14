import casadi as ca
from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from zipfile import ZipFile
import os
import datetime
from .TimeOptBez import *
from sim.multirotor_ref_traj import f_ref
from lie.SE23 import *


class Bezier:
#https://en.wikipedia.org/wiki/B%C3%A9zier_curve

    def __init__(self, P: ca.SX, T: float):
        self.P = P
        self.m = P.shape[0]
        self.n = P.shape[1]-1
        self.T = T
    
    def eval(self, t):
        #https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
        beta = t/self.T
        A = ca.SX(self.P)
        for j in range(1, self.n + 1):
            for k in range(self.n + 1 - j):
                A[:, k] = A[:, k] * (1 - beta) + A[:, k + 1] * beta
        return A[:, 0]
    
    def deriv(self, m=1):
        D = ca.SX(self.P)
        for j in range(0, m):
            D = (self.n - j)*ca.horzcat(*[ D[:, i+1] - D[:, i] for i in range(self.n - j) ])
        return Bezier(D/self.T**m, self.T)

def derive_bezier6():
    n = 6
    T = ca.SX.sym('T')
    t = ca.SX.sym('t')
    P = ca.SX.sym('P', 1, n)
    B = Bezier(P, T)

    # derivatives
    B_d = B.deriv()
    B_d2 = B_d.deriv()
    B_d3 = B_d2.deriv()
    B_d4 = B_d3.deriv()

    # boundary conditions

    # trajectory
    p = B.eval(t)
    v = B_d.eval(t)
    a = B_d2.eval(t)
    j = B_d3.eval(t)
    s = B_d4.eval(t)
    r = ca.vertcat(p, v, a, j, s)

    # given position/velocity boundary conditions, solve for bezier points
    wp_0 = ca.SX.sym('p0', 2, 1)  # pos/vel at waypoint 0
    wp_1 = ca.SX.sym('p1', 2, 1)  # pos/vel at waypoint 1

    constraints = []
    constraints += [(B.eval(0), wp_0[0])]  # pos @ wp0
    constraints += [(B_d.eval(0), wp_0[1])]  # vel @ wp0
    constraints += [(B_d2.eval(0), 0)]  # zero accel @ wp0
    constraints += [(B.eval(T), wp_1[0])]  # pos @ wp1
    constraints += [(B_d.eval(T), wp_1[1])]  # vel @ wp1
    constraints += [(B_d2.eval(T), 0)]  # zero accel @ wp1
    
    assert len(constraints) == 6

    Y = ca.vertcat(*[c[0] for c in constraints])
    b = ca.vertcat(*[c[1] for c in constraints])
    A = ca.jacobian(Y, P)
    A_inv = ca.inv(A)
    P_sol = (A_inv@b).T

    return {
        'bezier6_solve': ca.Function('bezier6_solve', [wp_0, wp_1, T], [P_sol], ['wp_0', 'wp_1', 'T'], ['P']),
        'bezier6_traj': ca.Function('bezier6_traj', [t, T, P], [r], ['t', 'T', 'P'], ['r']),
    }

def multirotor_timeOpt(bc,k_time): ## Currently outputs optimized time
    time_opt = find_opt_time(6, bc,k_time)
    return np.average(time_opt)

def multirotor_plan(bc,T0):
    bezier_6 = derive_bezier6()

    bc = np.array(bc)
    t0 = np.linspace(0, T0, 100)

    PX = bezier_6['bezier6_solve'](bc[:, 0, 0], bc[:, 1, 0], T0)
    traj_x = np.array(bezier_6['bezier6_traj'](np.array([t0]), T0, PX)).T

    PY = bezier_6['bezier6_solve'](bc[:, 0, 1], bc[:, 1, 1], T0)
    traj_y = np.array(bezier_6['bezier6_traj'](np.array([t0]), T0, PY)).T

    PZ = bezier_6['bezier6_solve'](bc[:, 0, 2], bc[:, 1, 2], T0)
    traj_z = np.array(bezier_6['bezier6_traj'](np.array([t0]), T0, PZ)).T

    # V = np.sqrt(vx**2 + vy**2)

    return PX, PY, PZ, traj_x, traj_y, traj_z, t0

def generate_path(bc_t, k):
    
    t_total = 0
    res = {
        'anchor_x': [],
        'anchor_y': [],
        'anchor_z': [],
        'traj_x': [],
        'traj_y': [],
        'traj_z': [],
        'acc_x': [],
        'acc_y': [],
        'acc_z': [],
        'T': [],
        'T0': [],
        'omega_1': [],
        'omega_2': [],
        'omega_3': []
    }

    for i in range(bc_t.shape[1]-1):
        bc = bc_t[:,i:i+2,:]
        T0 = multirotor_timeOpt(bc, k) 
        Px, Py, Pz, traj_x, traj_y, traj_z, t0 = multirotor_plan(bc,T0)
        t = t_total + t0
        t_total = t_total + T0 
        x = traj_x[:, 0]
        y = traj_y[:, 0]
        z = traj_z[:, 0]
        vx = traj_x[:, 1]
        vy = traj_y[:, 1]
        vz = traj_z[:, 1]
        ax = traj_x[:, 2]
        ay = traj_y[:, 2]
        az = traj_z[:, 2]
        jx = traj_x[:, 3]
        jy = traj_y[:, 3]
        jz = traj_z[:, 3]
        sx = traj_x[:, 4]
        sy = traj_y[:, 4]
        sz = traj_z[:, 4]

        romega1 = []
        romega2 = []
        romega3 = []
    
        for j in range(100):
            r_vx = vx[j]
            r_vy = vy[j]
            r_vz = vz[j]
            r_ax = ax[j]
            r_ay = ay[j]
            r_az = az[j]
            r_jx = jx[j]
            r_jy = jy[j]
            r_jz = jz[j]
            r_sx = sx[j]
            r_sy = sy[j]
            r_sz = sz[j]
            ref_v = f_ref(0, 0, 0, [r_vx, r_vy, r_vz], [r_ax, r_ay, r_az], [r_jx, r_jy, r_jz], [r_sx, r_sy, r_sz], 1, 9.8, 1, 1, 1, 0)
            R = ref_v[1]
            theta = ca.DM(Euler.from_dcm(R))
            theta = np.array(theta).reshape(3,)
            r_theta1 = theta[0]
            r_theta2 = theta[1]
            r_theta3 = theta[2]
            omega = ref_v[2]
            omega = np.array(omega).reshape(3,)
            r_omega1 = omega[0]
            r_omega2 = omega[1]
            r_omega3 = omega[2]
            romega1.append(r_omega1)
            romega2.append(r_omega2)
            romega3.append(r_omega3)

        res['anchor_x'].extend(np.array(Px).tolist())
        res['anchor_y'].extend(np.array(Py).tolist())
        res['anchor_z'].extend(np.array(Pz).tolist())
        res['traj_x'].extend(x)
        res['traj_y'].extend(y)
        res['traj_z'].extend(z)
        res['acc_x'].extend(ax)
        res['acc_y'].extend(ay)
        res['acc_z'].extend(az)
        res['T'].extend(np.array(t).tolist())
        res['T0'].append(T0)
        res['omega_1'].extend(np.array(romega1).tolist())
        res['omega_2'].extend(np.array(romega2).tolist())
        res['omega_3'].extend(np.array(romega3).tolist())

    return res