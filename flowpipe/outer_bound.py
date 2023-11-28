import numpy as np
import picos
import control
import itertools
import scipy
from lie.SE23 import *
from lie.se3 import *

def se23_solve_control(ax,ay,az,omega1,omega2,omega3):
    A = -ca.DM(SE23Dcm.ad_matrix(np.array([0,0,0,ax,ay,az,omega1,omega2,omega3]))+SE23Dcm.adC_matrix())
    B = np.array([[0,0,0,0], # vx
                  [0,0,0,0], # vy
                  [0,0,0,0], # vz
                  [0,0,0,0], # ax
                  [0,0,0,0], # ay
                  [1,0,0,0], # az
                  [0,1,0,0], # omega1
                  [0,0,1,0], # omega2
                  [0,0,0,1]]) # omega3 # control omega1,2,3, and az
    Q = 10*np.eye(9)  # penalize state
    R = 1*np.eye(4)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R) 
    K = -K # rescale K, set negative feedback sign
    BK = B@K
    return B, K, BK , A+B@K

def SE23LMIs(alpha, A, B, verbosity=0):
    prob = picos.Problem()
    P = picos.SymmetricVariable('P', (9, 9))
    P1 = P[:3, :]
    P2 = P[3:6, :]
    P3 = P[6:, :]
    mu1 = picos.RealVariable('mu_1')
    mu2 = picos.RealVariable('mu_2')
    mu3 = picos.RealVariable('mu_3')
    gam = mu1 + mu2 + mu3
    for Ai in A:
        block_eq1 = picos.block([
                [Ai.T*P + P*Ai + alpha*P, P1.T, P2.T, P3.T],
                [P1, -alpha*mu1*np.eye(3), 0, 0],
                [P2, 0, -alpha*mu2*np.eye(3), 0],
                [P3, 0, 0, -alpha*mu3*np.eye(3)]])
        prob.add_constraint(block_eq1 << 0)
    # block_eq1 = picos.block([
    #             [A.T*P + P*A + alpha*P, P1.T, P2.T, P3.T],
    #             [P1, -alpha*mu1*np.eye(3), 0, 0],
    #             [P2, 0, -alpha*mu2*np.eye(3), 0],
    #             [P3, 0, 0, -alpha*mu3*np.eye(3)]])
    # prob.add_constraint(block_eq1 << 0)
    prob.add_constraint(P >> 1)
    prob.add_constraint(mu1 >> 1e-5)
    prob.add_constraint(mu2 >> 1e-5)
    prob.add_constraint(mu3 >> 1e-5)
    prob.set_objective('min', gam)
    try:
        prob.solve(solver="cvxopt", options={'verbosity': verbosity})
        cost = mu1.value
    except Exception as e:
        print(e)
        cost = -1
    return {
        'cost': cost,
        'prob': prob,
        'mu1': mu1.value,
        'mu2': mu2.value,
        'mu3': mu3.value,
        'P': np.array(P.value),
        'alpha':alpha    
        }

def find_se23_invariant_set(ax,ay,az,omega1,omega2,omega3, alpha_opt, verbosity=0):

    # B_lie, K, BK, _ = se23_solve_control(0, 0, -9.8, 0, 0, 0)
    # A0 = -ca.DM(SE23Dcm.ad_matrix(np.array([0,0,0,ax,ay,az,omega1,omega2,omega3]))+SE23Dcm.adC_matrix())
    # A = np.array(A0+BK)
    # eig = np.linalg.eig(A)[0]

    iterables =[omega1, omega2, omega3, ax, ay, az]
    nu = []
    for m in itertools.product(*iterables):
        m = np.array(m)
        nu.append(m)
    
    A = []
    eig = []
    for nui in nu:
        omega1 = nui[0]
        omega2 = nui[1]
        omega3 = nui[2]
        ax = nui[3]
        ay = nui[4]
        az = nui[5]

        B_lie, K, BK, _ = se23_solve_control(0, 0, -9.8, 0, 0, 0)
        A0 = -ca.DM(SE23Dcm.ad_matrix(np.array([0,0,0,ax,ay,az,omega1,omega2,omega3]))+SE23Dcm.adC_matrix())
        Ai = np.array(A0+BK)
        A.append(Ai)
        eig.append(np.linalg.eig(Ai)[0])
    
    # we use fmin to solve a line search problem in alpha for minimum gamma
    if verbosity > 0:
        print('line search')
    # we perform a line search over alpha to find the largest convergence rate possible
    alpha_1 = -np.real(np.max(eig)) # smallest magnitude value from eig-value, and range has to be positive
    # if the alpha optimization fail, pick a fixed value for alpha.
    # alpha_opt = 0.71#scipy.optimize.fminbound(lambda alpha: SE23LMIs(alpha, A, B_lie, verbosity=verbosity)['cost'], x1=1e-5, x2=alpha_1, disp=True if verbosity > 0 else False)
    print(alpha_opt)
    
    sol = SE23LMIs(alpha_opt, A, B_lie)
    prob = sol['prob']
    print(prob.status)
    if prob.status == 'optimal':
        P = prob.variables['P'].value
        mu1 =  prob.variables['mu_1'].value
        if verbosity > 0:
            print(sol)
    else:
        raise RuntimeError('Optimization failed')
        
    return sol

def se23_invariant_set_points_theta(sol, t, w1_norm, w2_norm, beta): # w1_norm: omega # w2_norm: a  
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu3']*w1_norm**2 + sol['mu2']*w2_norm**2)*(1-np.exp(-sol['alpha']*t))) # V(t)
    # 1 = xT(P/V(t))x, equation for the ellipse
    P1 = sol['P']/val
    A1 = P1[:6,:6]
    B1 = P1[:6,6:]
    C1 = P1[6:,:6]
    D1 = P1[6:,6:]
    P = D1-C1@np.linalg.inv(A1)@B1
    
    evals, evects = np.linalg.eig(P)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 30
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    return R@points, val

def se23_invariant_set_points(sol, t, w1_norm, w2_norm, beta): # w1_norm: omega, w2_norm: a
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu3']*w1_norm**2 + sol['mu2']*w2_norm**2)*(1-np.exp(-sol['alpha']*t)))+0.05 # V(t)
    # 1 = xT(P/V(t))x, equation for the ellipse
    P1 = sol['P']/val
    A1 = P1[:3,:3]
    B1 = P1[:3,3:]
    C1 = P1[3:,:3]
    D1 = P1[3:,3:]
    P = A1-B1@np.linalg.inv(D1)@C1
    
    evals, evects = np.linalg.eig(P)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 30
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    return R@points, val

def inv_bound(sol, t, omegabound, abound, ebeta, ebeta_theta):
    points, val = se23_invariant_set_points(sol, t, omegabound, abound, ebeta)
    points_theta, val = se23_invariant_set_points_theta(sol, t, omegabound, 0.5, ebeta_theta)
    inv_points = np.zeros((3,points.shape[1]))
    for i in range(points.shape[1]):
        Lie_points = SE3Dcm.wedge(np.array([points[0,i], points[1,i], points[2,i], points_theta[0,i], points_theta[1,i], points_theta[2,i]]))
        exp_points = ca.DM(SE3Dcm.vector(SE3Dcm.exp(Lie_points)))
        exp_points = np.array(exp_points).reshape(6,)
        inv_points[:,i] = np.array([exp_points[0], exp_points[1], exp_points[2]])
    xmax = inv_points[0,:].max()
    ymax = inv_points[1,:].max()
    zmax = inv_points[2,:].max()
    xmin = inv_points[0,:].min()
    ymin = inv_points[1,:].min()
    zmin = inv_points[2,:].min()
    # due to the bound is obtained numerically, therefor, we set a small 
    return np.array([xmax,ymax,zmax,xmin,ymin,zmin])