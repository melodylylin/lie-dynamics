import numpy as np
from pytope import Polytope
import casadi as ca
import matplotlib.pyplot as plt
from lie.se3 import *
from .IntervalHull import qhull2D, minBoundingRect
from .outer_bound import se23_invariant_set_points, se23_invariant_set_points_theta, exp_map

def flowpipes(ref, n, beta, w1, w2, omegabound, sol):
    
    # ref_theta = ref_data['theta']

    x_r = ref['traj_x']
    y_r = ref['traj_y']
    z_r = ref['traj_z']
    
    nom = np.array([x_r,z_r]).T
    flowpipes = []
    intervalhull = []
    t_vect = []
    Ry1 = []
    Ry2 = []
    
    steps0 = int(len(x_r)/n)
    
    a = 0    
    for i in range(n):
        if i < len(x_r)%n:
            steps = steps0 + 1
        else:
            steps = steps0
        b = a + steps
        if i == n-1:
            nom_i = nom[a:len(x_r)+1,:]
            if nom_i.shape[0] < 3:
                nom_i = np.vstack((nom_i, np.array(nom[-1,:])+0.01))
        else:
            nom_i = nom[a:b+1,:]
        # Get interval hull
        hull_points = qhull2D(nom_i)
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
            
        t = 0.05*a
        t_vect.append(t)
        ang_list = []
        # for k in range(a, b):
        #     if k == 0:
        #         k = 1e-3
        #     angle = ref_theta(0.05*k)
        #     ang_list.append(angle)
        
        # invariant set in se2
        if t == 0:
            t = 1e-3
        points, val1 = se23_invariant_set_points(sol, t, omegabound, w1, beta) # invariant set at t0 in that time interval
        points2, val2 = se23_invariant_set_points(sol, 0.05*b, omegabound, w1, beta) # invariant set at t final in that time interval
        points_theta, _ = se23_invariant_set_points_theta(sol, t, omegabound, w2, beta)
        
        if val2 > val1: 
            points = points2
            points_theta, _ = se23_invariant_set_points_theta(sol, 0.05*b, omegabound, w1, beta)
        # lyap.append(val)
        
        # exp map (invariant set in Lie group) x, y, theta
        # inv_points = np.zeros((3,points.shape[1]))
        # for j in range(points.shape[1]):
        #     # print(points[2,j])
        #     # val = points2[:,j].T@sol['P']@points2[:,j]
        #     # lyap.append(val)
        #     exp_points = se2(points[0,j], points[1,j], points[2,j]).exp
        #     inv_points[:,j] = np.array([exp_points.x, exp_points.y, exp_points.theta])

        inv_points = exp_map(points, points_theta)

        inv_set = np.delete(inv_points,1,0)
        
        max_x = inv_set[0,:].max()
        min_x = inv_set[0,:].min()
        x_bound = np.sqrt(min_x**2 + max_x**2)
        max_y = inv_set[1,:].max()
        min_y = inv_set[1,:].min()
        y_bound = np.sqrt(min_y**2 + max_y**2)
        # max_z = inv_set[2,:].max()
        # min_z = inv_set[2,:].min()
        # z_bound = np.sqrt(min_z**2 + max_z**2)
            
        P2 = Polytope(inv_set.T) 
        
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        
        P = P1 + P2 # sum

        p1_vertices = P1.V
        p_vertices = P.V

        p_vertices = np.append(p_vertices, p_vertices[0].reshape(1,2), axis = 0) # add the first point to last, or the flow pipes will miss one line
        
        # create list for flow pipes and interval hull
        flowpipes.append(p_vertices)
        intervalhull.append(P1.V)
        
        a = b
    return flowpipes, intervalhull, nom, t_vect #, Ry1, Ry2

def plot_flowpipes(nom, flowpipes, n):
    # flow pipes
    plt.figure(figsize=(15,15))
    ax = plt.subplot(111)
    h_nom = ax.plot(nom[:,0], nom[:,1], color='k', linestyle='-')
    for facet in range(n):
        hs_ch_LMI = ax.plot(flowpipes[facet][:,0], flowpipes[facet][:,1], color='c', linestyle='--')

    # plt.axis('equal')
    plt.title('Flow Pipes')
    plt.xlabel('x')
    plt.ylabel('z')
    lgd = plt.legend(loc=2, prop={'size': 18})
    ax = lgd.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(h_nom[0])
    labels.append('Reference Trajectory')
    handles.append(hs_ch_LMI[0])
    labels.append('Flow Pipes')
    lgd._legend_box = None
    lgd._init_legend_box(handles, labels)
    lgd._set_loc(lgd._loc)
    lgd.set_title(lgd.get_title().get_text())
