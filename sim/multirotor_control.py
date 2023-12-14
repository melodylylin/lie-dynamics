import numpy as np
import casadi as ca
import control
from lie.SE23 import *
from .multirotor_ref_traj import f_ref
from scipy import signal
import matplotlib.pyplot as plt
from bezier.bezier_planning import *

bezier6 = derive_bezier6()

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

def control_law(B, K, e):
    e = SE23Dcm.vee(e)
    u = SE23Dcm.diff_correction_inv(e)@B@K@e # controller input
    # print(u)
    return u

def compute_control(t, y_vect, ref, freq_d, w1_mag, w2_mag, dist): # w1_mag: acceleration, w2_mag: omega
    # reference data (from planner, function of time)
    # x_opt is t coeff of poly
    T_opt = ref['T0']
    T = np.cumsum(T_opt)
    xr = ref['anchor_x']
    yr = ref['anchor_y']
    zr = ref['anchor_z']

    for i in range(T.shape[0]):
        if i==0 and t <= T[i]:
            traj_x = np.array(bezier6['bezier6_traj'](t, ref['T0'][i], xr[i])).T
            traj_y = np.array(bezier6['bezier6_traj'](t, ref['T0'][i], xr[i])).T
            traj_z = np.array(bezier6['bezier6_traj'](t, ref['T0'][i], xr[i])).T
            beta = t/T_opt[0]
        elif T[i-1] < t <= T[i]:
            traj_x = np.array(bezier6['bezier6_traj'](t-np.sum(T_opt[:i]), ref['T0'][i], xr[i])).T
            traj_y = np.array(bezier6['bezier6_traj'](t-np.sum(T_opt[:i]), ref['T0'][i], xr[i])).T
            traj_z = np.array(bezier6['bezier6_traj'](t-np.sum(T_opt[:i]), ref['T0'][i], xr[i])).T
            beta = (t-np.sum(T_opt[:i]))/T_opt[i]

    # reference input at time t
    # world frame
    r_vx = traj_x[:,1][0]
    r_vy = traj_y[:,1][0]
    r_vz = traj_z[:,1][0]
    r_ax = traj_x[:,2][0]
    r_ay = traj_y[:,2][0]
    r_az = traj_z[:,2][0]
    r_jx = traj_x[:,3][0]
    r_jy = traj_y[:,3][0]
    r_jz = traj_z[:,3][0]
    r_sx = traj_x[:,4][0]
    r_sy = traj_y[:,4][0]
    r_sz = traj_z[:,4][0]

    # body frame
    ref = f_ref(0, 0, 0, [r_vx, r_vy, r_vz], [r_ax, r_ay, r_az], [r_jx, r_jy, r_jz], [r_sx, r_sy, r_sz], 1, 9.8, 1, 1, 1, 0)
    R = np.array(ref[1])
    R_eb = np.linalg.inv(R) # earth frame to body frame
    ae = np.array([r_ax,r_ay,r_az]) # acc in earth frame
    ab = R@ae
    omega = ref[2]
    omega = np.array(omega).reshape(3,)
    r_omega1 = omega[0]
    r_omega2 = omega[1]
    r_omega3 = omega[2]
    
    # initial states of vehicle and reference and error
    e = SE23Dcm.wedge(np.array([y_vect[0], y_vect[1], y_vect[2], y_vect[3], y_vect[4], y_vect[5], y_vect[6], y_vect[7], y_vect[8]])) # log error
    
    B, K, _, _ = se23_solve_control(0, 0, 9.8, 0, 0, 0) # time-invariant at hover

    vr = np.array([0,0,0,ab[0],ab[1],ab[2],r_omega1,r_omega2,r_omega3])

    # disturbance
    if dist == 'sine':
        phi = 0.3
        phi2 = 0.5
        wax = np.cos(2*np.pi*freq_d*t+phi)*w1_mag
        way = np.sin(2*np.pi*freq_d*t+phi)*w1_mag/np.sqrt(2)
        waz = np.sin(2*np.pi*freq_d*t+phi)*w1_mag
        womega1 = np.cos(2*np.pi*freq_d*t+phi2)*w2_mag/np.sqrt(2)
        womega2 = np.sin(2*np.pi*freq_d*t+phi2)*w2_mag/np.sqrt(2)
        womega3 = np.sin(2*np.pi*freq_d*t+phi2)*w2_mag/np.sqrt(2)
    elif dist  == 'square':
        wax = signal.square(2*np.pi*freq_d*t+np.pi)*w1_mag/np.sqrt(2)
        way = signal.square(2*np.pi*freq_d*t)*w1_mag/2
        waz = signal.square(2*np.pi*freq_d*t)*w1_mag/2
        womega1 = signal.square(2*np.pi*freq_d*t+np.pi)*w2_mag/2
        womega2 = signal.square(2*np.pi*freq_d*t)*w2_mag/2
        womega3 = 0
    w = np.array([0,0,0,wax,way,waz,womega1,womega2,womega3])
    # print(w)
    
    # control law applied to log-linear error
    u = np.array(ca.DM(control_law(B, K, e))).reshape(9,)
        
    # log error dynamics
    U = ca.DM(SE23Dcm.diff_correction(SE23Dcm.vee(e)))
    # these dynamics don't hold exactly unless you can move sideways
    A = -ca.DM(SE23Dcm.ad_matrix(vr)+SE23Dcm.adC_matrix())
    # e_dot = (A+ B@K)@ca.DM(SE23Dcm.vee(e)) + U@w # vector form
    e_dot = (A)@ca.DM(SE23Dcm.vee(e)) + U@(u+w)
    e_dot = np.array(e_dot).reshape(9,)

    return [
            # log error
            e_dot[0],
            e_dot[1],
            e_dot[2],
            e_dot[3],
            e_dot[4],
            e_dot[5],
            e_dot[6],
            e_dot[7],
            e_dot[8],
        ]

def simulate_rover(ref, freq_d, w1, w2, x0, y0, z0, vx0, vy0, vz0, theta1_0, theta2_0, theta3_0, dist, plot=False):
    t = np.arange(0,np.sum(ref['T0']),0.01)
    X0 = SE23Dcm.matrix(np.array([x0, y0, z0, vx0, vy0, vz0, theta1_0, theta2_0, theta3_0]))  # initial vehicle position in SE2(3)
    X0_r = SE23Dcm.matrix(np.array([0,0,0,0,0,0,0,0,0]))  # initial reference position in SE2(3)
    e0 = SE23Dcm.log(SE23Dcm.inv(X0)@X0_r)  # initial log of error log(X^-1Xr)
    x0 = [ca.DM(SE23Dcm.vee(e0))][0] # intial state for system in vector form
    y0= np.array(x0).reshape(9,)

    import scipy.integrate
    res = scipy.integrate.solve_ivp(
        fun=compute_control,
        t_span=[t[0], t[-1]], t_eval=t,
        y0=y0, args=[ref, freq_d, w1, w2, dist], rtol=1e-6, atol=1e-9)
    return res

def compute_exp_log_err(rx, ry, rz, rvx, rvy, rvz, rtheta1, rtheta2, rtheta3, ex, ey, ez, evx, evy, evz, etheta1, etheta2, etheta3):
    zeta = SE23Dcm.wedge(np.array([ex, ey, ez, evx, evy, evz, etheta1, etheta2, etheta3]))
    eta = SE23Dcm.exp(zeta)
    eta_inv = ca.DM(SE23Dcm.inv(eta))
    Xr = ca.DM(SE23Dcm.matrix(np.array([rx, ry, rz, rvx, rvy, rvz, rtheta1, rtheta2, rtheta3])))
    X = Xr@eta_inv
    x_vect = ca.DM(SE23Dcm.vector(X))
    x_vect = np.array(x_vect).reshape(9,)
    return x_vect

def plot_rover_sim(freq, ref, abound, omegabound, invbound):
    fig = plt.figure(figsize=(9,18))
    plt.rcParams.update({'font.size': 18})
    fig.subplots_adjust(hspace=0.2, top=0.95)

    #   Add the main title
    fig.suptitle("Small Disturbance Case", fontsize=24)
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    T0_list = ref['T0']
    T = np.cumsum(T0_list)
    x = ref['traj_x']
    y = ref['traj_y']
    z = ref['traj_z']
    t = ref['T']
    Px = ref['anchor_x']
    Py = ref['anchor_y']
    Pz = ref['anchor_z']
    
    label_added =False
    for f in freq:
        print(f)
        res = simulate_rover(ref, f, abound, omegabound, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'sine')
        t = res['t']
    
        y_vect = res['y']
        ex, ey, ez, evx, evy, evz, etheta1, etheta2, etheta3 = [y_vect[i, :] for i in range(len(y_vect))]
        exp_log_err = np.zeros((9,len(t)))

        # ressq = simulate_rover(ref, f, abound, omegabound, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'square')
        # tsq = ressq['t']
        # y_vectsq = ressq['y']
        # exsq, eysq, ezsq, evxsq, evysq, evzsq, etheta1sq, etheta2sq, etheta3sq = [y_vectsq[i, :] for i in range(len(y_vectsq))]
        # exp_log_errsq = np.zeros((9,len(tsq)))
        

        for j in range(len(t)):
            for k in range(T.shape[0]):
                if k==0 and t[j] <= T[k]:
                    traj_x = np.array(bezier6['bezier6_traj'](t[j], ref['T0'][k], Px[k])).T
                    traj_y = np.array(bezier6['bezier6_traj'](t[j], ref['T0'][k], Py[k])).T
                    traj_z = np.array(bezier6['bezier6_traj'](t[j], ref['T0'][k], Pz[k])).T
                elif k > 0 and T[k-1] < t[j] <= T[k]:
                    traj_x = np.array(bezier6['bezier6_traj'](t[j]-T[k-1], ref['T0'][k], Px[k])).T
                    traj_y = np.array(bezier6['bezier6_traj'](t[j]-T[k-1], ref['T0'][k], Py[k])).T
                    traj_z = np.array(bezier6['bezier6_traj'](t[j]-T[k-1], ref['T0'][k], Pz[k])).T
            r_x = traj_x[:,0][0]
            r_y = traj_y[:,0][0]
            r_z = traj_z[:,0][0]
            r_vx = traj_x[:,1][0]
            r_vy = traj_y[:,1][0]
            r_vz = traj_z[:,1][0]
            r_ax = traj_x[:,2][0]
            r_ay = traj_y[:,2][0]
            r_az = traj_z[:,2][0]
            r_jx = traj_x[:,3][0]
            r_jy = traj_y[:,3][0]
            r_jz = traj_z[:,3][0]
            r_sx = traj_x[:,4][0]
            r_sy = traj_y[:,4][0]
            r_sz = traj_z[:,4][0]
            ref_v = f_ref(0, 0, 0, [r_vx, r_vy, r_vz], [r_ax, r_ay, r_az], [r_jx, r_jy, r_jz], [r_sx, r_sy, r_sz], 1, 9.8, 1, 1, 1, 0)
            R = ref_v[1]
            theta = ca.DM(Euler.from_dcm(R))
            theta = np.array(theta).reshape(3,)
            r_theta1 = theta[0]
            r_theta2 = theta[1]
            r_theta3 = theta[2]
            omega = ref_v[2]
            omega = np.array(omega).reshape(3,)
            exp_log_err[:,j] = np.array([compute_exp_log_err(r_x, r_y, r_z, r_vx, r_vy, r_vz, r_theta1, r_theta2, r_theta3,
                                                            ex[j], ey[j], ez[j], evx[j], evy[j], evz[j], etheta1[j], etheta2[j], etheta3[j])])
            # exp_log_errsq[:,j] = np.array([compute_exp_log_err(r_x, r_y, r_z, r_vx, r_vy, r_vz, r_theta1, r_theta2, r_theta3,
            #                                                 exsq[j], eysq[j], ezsq[j], evxsq[j], evysq[j], evzsq[j], etheta1sq[j], etheta2sq[j], etheta3sq[j])])
    
        if not label_added:
            ax1.plot(t, exp_log_err[0,:], 'g', label='x',linewidth=0.7)
            ax2.plot(t, exp_log_err[1,:], 'g', label='y',linewidth=0.7)
            ax3.plot(t, exp_log_err[2,:], 'g', label='z',linewidth=0.7)
            label_added = True
        else:
            ax1.plot(t, exp_log_err[0,:], 'g',linewidth=0.7)
            ax2.plot(t, exp_log_err[1,:], 'g',linewidth=0.7)
            ax3.plot(t, exp_log_err[2,:], 'g',linewidth=0.7)
            # ax1.plot(t, exp_log_errsq[0,:], 'g',linewidth=0.7)
            # ax2.plot(t, exp_log_errsq[1,:], 'g',linewidth=0.7)
            # ax3.plot(t, exp_log_errsq[2,:], 'g',linewidth=0.7)
            

    t_vect = np.linspace(1e-5,np.cumsum(T0_list)[-1],80)
    ax1.plot(ref['T'], ref['traj_x'], 'r--', label='reference x')
    ax1.plot(t_vect, invbound[0,:], 'c', label='LMI')
    ax1.plot(t_vect, invbound[3,:], 'c')
    ax1.set_xlabel('t, sec')
    ax1.set_ylabel('x, m')
    ax1.grid(True)
    ax1.legend(loc=2)

    ax2.plot(ref['T'], ref['traj_y'], 'r--', label='reference y')
    ax2.plot(t_vect, invbound[1,:], 'c', label='LMI')
    ax2.plot(t_vect, invbound[4,:], 'c')
    ax2.set_xlabel('t, sec')
    ax2.set_ylabel('y, m')
    ax2.grid(True)
    ax2.legend(loc=2)

    ax3.plot(ref['T'], ref['traj_z'], 'r', label='reference z')
    ax3.plot(t_vect, invbound[2,:], 'c', label='LMI')
    ax3.plot(t_vect, invbound[5,:], 'c')
    ax3.set_xlabel('t, sec')
    ax3.set_ylabel('z, m')
    ax3.grid(True)
    ax3.legend(loc=2)

    # plt.savefig('figures/Inv_bound_s.eps', format='eps', bbox_inches='tight')

    return 