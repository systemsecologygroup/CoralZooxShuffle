# -*- coding: utf-8 -*-
from __future__ import division

from scipy import exp, linspace, array, zeros, e, sqrt, mean,var, ones, cumsum, random, sin, pi, load, floor, ceil
from scipy.stats import norm
from numpy import amin, amax, meshgrid, arange, isnan, logical_not, interp, concatenate, arange, isnan, nan
from scipy.integrate import ode
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon

font = {'family' : 'normal',
        'weight' : 'bold'}
#matplotlib.rc('font',**{'family':'Times', 'sans-serif':['Times']})
#matplotlib.rc('text', usetex = True)
#matplotlib.rc('font', **font)


plt.rcParams["font.family"] = "arial"   

font0 = FontProperties()
font = font0.copy()
weight = "bold"
font.set_weight(weight)   

#### Model with a dynamics on 2 types of symbiont population using odeint solver + bleaching + sinusoidal temperature #####
G_C = 10  # this is G_max in the model, per year
a = 1.0768 # Symbiont specific growth rate - linear growth rate
b = 0.0633 # Exponential Growth constant of the symbiont

alpha = 1e-3 # slope of cost associated to investment in symbiont, in case we assume the same cost for any symbiont

M_C = 10e-3 # coral mortality 

K_C = 5125112.94092e10 # earth coral carying capacity
Ksmax = 3e6 # healthy measure of carying capacity of symbiont per host biomass
N = 1e-5 # speed of adaptation assumed to be 1e6 times faster than the value we found for MS2, however this is necessary in order to reach steady state fast
         # and within our study timerange. We could say that we assume an unreasonably high capacity of the corals to acclimate to change in temperature 
         # so that the success of the holobiont uniquely depends on the symbiont contribution.
beta = 1e2

gammaH = 0.25e6 # symbiont to coral biomass ration for which contribution of symbiont to coral growht reaches a proportion of 0.5 (there are 3 symbiont population that could reach gammaH so there could be 3*gammaH symbiont within the coral host)


r = 1e3
error = 1e1 # prevent division by zeros these scales are alright because we are dealing with very high numbers e14
error2 = 1e1 # prevent division by zeros in the symbiont population dynamics
error3 = 1e1 # prevent division by zeros

color_list = array([(0.647*col, 0.333*col, 0.075*col) for col in xrange(2,5+2)])/5

tau1 = 28 # mean thermal tolerance of highly thermally tolerant symbiont
tau2 = 27 # mean thermal tolerance of intermediatelly thermaly tolerant symbiont
tau3 = 26 # mean thermal tolerance of barely thermally tolerant symbiont

sigma1 = 2 # tolerance range of highly thermally tolerant symbiont 1 
sigma2 = 1.5 # tolerance range of intermediatelly thermaly tolerant symbiont
sigma3 = 0.5 #tolerance range of barely thermally tolerant symbiont

# Growth deficiency of corals due to association with symbiont type
g1 = 0.55
g2 = 0.35
g3 = 0.00
# Fraction of coral growth achieve through association with symbiont type
p1 = 1 - g1 # 
p2 = 1 - g2 # 
p3 = 1 - g3 # 

# Maximal symbiont Competition term
cmax = 2 # in the case of competition depending on temperature
# intraspefic competition term
c11 = 1.
c22 = 1.
c33 = 1.

# precent symbiont entering the system every year
S00 = 0.1
  
# Gradient of coral fitness 
#def Gradient(coral, u_i, u, cost_gam, Gx1, beta, K_symb, kappa_i, alpha, p_i, c_i):
#    Grad = Gx1*beta*p_i*kappa_i*(1 - coral/K_C)*exp(-beta*u_i) - r*alpha*cost_gam*exp(r*u)
#    return Grad 
 
# Gradient of coral fitness, with bound to avoid negative investment, not needed here
def DELTA(u_i): # make delta high relative to u_i
    if abs(u_i)<=1e-5:
        res = 1e7
    elif abs(u_i)<1e-3 and abs(u_i)>1e-5:
        res = 1e6
    elif abs(u_i)>=1e-3 and abs(u_i)<1e3:
        res = 1e5
    else: # this is for abs(u_i)>=1e3 the complementary of all the above conditions
        res = 1e2
    return res
    
def Gradient(Bound, coral, u_i, u, cost_gam, Gx1, beta, K_symb, kappa_i, alpha, delta, p_i, c_i):
    Grad = Gx1*beta*p_i*c_i*kappa_i*(1 - coral/K_C)*exp(-beta*u_i) - (Bound*(r*alpha*cost_gam*exp(r*u)) - delta*r*alpha*cost_gam*exp(r*u)*exp(-delta*u_i))
    return Grad    
               
# System of ODE for hypothesis 1
def SystemForcing1(t, y, Temperature, beta, gammaH, sigma1, sigma2, sigma3, S00, N, cmax):
    dSystem = zeros(len(y))
    coral = y[0]
    
    symbiont1 = y[4]  # Group of highly thermaly tolerant symbiont
    symbiont2 = y[5]  # Group of intermediately thermaly tolerant symbiont
    symbiont3 = y[6]  # Group of  barely thermally tolerant symbiont
    
    # Coral investement in symbiont_i, obviously only if there are symbiont
    if symbiont1 !=0:
        u1 = y[1] 
    else:
        u1 = 0  
    if symbiont2 !=0:
        u2 = y[2]
    else:
        u2 = 0
    if symbiont3 !=0:
        u3 = y[3]
    else:
        u3 = 0
    
    E1 = 1 - exp(-beta*u1)
    E2 = 1 - exp(-beta*u2)
    E3 = 1 - exp(-beta*u3)
        
    Gx1Forcing = G_C 
    
    # Computing the derivative
    K_symb = Ksmax*coral # symbiont carrying capacity
    symbiontH = (gammaH)*coral
    
    c1 = exp(-(Temperature - tau1)**2/(2*sigma1**2))
    c2 = exp(-(Temperature - tau2)**2/(2*sigma2**2)) 
    c3 = exp(-(Temperature - tau3)**2/(2*sigma3**2))
                    
    kappa1 = symbiont1/(symbiontH + symbiont1 + error3)
    kappa2 = symbiont2/(symbiontH + symbiont2 + error3)
    kappa3 = symbiont3/(symbiontH + symbiont3 + error3)
    
    kappaE = (p1*c1*kappa1*E1 + p2*c2*kappa2*E2 + p3*c3*kappa3*E3)  
    
    symbiont = symbiont1 + symbiont2 +symbiont3
    
    cost_gam = symbiont/(K_symb + error)
    
    Benefit = Gx1Forcing*kappaE*(1-coral/K_C)
    
    u = u1 + u2 + u3  
    
    delta1 = DELTA(u1)
    delta2 = DELTA(u2)
    delta3 = DELTA(u3)
    Bound = 1 + exp(-delta1*u1) + exp(-delta2*u2) + exp(-delta3*u3) 
    
    Cost = Bound*alpha*exp(r*u)*(cost_gam) + M_C # including a bound to prevent negative investements
    #Cost = alpha*exp(r*u)*(cost_gam) + M_C
    
    Fitness = (Benefit - Cost)
    dSystem[0] = Fitness*coral
    
    #dSystem[1] = N*Gradient(coral, u1, u, cost_gam, Gx1Forcing, beta, K_symb, kappa1, alpha)
    #dSystem[2] = N*Gradient(coral, u2, u, cost_gam, Gx1Forcing, beta, K_symb, kappa2, alpha)
    #dSystem[3] = N*Gradient(coral, u3, u, cost_gam, Gx1Forcing, beta, K_symb, kappa3, alpha)
    
    dSystem[1] = N*Gradient(Bound, coral, u1, u, cost_gam, Gx1Forcing, beta, K_symb, kappa1, alpha, delta1, p1, c1)
    dSystem[2] = N*Gradient(Bound, coral, u2, u, cost_gam, Gx1Forcing, beta, K_symb, kappa2, alpha, delta2, p2, c2)
    dSystem[3] = N*Gradient(Bound, coral, u3, u, cost_gam, Gx1Forcing, beta, K_symb, kappa3, alpha, delta3, p3, c3)
    
    G_S =  a*exp(b*Temperature) # symbiont temperature associated growth
    # Symbiont carrying capacity, error3 is very small compared to the ranges of K_symb and prevents division by zero
    K_symb1 = K_symb + error3
    K_symb2 = K_symb + error3
    K_symb3 = K_symb + error3
     
        
    # symbiont competition 
    c21 = cmax*c1 # S_1 competition with S_2 # S_1 are high thermal tolerant
    c31 = cmax*c1 # S_1 competition with S_3 
    
    c12 = cmax*c2  # S_2 competition with S_1 # S_1 are high thermal tolerant
    c32 = cmax*c2  # S_2 competition with S_3

    
    c13 = cmax*c3 # S_3 competition with S_1 # S_1 are high thermal tolerant
    c23 = cmax*c3 # S_3 competition with S_2
    
    #Symbiont population dynamics
    
    dSystem[4] = G_S*(1-(c11*symbiont1+c12*symbiont2+c13*symbiont3)/K_symb1)*(symbiont1) + S00*Ksmax*coral*exp(-symbiont1/K_symb1)
    
    dSystem[5] = G_S*(1-(c21*symbiont1+c22*symbiont2+c23*symbiont3)/K_symb2)*(symbiont2) + S00*Ksmax*coral*exp(-symbiont2/K_symb2)
   
    dSystem[6] = G_S*(1-(c31*symbiont1+c32*symbiont2+c33*symbiont3)/K_symb3)*(symbiont3) + S00*Ksmax*coral*exp(-symbiont3/K_symb3)

    return dSystem
    
# System of ODE for hypothesis 2
rho = 10
def SystemForcing2(t, y, Temperature, beta, gammaH, sigma1, sigma2, sigma3, S00, N, cmax, rho):
    dSystem = zeros(len(y))
    coral = y[0]
    
    symbiont1 = y[4]  # Group of highly thermaly tolerant symbiont
    symbiont2 = y[5]  # Group of intermediately thermaly tolerant symbiont
    symbiont3 = y[6]  # Group of  barely thermally tolerant symbiont
    
    # Coral investement in symbiont_i, obviously only if there are symbiont
    if symbiont1 !=0:
        u1 = y[1] 
    else:
        u1 = 0  
    if symbiont2 !=0:
        u2 = y[2]
    else:
        u2 = 0
    if symbiont3 !=0:
        u3 = y[3]
    else:
        u3 = 0
    
    E1 = 1 - exp(-beta*u1)
    E2 = 1 - exp(-beta*u2)
    E3 = 1 - exp(-beta*u3)
        
    Gx1Forcing = G_C 
    
    # Computing the derivative
    K_symb = Ksmax*coral # symbiont carrying capacity

    K_symb1 = K_symb 
    K_symb2 = K_symb 
    K_symb3 = K_symb 
     
    symbiontH = (gammaH)*coral
    
    c1 = exp(-(Temperature - tau1)**2/(2*sigma1**2))
    c2 = exp(-(Temperature - tau2)**2/(2*sigma2**2)) 
    c3 = exp(-(Temperature - tau3)**2/(2*sigma3**2))
                    
    kappa1 = symbiont1/(symbiontH + symbiont1 + error3)
    kappa2 = symbiont2/(symbiontH + symbiont2 + error3)
    kappa3 = symbiont3/(symbiontH + symbiont3 + error3)
    
    kappaE = (p1*c1*kappa1*E1 + p2*c2*kappa2*E2 + p3*c3*kappa3*E3)  
    
    symbiont = symbiont1 + symbiont2 +symbiont3
    
    cost_gam = symbiont/(K_symb + error)
    
    Benefit = Gx1Forcing*kappaE*(1-coral/K_C)
    
    u = u1 + u2 + u3  
    
    delta1 = DELTA(u1)
    delta2 = DELTA(u2)
    delta3 = DELTA(u3)
    Bound = 1 + exp(-delta1*u1) + exp(-delta2*u2) + exp(-delta3*u3) 
    
    Cost = Bound*alpha*exp(r*u)*(cost_gam) + M_C # including a bound to prevent negative investements
    #Cost = alpha*exp(r*u)*(cost_gam) + M_C # without bound
    
    Fitness = (Benefit - Cost)
    dSystem[0] = Fitness*coral
    
    # Without bound
    #dSystem[1] = N*Gradient(coral, u1, u, cost_gam, Gx1Forcing, beta, K_symb, kappa1, alpha, p1, c1)
    #dSystem[2] = N*Gradient(coral, u2, u, cost_gam, Gx1Forcing, beta, K_symb, kappa2, alpha, p2, c2)
    #dSystem[3] = N*Gradient(coral, u3, u, cost_gam, Gx1Forcing, beta, K_symb, kappa3, alpha, p3, c3)
    
    dSystem[1] = N*Gradient(Bound, coral, u1, u, cost_gam, Gx1Forcing, beta, K_symb, kappa1, alpha, delta1, p1, c1)
    dSystem[2] = N*Gradient(Bound, coral, u2, u, cost_gam, Gx1Forcing, beta, K_symb, kappa2, alpha, delta2, p2, c2)
    dSystem[3] = N*Gradient(Bound, coral, u3, u, cost_gam, Gx1Forcing, beta, K_symb, kappa3, alpha, delta3, p3, c3)
    
    G_S =  a*exp(b*Temperature) # symbiont temperature associated growth
    # Symbiont carrying capacity, error3 is very small compared to the ranges of K_symb and prevents division by zero
    K_symb1 = K_symb + error3
    K_symb2 = K_symb + error3
    K_symb3 = K_symb + error3
     
       
    # first feed back loop
    f11 = 1+rho*E1
    f12 = 1+rho*E2
    f13 = 1+rho*E3       
    # symbiont competition 
    c21 = cmax*c1 # S_1 competition with S_2 # S_1 are high thermal tolerant
    c31 = cmax*c1 # S_1 competition with S_3 
    
    c12 = cmax*c2  # S_2 competition with S_1 # S_1 are high thermal tolerant
    c32 = cmax*c2  # S_2 competition with S_3

    
    c13 = cmax*c3 # S_3 competition with S_1 # S_1 are high thermal tolerant
    c23 = cmax*c3 # S_3 competition with S_2
    
    #Symbiont population dynamics
    
    dSystem[4] = G_S*f11*(1-(c11*symbiont1+c12*symbiont2+c13*symbiont3)/K_symb1)*(symbiont1) + S00*Ksmax*coral*exp(-symbiont1/K_symb1)
    
    dSystem[5] = G_S*f12*(1-(c21*symbiont1+c22*symbiont2+c23*symbiont3)/K_symb2)*(symbiont2) + S00*Ksmax*coral*exp(-symbiont2/K_symb2)
    
    dSystem[6] = G_S*f13*(1-(c31*symbiont1+c32*symbiont2+c33*symbiont3)/K_symb3)*(symbiont3) + S00*Ksmax*coral*exp(-symbiont3/K_symb3)
    
    return dSystem

"""
# For Figure 5 Sensitivity of timing of symbiont shuffling with respect to S_0, Using temperature forcing 2
"""
# Use forcing 2 so that we can also say something about the recovery of initial symbiont population
# the first phase (t<t0) of each forcing are identical but t1 is not needed for Forcing1 since temperature is not recovering, I just need to have this here for consistency in the next lines of codes 
def Forcing1(t, t0, t1, tk, tl, Tbar0, AorB): # temperature is not recovering
    if t<t0:
        forc1 = Tbar0 
    elif t>=t0:
        Half = tk # half saturation 
        forc1 = Tbar0 + AorB*((t - t0)**tl/(Half**tl + (t-t0)**tl))
    return forc1    
    
def Forcing2(t, t0, t1, tk, tl, Tbar0, AorB): # temperature is recovering
    midDuration = (t1-t0)/2
    if t<t0+midDuration:
        forc2 = Forcing1(t, t0, t1, tk, tl, Tbar0, AorB)
    elif t>=t0 + midDuration and t<t1:
        Half = tk
        t0 = t0+midDuration # shift the origin of the sigmoid 
        t = t1 + t0 - t # invert the sigmoid symetrically
        forc2 = Tbar0+AorB*(((t - t0)**tl/(Half**tl + (t-t0)**tl)))
    else:
        forc2 = Tbar0
    return forc2    
        
F_list = [Forcing1, Forcing2]
Forc = ["Forcing1", "Forcing2"]  

cases = ["case1/", "case2/"]

#Simulation step and time in year after spine up
step = 0.001
tmax = 10

# Initial conditions
symb1_list = array([0.1, 0.8]) # proportion of symbiont1 to be studied
symb2_list = array([0.1, 0.8]) # proportion of symbiont2 to be studied
symb3_list = array([0.1, 0.8]) # proportion of symbiont3 to be studied

# init symb biomass index low = 1 , high = 2 
low = 0
high = 1

# Parameters for temperature forcing
Tk = [(0.25/12)/14] # half saturation of sigmoid is 1/2 day after t0, the same for all forcing (see forcing functions)
Tl = [2] # power of the sigmoid
# list of deviations from tbar0
ab_steps = 0.1
A_B = arange(0, 4+ab_steps, ab_steps)

F_list = [Forcing1, Forcing2] # forcing list
Forc = ["Forcing1", "Forcing2"]  

A_Bstr =["Height%d"%i for i in xrange(len(A_B))] # filenams for each elements in the list of deviations from Tbar0
Tbar0_list  = [26, 27] # 28°C doesn't show any symbiont shuffling because there are not symbiont more adapted to temperature higher than 28°C

t0 = 2 # maximum year of spine up
dur_steps = 0.25/12  # steps of week i.e quater a month
dur_list = arange(0.25/12, 4+dur_steps, dur_steps) # list of duration of stress, by steps of half a month, minimun duration is one week 
dur_str=["Dur%d"%i for i in xrange(len(dur_list))]


# symbiont more adapted to mean temperature Tbar0 is the more abundant       
IndexInit_26 = (low, low, high)  # for Tbar0 = 26
IndexInit_27 = (low, high, low)  # for Tbar0 = 27
IndexInit_28 = (high, low, low) # for Tbar0 = 28
IndexInit_list = [IndexInit_26, IndexInit_27, IndexInit_28] 


# specific duration and magnitude of termal stress to study resp. corresponding to each temperature in Tbar0_list
dl_list = [30, 54] # index of duration 
iAB_list = [10, 20] # in xrange(len(A_B))] # Index of temperature deviations to study

# S0 range to study
maxprop = 20*S00 # 2
S0_list =  linspace(0, maxprop, 200) # good number of points = 200 

fsize = 18#14#10#14
fsizeT = 20#16#12#16    

"""
fig2 = plt.figure(figsize = (12, 9))
plt.subplots_adjust(bottom = 0.1, right = 0.70, left = 0.15, top = 0.95, wspace = 0.1, hspace = 0.165)            
"""
# fig labels
Lab1 = ["a", "b"]
Lab2 = ["c", "d"]

#linewidths
lw_case1 = 1.5
lw_case2 = 6

### Saving data. For just plotting, put this block as comment after saving the data, because it takes time to run 
### Forcing 2 is used
"""
ShiftList1 = []
ShiftList2 = []
RecovList1 = []
RecovList2 = []

for tb in xrange(len(Tbar0_list)): 
    DL = dl_list[tb]  
    iAB = iAB_list[tb]                              
    t1 = t0 + dur_list[DL]
    IndexInit = IndexInit_list[tb]  
    Tbar0 = Tbar0_list[tb]                                                                            
    print IndexInit
    i1 = IndexInit[0]
    i2 = IndexInit[1]
    i3 = IndexInit[2]
    symb1 = symb1_list[i1] 
    symb2 = symb2_list[i2]
    symb3 = symb3_list[i3]
    Initial = array([0.75*K_C, 0.0000005, 0.0000005, 0.0000005, symb1*Ksmax*0.75*K_C, symb2*Ksmax*0.75*K_C, symb3*0.75*Ksmax*K_C])
    Time = arange(0, t0+tmax+step, step)
    
    ShiftDomTiming1 = zeros(len(S0_list)) 
    ShiftDomTiming2 = zeros(len(S0_list))
    RecovTiming1 = zeros(len(S0_list))
    RecovTiming2 = zeros(len(S0_list))
    for iS0 in xrange(len(S0_list)):
        S00_i = S0_list[iS0]
        tk = Tk[0]
        tl = Tl[0]
        print Tbar0, iS0, len(S0_list)
        Temperature = Forcing2(0, t0, t1, tk, tl, Tbar0, A_B[iAB])
        for case in xrange(2):
            if case == 0:
                ode15s = ode(SystemForcing1)
                ode15s.set_f_params(Temperature, beta, gammaH, sigma1, sigma2, sigma3, S00_i, N, cmax)
            else:
                ode15s = ode(SystemForcing2)
                ode15s.set_f_params(Temperature, beta, gammaH, sigma1, sigma2, sigma3, S00_i, N, cmax, rho)

            # Introducing temperature dependence, environmental temperature forcing
            ode15s.set_integrator('vode', method='bdf', order=15, nsteps=3000)
            ode15s.set_initial_value(Initial, 0)
            Dynamics = zeros((len(Time), len(Initial)))
            Dynamics[0, 0] = ode15s.y[0]
            Dynamics[0, 1] = ode15s.y[1]
            Dynamics[0, 2] = ode15s.y[2]
            Dynamics[0, 3] = ode15s.y[3]
            Dynamics[0, 4] = ode15s.y[4]
            Dynamics[0, 5] = ode15s.y[5]
            Dynamics[0, 6] = ode15s.y[6]
        
            k=1
            while ode15s.successful() and k<len(Time):#ode15s.t <= max(Time)-step+1:
                #print len(Time), k, max(Time) - step
                ode15s.integrate(ode15s.t+step)
                Dynamics[k, 0] = ode15s.y[0]
                Dynamics[k, 1] = ode15s.y[1]
                Dynamics[k, 2] = ode15s.y[2]
                Dynamics[k, 3] = ode15s.y[3]
                Dynamics[k, 4] = ode15s.y[4]
                Dynamics[k, 5] = ode15s.y[5]
                Dynamics[k, 6] = ode15s.y[6]
                # Updating environmental temperature forcing
                Temperature = Forcing2(ode15s.t+step, t0, t1, tk, tl, Tbar0, A_B[iAB])
                if case == 0:
                    ode15s.set_f_params(Temperature, beta, gammaH, sigma1, sigma2, sigma3, S00_i, N, cmax)
                if case == 1:
                    ode15s.set_f_params(Temperature, beta, gammaH, sigma1, sigma2, sigma3, S00_i, N, cmax, rho)
                k+=1
            
            #traj 
            
            if tb == 0 and iS0 <= 10:
                if case == 0:
                    sub0 = fig2.add_subplot(221)
                    sub0.set_title("%d"%Tbar0+u"\N{DEGREE SIGN}C", fontsize = fsizeT, fontproperties = font)
                    sub0.plot(Time, (Dynamics[:,6]/Dynamics[:,0])/1e6, linewidth = lw_case1, color = "#fbceb1")
                    sub0.plot(Time, (Dynamics[:,5]/Dynamics[:,0])/1e6,linewidth = lw_case1, color = "#d99058")
                    sub0.plot(Time, (Dynamics[:,4]/Dynamics[:,0])/1e6,linewidth = lw_case1, color = "#964b00")
                if case == 1:
                    sub0 = fig2. add_subplot(223)
                    sub0.plot(Time, (Dynamics[:,6]/Dynamics[:,0])/1e6, linewidth = lw_case2, color = "#fbceb1")
                    sub0.plot(Time, (Dynamics[:,5]/Dynamics[:,0])/1e6,linewidth = lw_case2, color = "#d99058")
                    sub0.plot(Time, (Dynamics[:,4]/Dynamics[:,0])/1e6,linewidth = lw_case2, color = "#964b00")
            if tb == 1 and iS0 <= 10:
                if case == 0:
                    sub0 = fig2.add_subplot(222)
                    sub0.set_title("%d"%Tbar0+u"\N{DEGREE SIGN}C", fontsize = fsizeT, fontproperties = font)
                    sub0.plot(Time, (Dynamics[:,6]/Dynamics[:,0])/1e6, linewidth = lw_case1, color = "#fbceb1")
                    sub0.plot(Time, (Dynamics[:,5]/Dynamics[:,0])/1e6,linewidth = lw_case1, color = "#d99058")
                    sub0.plot(Time, (Dynamics[:,4]/Dynamics[:,0])/1e6,linewidth = lw_case1, color = "#964b00")
                if case == 1:
                    sub0 = fig2. add_subplot(224)
                    sub0.plot(Time, (Dynamics[:,6]/Dynamics[:,0])/1e6, linewidth = lw_case2, color = "#fbceb1")
                    sub0.plot(Time, (Dynamics[:,5]/Dynamics[:,0])/1e6,linewidth = lw_case2, color = "#d99058")
                    sub0.plot(Time, (Dynamics[:,4]/Dynamics[:,0])/1e6,linewidth = lw_case2, color = "#964b00")
            
            # Find the time when the density of more adapted symbiont becomes lower than the density of the other symbiont
            # knowing that this occurs for this particula temperature forcing and parameters
            if Tbar0 == 26:
                # testing for occurence of shuffling, it happens if one of the test is true, therefore a +
                Test = (Dynamics[:, 6] < Dynamics[:, 5])+(Dynamics[:, 6] < Dynamics[:, 4])
            elif Tbar0 == 27:
                Test = (Dynamics[:, 5] < Dynamics[:, 4])+(Dynamics[:, 5] < Dynamics[:, 6])
                
            if case == 0:
                if len(Time[Test]) !=0:
                    ShiftDomTiming1[iS0] = Time[Test][0] # take the first time the test is true
                    # testing for recovery after temperature returen to baseline, it happens only if all tests are true, therefore *
                    if Tbar0 == 26:
                        Test1 = (Dynamics[Time>=t1, 6] > Dynamics[Time>=t1, 5])*(Dynamics[Time>=t1, 6] > Dynamics[Time>=t1, 4])
                        if sum(Test1)!=0:
                            RecovTiming1[iS0] = Time[Time>=t1][Test1][0]
                        else:
                            RecovTiming1[iS0] = 1e6 # some impossible number
                    elif Tbar0 == 27:
                        Test1 = (Dynamics[Time>=t1, 5] > Dynamics[Time>=t1, 4])*(Dynamics[Time>=t1, 5] > Dynamics[Time>=t1, 6])
                        if sum(Test1)!=0:
                            RecovTiming1[iS0] = Time[Time>=t1][Test1][0]
                        else:
                            RecovTiming1[iS0] = 1e6 # some impossible number
                else:
                    ShiftDomTiming1[iS0] = 1e6 # some impossible number 
                    RecovTiming1[iS0] = 1e6
            else:
                if len(Time[Test]) !=0:
                    ShiftDomTiming2[iS0] = Time[Test][0] # take the first time the test is true    
                    # Testing for recovery , it happens only if all tests are true, therefore *
                    if Tbar0 == 26:
                        Test1 = (Dynamics[Time>=t1, 6] > Dynamics[Time>=t1, 5])*(Dynamics[Time>=t1, 6] > Dynamics[Time>=t1, 4])
                        if sum(Test1)!=0:
                            RecovTiming2[iS0] = Time[Time>=t1][Test1][0]
                        else:
                            RecovTiming2[iS0] = 1e6 # some impossible number
                    elif Tbar0 == 27:
                        Test1 = (Dynamics[Time>=t1, 5] > Dynamics[Time>=t1, 4])*(Dynamics[Time>=t1, 5] > Dynamics[Time>=t1, 6])
                        if sum(Test1)!=0:
                            RecovTiming2[iS0] = Time[Time>=t1][Test1][0]
                        else:
                            RecovTiming2[iS0] = 1e6 # some impossible number
                else:
                    ShiftDomTiming2[iS0] = 1e6  # some impossible number 
                    RecovTiming2[iS0] = 1e6                        
    ShiftList1.append(ShiftDomTiming1)
    ShiftList2.append(ShiftDomTiming2)
    RecovList1.append(RecovTiming1)
    RecovList2.append(RecovTiming2)

# saving files
fileShift1 = open("TimingData/Shift_H1.dat", "wr")
fileShift2 = open("TimingData/Shift_H2.dat", "wr")
fileRecov1 = open("TimingData/Recov_H1.dat", "wr")
fileRecov2 = open("TimingData/Recov_H2.dat", "wr")

array(ShiftList1).dump(fileShift1)
fileShift1.flush()
array(ShiftList2).dump(fileShift2)
fileShift2.flush()
array(RecovList1).dump(fileRecov1)
fileRecov1.flush()
array(RecovList2).dump(fileRecov2)
fileRecov2.flush()
"""

""" Figure 5 """
# Plotting saved file (# Sensitivity of timing of symbiont shuffling with respect to S_0, Using temperature forcing 2 displayed in the 1D figures)

fig = plt.figure(figsize = (17, 9))
#plt.subplots_adjust(bottom = 0.1, right = 0.85, left = 0.10, top = 0.95, wspace = 0.1, hspace = 0.1) # small screen  for poportion of carrying capacity on the x-axis
#plt.subplots_adjust(bottom = 0.1, right = 0.85, left = 0.10, top = 0.95, wspace = 0.15, hspace = 0.1) # small screen  for million cell/cm2 per month

#plt.subplots_adjust(bottom = 0.25, right = 0.70, left = 0.10, top = 0.95, wspace = 0.1, hspace = 0.1) # large screen           

h1col = "blue"
h2col = "orange"

fileShift1 = open("TimingData-tmax10yrs/Shift_H1.dat", "r")
fileShift2 = open("TimingData-tmax10yrs/Shift_H2.dat", "r")
fileRecov1 = open("TimingData-tmax10yrs/Recov_H1.dat", "r")
fileRecov2 = open("TimingData-tmax10yrs/Recov_H2.dat", "r")

ShiftDomTimingList1 = load(fileShift1, allow_pickle = True)
fileShift1.close()
ShiftDomTimingList2 = load(fileShift2, allow_pickle = True)
fileShift2.close()
RecovTimingList1 = load(fileRecov1, allow_pickle = True)
fileRecov1.close()
RecovTimingList2 = load(fileRecov2, allow_pickle = True)
fileRecov2.close()
count = 1
Marksiz = 50

maxprop = 1 # could go up to 2
maxMonth1 = 15 # maximum month to observe shuffling
maxMonth2 = 40 # maximum month to observe recovery
for tb in xrange(len(Tbar0_list)): 
    DL = dl_list[tb]  
    iAB = iAB_list[tb]                              
    t1 = t0 + dur_list[DL]
    IndexInit = IndexInit_list[tb]  
    Tbar0 = Tbar0_list[tb]                                                                            
    print IndexInit
    i1 = IndexInit[0]
    i2 = IndexInit[1]
    i3 = IndexInit[2]
    symb1 = symb1_list[i1] 
    symb2 = symb2_list[i2]
    symb3 = symb3_list[i3]
    Initial = array([0.75*K_C, 0.0000005, 0.0000005, 0.0000005, symb1*Ksmax*0.75*K_C, symb2*Ksmax*0.75*K_C, symb3*0.75*Ksmax*K_C])
    Time = arange(0, t0+tmax+step, step)
    
    ShiftDomTiming1 = ShiftDomTimingList1[tb]
    ShiftDomTiming2 = ShiftDomTimingList2[tb]
    RecovTiming1 = RecovTimingList1[tb]
    RecovTiming2 = RecovTimingList2[tb]
                  
    sub1 = fig.add_subplot(2, 2, count)
    sub2 = fig.add_subplot(2, 2, count+2)
    
    notIgnore1a = ShiftDomTiming1 != 1e6
    notIgnore2a = ShiftDomTiming2 != 1e6   

    notIgnore1b = (RecovTiming1!=1e6)
    notIgnore2b = (RecovTiming2!=1e6) 
    
    
    # plot scatter points open filled circle for H1, circle for H2
    sub1.scatter(S0_list[notIgnore1a], ShiftDomTiming1[notIgnore1a]-t0, s = Marksiz, color = h1col, label = "H1")
    sub1.scatter(S0_list[notIgnore2a], ShiftDomTiming2[notIgnore2a]-t0, s = Marksiz, color= h2col, label = "H2")
    if tb == 0:
        sub1.legend(fontsize = fsize)
    
    sub2.scatter(S0_list[notIgnore1b], RecovTiming1[notIgnore1b]-t1, s = Marksiz, color = h1col)
    sub2.scatter(S0_list[notIgnore2b], RecovTiming2[notIgnore2b]-t1, s = Marksiz, color= h2col)

        
    sub1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 10, labelsize = fsize)
    sub2.tick_params(labelbottom = True, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 10, labelsize = fsize)
    
    if count == 1:
        sub1.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 5, labelsize = fsize)
        sub2.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 5, labelsize = fsize)

    if count == 2:
        sub1.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 5, labelsize = fsize)
        sub2.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 5, labelsize = fsize)
 
    #sub1.set_xticks(arange(min(S0_list), max(S0_list)+0.1, 0.1))
    #sub1.set_xticklabels(["%.1f"%s0 for s0 in arange(min(S0_list), max(S0_list)+0.1, 0.1)]) 
    sub1.set_xticks(arange(0, maxprop+0.1, 0.1)[::2])
    #sub1.set_xticklabels(["%.1f"%(s0*Ksmax/(1e6) for s0 in arange(0, maxprop+0.1, 0.1)]) 
    sub2.set_xticks(arange(0, maxprop+0.1, 0.1)[::2])
    sub2.set_xticklabels(["%.1f"%(s0*Ksmax/(1e6)) for s0 in arange(0, maxprop+0.1, 0.1)]) # converting s0 to million cells/cm2 per year to for consitency with manuscript text
    
    maxTa = max(max(ShiftDomTiming1[notIgnore1a])-t0, max(ShiftDomTiming2[notIgnore2a])-t0)
    minTa = min(min(ShiftDomTiming1[notIgnore1a])-t0, min(ShiftDomTiming2[notIgnore2a])-t0)
    
    if sum(notIgnore1b)!=0 and sum(notIgnore2b)!=0:
        maxTb = max(max(RecovTiming1[notIgnore1b])-t1, max(RecovTiming2[notIgnore2b])-t1)
        minTb = min(min(RecovTiming1[notIgnore1b])-t1, min(RecovTiming2[notIgnore2b])-t1)
    elif sum(notIgnore1b)!=0:
        maxTb = max(RecovTiming1[notIgnore1b])-t1
        mintTb =min(RecovTiming1[notIgnore1b])-t1
    elif sum(notIgnore2b)!=0:
        maxTb = max(RecovTiming2[notIgnore2b])-t1
        mintTb =min(RecovTiming2[notIgnore2b])-t1
        
    sub1.set_ylim((0, maxMonth1/12))#((minTa-0.1, maxTa+0.1))
    sub1.set_xlim((0, maxprop))#((min(S0_list), max(S0_list)))
    
    sub2.set_ylim((0, maxMonth2/12))#((minTb-0.1, maxTb+0.1))
    sub2.set_xlim((0, maxprop))#((min(S0_list), max(S0_list)))
    
    #sub1.plot(S00*ones(5), linspace(minTa-0.1, maxTa+0.1, 5), color = "green", linewidth = 2) # indicating the value used in main results
    #sub2.plot(S00*ones(5), linspace(minTb-0.1, maxTb+0.1, 5), color = "green", linewidth = 2) # indicating the value used in main results
    
    sub1.plot(S00*ones(5), linspace(0, maxMonth1/12, 5),"--", color = "black", linewidth = 2) # indicating the value used in main results
    sub2.plot(S00*ones(5), linspace(0, maxMonth2/12, 5),"--", color = "black", linewidth = 2, label = "$S_0$ for Fig 3-4") # indicating the value used in main results
    
    if tb == 0:
        sub2.legend(fontsize = fsize - 2)
    
    sub1.set_yticks(arange(0, maxMonth1/12 + 1/12, 1/12))#(linspace(minTa-0.1, maxTa+0.1, 10))
    sub1.set_yticklabels(["%d"%temp for temp in arange(0, maxMonth1 + 1, 1)])
    
    sub2.set_yticks(arange(0, maxMonth2/12 + 1/12, 5/12))#(linspace(minTb-0.1, maxTb+0.1, 10))
    sub2.set_yticklabels(["%d"%temp for temp in arange(0, maxMonth2 + 1, 5)])
    if count == 1:
        #sub1.set_ylabel("Shift of symbiont dominance\nfrom the start of thermal shift\n(months) ", fontsize = fsizeT, labelpad = 8) #labelpad= 9)
        sub1.set_ylabel("Symbiont shuffling from\nonset of temperature shift\n(month) ", fontsize = fsizeT, labelpad = 8)
        #sub2.set_ylabel("Recovery of initial symbiont\nfrom the end of thermal shift\n(months) ", fontsize = fsizeT, labelpad = 1) #labelpad= 9)
        sub2.set_ylabel("Symbiont recovery from\nend of temperature shift\n(month) ", fontsize = fsizeT, labelpad = 1) #labe
    if count == 2:
        #sub2.text(S0_list[0]-0.5, minTb - 1.2, "Rate of symbiont sustainance $S_0$ (year$^{-1}$)", fontsize = fsizeT)
        #sub2.text(0, 0 - 42/12, "Non-density dependent growth ($S_0$)\nallocated to collapsing symbiont population\n(proportion of carrying capacity per year)", horizontalalignment = "center", fontsize = fsizeT)
        #sub2.text(0, 0 - 40/12, "Non-density dependent growth ($S_0$)\nallocated to collapsing symbiont population\n(proportion of carrying capacity per year)", horizontalalignment = "center", fontsize = fsizeT)
        #sub2.text(0, 0 - 19/12, "Non-density dependent growth ($S_0$)\nallocated to collapsing symbiont population\n(million cells cm$^{-2}$ month$^{-1}$)", horizontalalignment = "center", fontsize = fsizeT)
        sub2.text(0, 0-8/12, r"$S_0$ ($\times 10^{6}$ cells cm$^{-2}$ month$^{-1}$)", horizontalalignment = "center", fontsize = fsizeT)
    sub1.set_title("%d"%Tbar0+u"\N{DEGREE SIGN}C", fontsize = fsizeT, fontproperties = font)

    if count == 1:
        #sub1.text(S0_list[0]-0.20, maxTa+0.1, Lab1[tb], fontproperties = font, fontsize = fsize)
        #sub2.text(S0_list[0]-0.20, maxTb+0.1, Lab2[tb], fontproperties = font, fontsize = fsize)
        sub1.text(S0_list[0]-0.155, maxMonth1/12, Lab1[tb], fontproperties = font, fontsize = fsize)
        sub2.text(S0_list[0]-0.155, maxMonth2/12, Lab2[tb], fontproperties = font, fontsize = fsize)
    else:
        #sub1.text(S0_list[0]-0.05, maxTa+0.1, Lab1[tb], fontproperties = font, fontsize = fsize)
        #sub2.text(S0_list[0]-0.05, maxTb+0.1, Lab2[tb], fontproperties = font, fontsize = fsize)
        sub1.text(S0_list[0]-0.05, maxMonth1/12, Lab1[tb], fontproperties = font, fontsize = fsize)
        sub2.text(S0_list[0]-0.05, maxMonth2/12, Lab2[tb], fontproperties = font, fontsize = fsize)
    count += 1 
    
    """
    if tb == 0:
        sub1.annotate(s='', xy=(1,0.65), xytext=(0.25,0.65), arrowprops=dict(arrowstyle='<->'), color = h1col)
        sub2.annotate(s='', xy=(1,1.4), xytext=(0.225,1.4), arrowprops=dict(arrowstyle='<->'), color = h1col)
        
        sub1.annotate(s='', xy=(1,0.7), xytext=(0.07,0.7), arrowprops=dict(arrowstyle='<->', color = h2col))
        sub2.annotate(s='', xy=(0.06,1.4), xytext=(0.15,1.4), arrowprops=dict(arrowstyle='<->', color = h2col))
    
    else:
        sub1.annotate(s='', xy=(1,0.2), xytext=(0.2,0.2), arrowprops=dict(arrowstyle='<->'), color = h1col)
        sub2.annotate(s='', xy=(0.15,1.7), xytext=(0.25,1.7), arrowprops=dict(arrowstyle='<->'), color = h1col)
        sub2.annotate(s='', xy=(1,1.7), xytext=(0.725,1.7), arrowprops=dict(arrowstyle='<->'), color = h1col)

        sub1.annotate(s='', xy=(1,0.1), xytext=(0.07,0.1), arrowprops=dict(arrowstyle='<->', color = h2col))
    """
#Separately
#plt.subplots_adjust(bottom = 0.5, right = 0.80, left = 0.10, top = 0.90, wspace = 0.1, hspace = 0.15) # big screen            
#plt.subplots_adjust(bottom = 0.5, right = 0.60, left = 0.10, top = 0.90, wspace = 0.1, hspace = 0.15) # small screen           

#Together
plt.subplots_adjust(bottom = 0.1, right = 0.60, left = 0.10, top = 0.95, wspace = 0.1, hspace = 0.15)            


plt.savefig("Figures/Fig5.pdf", bbox_inches = 'tight')

       
"""
Sensitivity bar plot with respect to the standard run which is the run under constant temperature over a 2 year period and the default parameters. 
"""

"""
# Time and initial conditions params
step = 0.001
tmax = 2
Time = arange(0, tmax+step, step)

# Initial conditions
symb1_list = array([0.1, 0.8]) # proportion of symbiont1 to be studied
symb2_list = array([0.1, 0.8]) # proportion of symbiont2 to be studied
symb3_list = array([0.1, 0.8]) # proportion of symbiont3 to be studied

# init symb biomass index low = 1 , high = 2 
low = 0
high = 1

# symbiont more adapted to mean temperature Tbar0 is the more abundant at initial condition     
IndexInit_26 = (low, low, high)  # for Tbar0 = 26
IndexInit_27 = (low, high, low)  # for Tbar0 = 27
IndexInit_28 = (high, low, low) # for Tbar0 = 28
IndexInit_list = [IndexInit_26, IndexInit_27, IndexInit_28] 

StudiedParams = [r"$\beta$", "$\gamma_H$", "$\sigma_1$", "$\sigma_2$", "$\sigma_3$", "$S_0$", "$N$", "$c_{max}$", r"$\rho$"]

Tbar0_list = [26, 27, 28]
fsize = 14
fsizeT = 16    
fig = plt.figure(figsize = (12, 9))

# fig labels
Lab = [("a", "b"), ("c", "d"), ("e", "f")]

count = 1
for tb in xrange(len(Tbar0_list)):
    Tbar0 = Tbar0_list[tb]
    IndexInit = IndexInit_list[tb]  
    i1 = IndexInit[0]
    i2 = IndexInit[1]
    i3 = IndexInit[2]
    symb1 = symb1_list[i1] 
    symb2 = symb2_list[i2]
    symb3 = symb3_list[i3]
    Initial = array([0.75*K_C, 0.0000005, 0.0000005, 0.0000005, symb1*Ksmax*0.75*K_C, symb2*Ksmax*0.75*K_C, symb3*0.75*Ksmax*K_C])
    # Standard Run for H 1
    ode15s_1 = ode(SystemForcing1)
    ode15s_1.set_f_params(Tbar0, beta, gammaH, sigma1, sigma2, sigma3, S00, N, cmax)
    ode15s_1.set_integrator('vode', method='bdf', order=15, nsteps=3000)
    ode15s_1.set_initial_value(Initial, 0)
    Standard_1 = zeros((len(Time), len(Initial)))
    Standard_1[0, 0] = ode15s_1.y[0]
    Standard_1[0, 1] = ode15s_1.y[1]
    Standard_1[0, 2] = ode15s_1.y[2]
    Standard_1[0, 3] = ode15s_1.y[3]
    Standard_1[0, 4] = ode15s_1.y[4]
    Standard_1[0, 5] = ode15s_1.y[5]
    Standard_1[0, 6] = ode15s_1.y[6]
    
    k=1
    while ode15s_1.successful() and k<len(Time):
        #print len(Time), k, max(Time) - step
        ode15s_1.integrate(ode15s_1.t+step)
        Standard_1[k, 0] = ode15s_1.y[0]
        Standard_1[k, 1] = ode15s_1.y[1]
        Standard_1[k, 2] = ode15s_1.y[2]
        Standard_1[k, 3] = ode15s_1.y[3]
        Standard_1[k, 4] = ode15s_1.y[4]
        Standard_1[k, 5] = ode15s_1.y[5]
        Standard_1[k, 6] = ode15s_1.y[6]
        k+=1

    # Standard Run for H_2
    ode15s_2 = ode(SystemForcing2)
    ode15s_2.set_f_params(Tbar0, beta, gammaH, sigma1, sigma2, sigma3, S00, N, cmax, rho)
    ode15s_2.set_integrator('vode', method='bdf', order=15, nsteps=3000)
    ode15s_2.set_initial_value(Initial, 0)
    Standard_2 = zeros((len(Time), len(Initial)))
    Standard_2[0, 0] = ode15s_2.y[0]
    Standard_2[0, 1] = ode15s_2.y[1]
    Standard_2[0, 2] = ode15s_2.y[2]
    Standard_2[0, 3] = ode15s_2.y[3]
    Standard_2[0, 4] = ode15s_2.y[4]
    Standard_2[0, 5] = ode15s_2.y[5]
    Standard_2[0, 6] = ode15s_2.y[6]
    
    k=1
    while ode15s_2.successful() and k<len(Time):#ode15s.t <= max(Time)-step+1:
        #print len(Time), k, max(Time) - step
        ode15s_2.integrate(ode15s_2.t+step)
        Standard_2[k, 0] = ode15s_2.y[0]
        Standard_2[k, 1] = ode15s_2.y[1]
        Standard_2[k, 2] = ode15s_2.y[2]
        Standard_2[k, 3] = ode15s_2.y[3]
        Standard_2[k, 4] = ode15s_2.y[4]
        Standard_2[k, 5] = ode15s_2.y[5]
        Standard_2[k, 6] = ode15s_2.y[6]
        k+=1
    # Run +25% or -25% changes in the Studied Parameters for H_1, rho is not a parameter for H_1 so only len(StudiedParams) - 1  
    # Taking last value of each symbiont biomass as sensitivity measure
    Changes_plus_1 = zeros((len(StudiedParams) - 1, 3)) 
    Changes_minus_1 = zeros((len(StudiedParams) - 1, 3)) 
    # Run +25% or -25% changes in the Studied Parameters for H_2, rho is included
    # Taking last value of each symbiont biomass as sensitivity measure
    Changes_plus_2 = zeros((len(StudiedParams), 3)) 
    Changes_minus_2 = zeros((len(StudiedParams), 3)) 
    
    for case in xrange(2):
        if case == 0:
            lenS = len(StudiedParams) - 1
            SystemForcing = SystemForcing1
            Changes_plus = Changes_plus_1
            Changes_minus = Changes_minus_1
            Standard = Standard_1
        else:
            lenS = len(StudiedParams)
            SystemForcing = SystemForcing2
            Changes_plus = Changes_plus_2
            Changes_minus = Changes_minus_2
            Standard = Standard_2
        for i in xrange(lenS):
            print Tbar0, case, StudiedParams[i]
            
            # Determine the parameter to be changed
            Bool_beta = (StudiedParams[i] == r"$\beta$")*beta
            Bool_gammaH = (StudiedParams[i] == "$\gamma_H$")*gammaH
            Bool_sigma1 = (StudiedParams[i] == "$\sigma_1$")*sigma1
            Bool_sigma2 = (StudiedParams[i] == "$\sigma_2$")*sigma2
            Bool_sigma3 = (StudiedParams[i] == "$\sigma_3$")*sigma3
            Bool_S00 = (StudiedParams[i] == "$S_0$")*S00
            Bool_N = (StudiedParams[i] == "$N$")*N
            Bool_cmax = (StudiedParams[i] == "$c_{max}$")*cmax
            ode15sA = ode(SystemForcing)
            if SystemForcing == SystemForcing1:
                ode15sA.set_f_params(Tbar0, beta + 0.25*Bool_beta, gammaH + 0.25*Bool_gammaH, sigma1 + 0.25*Bool_sigma1, sigma2 + 0.25*Bool_sigma2, sigma3 + 0.25*Bool_sigma3, S00 + 0.25*Bool_S00, N + 0.25*Bool_N, cmax + 0.25*Bool_cmax)
            else:
                Bool_rho = (StudiedParams[i] == r"$\rho$")*rho
                ode15sA.set_f_params(Tbar0, beta + 0.25*Bool_beta, gammaH + 0.25*Bool_gammaH, sigma1 + 0.25*Bool_sigma1, sigma2 + 0.25*Bool_sigma2, sigma3 + 0.25*Bool_sigma3, S00 + 0.25*Bool_S00, N + 0.25*Bool_N, cmax + 0.25*Bool_cmax, rho + 0.25*Bool_rho)

                
            ode15sA.set_integrator('vode', method='bdf', order=15, nsteps=3000)
            ode15sA.set_initial_value(Initial, 0)
            Changed_a = zeros((len(Time), len(Initial)))
            Changed_a[0, 0] = ode15sA.y[0]
            Changed_a[0, 1] = ode15sA.y[1]
            Changed_a[0, 2] = ode15sA.y[2]
            Changed_a[0, 3] = ode15sA.y[3]
            Changed_a[0, 4] = ode15sA.y[4]
            Changed_a[0, 5] = ode15sA.y[5]
            Changed_a[0, 6] = ode15sA.y[6]
            k=1
            while ode15sA.successful() and k<len(Time):
                #print len(Time), k, max(Time) - step
                ode15sA.integrate(ode15sA.t+step)
                Changed_a[k, 0] = ode15sA.y[0]
                Changed_a[k, 1] = ode15sA.y[1]
                Changed_a[k, 2] = ode15sA.y[2]
                Changed_a[k, 3] = ode15sA.y[3]
                Changed_a[k, 4] = ode15sA.y[4]
                Changed_a[k, 5] = ode15sA.y[5]
                Changed_a[k, 6] = ode15sA.y[6]
                k+=1
            range0 = Time>=1
            Changes_plus[i, 0] = 100*(mean(Changed_a[range0, 6]) - mean(Standard[range0, 6]))/mean(Standard[range0, 6])         # for low T° tolerant symbiont
            Changes_plus[i, 1] = 100*(mean(Changed_a[range0, 5]) - mean(Standard[range0, 5]))/mean(Standard[range0, 5])        # for mid T° tolerant symbiont
            Changes_plus[i, 2] = 100*(mean(Changed_a[range0, 4]) - mean(Standard[range0, 4]))/mean(Standard[range0, 4])        # for high T° tolerant symbiont
            # Run for -25% change
            ode15sB = ode(SystemForcing)
            if SystemForcing == SystemForcing1:
                ode15sB.set_f_params(Tbar0, beta - 0.25*Bool_beta, gammaH - 0.25*Bool_gammaH, sigma1 - 0.25*Bool_sigma1, sigma2 - 0.25*Bool_sigma2, sigma3 - 0.25*Bool_sigma3, S00 - 0.25*Bool_S00, N - 0.25*Bool_N, cmax - 0.25*Bool_cmax)
            else:
                Bool_rho = (StudiedParams[i] == r"$\rho$")
                ode15sB.set_f_params(Tbar0, beta - 0.25*Bool_beta, gammaH - 0.25*Bool_gammaH, sigma1 - 0.25*Bool_sigma1, sigma2 - 0.25*Bool_sigma2, sigma3 - 0.25*Bool_sigma3, S00 - 0.25*Bool_S00, N - 0.25*Bool_N, cmax - 0.25*Bool_cmax, rho - 0.25*Bool_rho)
            
            ode15sB.set_integrator('vode', method='bdf', order=15, nsteps=3000)
            ode15sB.set_initial_value(Initial, 0)
            Changed_b = zeros((len(Time), len(Initial)))
            Changed_b[0, 0] = ode15sB.y[0]
            Changed_b[0, 1] = ode15sB.y[1]
            Changed_b[0, 2] = ode15sB.y[2]
            Changed_b[0, 3] = ode15sB.y[3]
            Changed_b[0, 4] = ode15sB.y[4]
            Changed_b[0, 5] = ode15sB.y[5]
            Changed_b[0, 6] = ode15sB.y[6]
            
            k=1
            while ode15sB.successful() and k<len(Time):
                #print len(Time), k, max(Time) - step
                ode15sB.integrate(ode15sB.t+step)
                Changed_b[k, 0] = ode15sB.y[0]
                Changed_b[k, 1] = ode15sB.y[1]
                Changed_b[k, 2] = ode15sB.y[2]
                Changed_b[k, 3] = ode15sB.y[3]
                Changed_b[k, 4] = ode15sB.y[4]
                Changed_b[k, 5] = ode15sB.y[5]
                Changed_b[k, 6] = ode15sB.y[6]
                k+=1
            
            Changes_minus[i, 0] = 100*(mean(Changed_b[range0, 6]) - mean(Standard[range0, 6]))/mean(Standard[range0, 6])    # for low T° tolerant symbiont
            Changes_minus[i, 1] = 100*(mean(Changed_b[range0, 5]) - mean(Standard[range0, 5]))/mean(Standard[range0, 5])    # for mid T° tolerant symbiont
            Changes_minus[i, 2] = 100*(mean(Changed_b[range0, 4]) - mean(Standard[range0, 4]))/mean(Standard[range0, 4])    # for high T° tolerant symbiont
            
    # Plotting
    sub1 = plt.subplot(3, 2, count)
    #plt.text(-17, 40, "%d"%Tbar0+u"\N{DEGREE SIGN}C", fontsize = fsizeT, fontproperties = font, rotation = "vertical")
    if tb == 0:
        plt.text(2-1, 140, "$S_1$", color = "#fbceb1", fontsize = fsize)
        plt.text(8-1, 140, "$S_2$", color = "#d99058", fontsize = fsize)
        plt.text(14-1, 140, "$S_3$", color = "#964b00", fontsize = fsize)
        sub1.add_patch(Polygon([[2-1, 80+40], [2-1, 90+40], [6-1, 90+40], [6-1, 80+40]], closed=True, fill=True, color = "black"))
        sub1.add_patch(Polygon([[2-1, 60+40], [2-1, 70+40], [6-1, 70+40], [6-1, 60+40]], closed=True, fill=False))
        plt.text(6.5-1, 80+40, "$+25\%$", fontsize = fsize)
        plt.text(6.5-1, 60+40, "$-25\%$", fontsize = fsize)
    
    sub2 = plt.subplot(3, 2, count+1)
    NumP1 = linspace(1.5, 80, len(StudiedParams)-1) 
    NumP2 = linspace(1.5, 80, len(StudiedParams)) 
    width = 1.5
    
    sub1.bar(NumP1, Changes_plus_1[:, 0], width, color = "#fbceb1")
    sub1.bar(NumP1+width, Changes_minus_1[:, 0], width, fill = False, edgecolor = "#fbceb1") 

    sub1.bar(NumP1+2*width, Changes_plus_1[:, 1], width, color = "#d99058")
    sub1.bar(NumP1+3*width, Changes_minus_1[:, 1], width, fill = False, edgecolor = "#d99058")

    sub1.bar(NumP1+4*width, Changes_plus_1[:, 2], width, color = "#964b00")
    sub1.bar(NumP1+5*width, Changes_minus_1[:, 2], width, fill = False, edgecolor = "#964b00")
    
    sub2.bar(NumP2, Changes_plus_2[:, 0], width, color = "#fbceb1")
    sub2.bar(NumP2+width, Changes_minus_2[:, 0], width, fill = False, edgecolor = "#fbceb1") 

    sub2.bar(NumP2+2*width, Changes_plus_2[:, 1], width, color = "#d99058")
    sub2.bar(NumP2+3*width, Changes_minus_2[:, 1], width, fill = False, edgecolor = "#d99058")

    sub2.bar(NumP2+4*width, Changes_plus_2[:, 2], width, color = "#964b00")
    sub2.bar(NumP2+5*width, Changes_minus_2[:, 2], width, fill = False, edgecolor = "#964b00")
    
    sub1.set_ylim(-40, 160)
    sub2.set_ylim(-40, 160)
    
    if count in (1, 3):
        sub1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        sub2.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
    if count == 5:
        sub1.tick_params(labelbottom = True, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        sub2.tick_params(labelbottom = True, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
     
    sub1.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
    sub2.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
          
    sub1.set_xticks(NumP1+2.5*width)
    sub1.set_xticklabels(StudiedParams[0:(len(StudiedParams)-1)])
    sub2.set_xticks(NumP2+2.5*width)
    sub2.set_xticklabels(StudiedParams)
    
    sub1.set_yticks(arange(-40, 160+20, 20))
    sub1.set_yticklabels(arange(-40, 160+20, 20))
    sub2.set_yticks(arange(-40, 160+20, 20))
    sub2.set_yticklabels(arange(-40, 160+20, 20))
    
    if count == 3:
        sub1.set_ylabel("$\%$ change in mean symbiont biomass", fontsize = fsizeT, labelpad= 13)
 
    if count == 1:
        sub1.set_title("H1", fontsize = fsizeT, fontproperties = font)
        sub2.set_title("H2", fontsize = fsizeT, fontproperties = font)
    
    Line1 = linspace(0, max(NumP1) + 5*width)
    Line2 = linspace(0, max(NumP2) + 5*width)
    
    sub1.plot(Line1, zeros(len(Line1)),  linewidth = 0.5, color = "black")
    sub2.plot(Line2, zeros(len(Line2)),  linewidth = 0.5, color = "black")
    
    sub1.text(min(NumP1)-16, 160, Lab[tb][0], fontproperties = font, fontsize = fsize)
    sub2.text(min(NumP2)-9, 160, Lab[tb][1], fontproperties = font, fontsize = fsize)
    
    count += 2 
plt.subplots_adjust(bottom = 0.05, right = 0.95, left = 0.1, top = 0.90, wspace = 0.05, hspace = 0.10)

plt.savefig("Figures/SuppFig_barplot.pdf", bbox_inches = 'tight')
"""




