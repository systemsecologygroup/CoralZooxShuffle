# -*- coding: utf-8 -*-
from __future__ import division

from scipy import exp, linspace, array, zeros, e, sqrt, mean,var, ones, cumsum, random, sin, pi, load, floor, ceil
from scipy.stats import norm
from numpy import amin, amax, meshgrid, arange, isnan, logical_not, interp, concatenate, arange, isnan
from scipy.integrate import ode
from matplotlib import pyplot as plt
import matplotlib


#### Model with a dynamics of 3 types of symbiont population using odeint solver #####
G_C = 10  # this is G_max in the model, per year
a = 1.0768 # Symbiont specific growth rate - linear growth rate
b = 0.0633 # Exponential Growth constant of the symbiont

alpha = 1e-3 # slope of cost associated to investment in symbiont, in case we assume the same cost for any symbiont

M_C = 10e-3 # coral mortality 

K_C = 5125112.94092e10 # earth coral carying capacity
Ksmax = 3e6 # healthy measure of carying capacity of symbiont per host biomass
N = 1e-5 # speed of adaptation assumed to be 1e6 times faster than the value we found for MS2
         # We assume an unreasonably high capacity of the corals to acclimate to change in temperature 
         # so that the success of the holobiont uniquely depends on the symbiont contribution.
beta = 1e2

gammaH = 0.25e6 # free param 
fsize = 18

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

# proportion of symbiont carrying capacity added to the the system every year
S00 = 0.1
  
# Gradient of coral fitness, without bound 
#def Gradient(coral, u_i, u, cost_gam, Gx1, beta, K_symb, kappa_i, alpha, p_i, c_i):
#    Grad = Gx1*beta*p_i*c_i*kappa_i*(1 - coral/K_C)*exp(-beta*u_i) - r*alpha*cost_gam*exp(r*u)
#    return Grad 
 
# Gradient of coral fitness, with bound to avoid negative investment, needed here

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
def Gradient(Bound, coral, u_i, u, cost_gam, Gx1, beta, K_symb, kappa_i, alpha, delta,p_i, c_i):
    Grad = Gx1*beta*p_i*c_i*kappa_i*(1 - coral/K_C)*exp(-beta*u_i) - (Bound*(r*alpha*cost_gam*exp(r*u)) - delta*r*alpha*cost_gam*exp(r*u)*exp(-delta*u_i))
    return Grad    
         
# Temperature forcing function for first forcing, mid duration is included so that the first
# the first phase (t<t0) of each forcing are identical but t1 is not needed for Forcing1 since temperature is not recovering, I just need to have this here for consistency in the next lines of codes 
def Forcing1(t, t0, t1, tk, tl, Tbar0, AorB):
    if t<t0:
        forc1 = Tbar0 
    elif t>=t0:
        Half = tk # half saturation 
        forc1 = Tbar0 + AorB*((t - t0)**tl/(Half**tl + (t-t0)**tl))
    return forc1    
    
def Forcing2(t, t0, t1, tk, tl, Tbar0, AorB):
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

        
# System of ODE
def SystemForcing(t, y, Temperature):
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

# Simulation step and time in year after spine up
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

A_Bstr =["Height%d"%i for i in xrange(len(A_B))] # filenames for each elements in the list of deviations from Tbar0
Tbar0_list  = [26, 27, 28]


t0 = 2 # maximum year of spine up
dur_steps = 0.25/12  # steps of week i.e quater a month
dur_list = arange(0.25/12, 4+dur_steps, dur_steps) # list of duration of stress, by steps of one week (a quater of a month), minimun duration is one week 
dur_str=["Dur%d"%i for i in xrange(len(dur_list))]

"""
dl = 0 # index of duration of stress to plot
tb = 1 # Index of Tbar0 to plot
ABindex = 3#i for i in xrange(len(A_B))] # Index of termperature deviations to plot
Time = arange(0, t0+tmax+step, step)
tk = Tk[0]
tl = Tl[0]

#for dl in xrange(10):
print dl
t1 = t0 + dur_list[dl]
Tbar0 = Tbar0_list[tb] 


Forcing2A = zeros(len(Time))
Forcing1A = zeros(len(Time))

for time in xrange(len(Time)):
    Forcing1A[time] = Forcing1(Time[time], t0, t1, tk, tl, Tbar0, A_B[ABindex])
    Forcing2A[time] = Forcing2(Time[time], t0, t1, tk, tl, Tbar0, A_B[ABindex])

plt.plot(Time, Forcing1A) 
plt.plot([t0+(t1-t0)/2, t0+(t1-t0)/2], (27, 27.5), color ="blue", label = "mid")
plt.plot([t0, t0], (27, 27.5), color="red", label="$t_0$")
plt.plot([t1, t1], [27, 27.5], color ="green", label="$t1$")
plt.legend()
            
plt.plot(Time, Forcing2A)
plt.xlim((t0-0.1, t1+0.1))    
plt.show()

"""
# symbiont more adapted to mean temperature Tbar0 is the more abundant at initial condition     
IndexInit_26 = (low, low, high)  # for Tbar0 = 26
IndexInit_27 = (low, high, low)  # for Tbar0 = 27
IndexInit_28 = (high, low, low) # for Tbar0 = 28
IndexInit_list = [IndexInit_26, IndexInit_27, IndexInit_28] 


for tb in xrange(len(Tbar0_list)): 
    for dl in xrange(len(dur_list)):                                 
        t1 = t0 + dur_list[dl]
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
        for i4 in xrange(len(Tk)):
            for i5 in xrange(len(Tl)):
                tk = Tk[i4]
                tl = Tl[i5]
                for f in xrange(len(F_list)):
                    for ab in xrange(len(A_B)):
                        print (Tbar0, tb, len(Tbar0_list)), (Forc[f], len(Forc)), (dl, len(dur_list)), (ab, len(A_B)), (i4, len(Tk)), (i5,len(Tl)) 
                        AorB = A_B[ab]
                        if Forc[f] == "Forcing1" and dl == 0:  # There is not need to consider duration of stress for forcing 1 so do the run only for one value of dl and whatever value of dl is OK
                            fileCoral = open(str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+".dat", "wr")
                            fileTrait = open(str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+".dat", "wr")
                            fileSymb = open(str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+".dat", "wr")        
                            Temperature = Forcing1(0, t0, t1, tk, tl, Tbar0, AorB)
                            ode15s = ode(SystemForcing)
                            # Introducing temperature dependence, environmental temperature forcing
                            ode15s.set_f_params(Temperature)
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
                                Temperature = Forcing1(ode15s.t+step, t0, t1, tk, tl,Tbar0, AorB)
                                ode15s.set_f_params(Temperature)
                                k+=1
                            Dynamics[:, 0].dump(fileCoral)
                            array([Dynamics[:, 1], Dynamics[:, 2], Dynamics[:, 3]]).dump(fileTrait)
                            array([Dynamics[:, 4], Dynamics[:, 5], Dynamics[:, 6]]).dump(fileSymb)
                            fileCoral.flush()
                            fileTrait.flush()
                            fileSymb.flush()
                            fileCoral.close()
                            fileTrait.close()
                            fileSymb.close()
                        
                        elif Forc[f] == "Forcing2":
                            fileCoral = open(str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+dur_str[dl]+".dat", "wr")
                            fileTrait = open(str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+dur_str[dl]+".dat", "wr")
                            fileSymb = open(str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+dur_str[dl]+".dat", "wr")        
                            Temperature = Forcing2(0, t0, t1, tk, tl, Tbar0, AorB)
                            ode15s = ode(SystemForcing)
                            # Introducing temperature dependence, environmental temperature forcing
                            ode15s.set_f_params(Temperature)
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
                                Temperature = Forcing2(ode15s.t+step, t0, t1, tk, tl, Tbar0, AorB)
                                ode15s.set_f_params(Temperature)
                                k+=1
                            Dynamics[:, 0].dump(fileCoral)
                            array([Dynamics[:, 1], Dynamics[:, 2], Dynamics[:, 3]]).dump(fileTrait)
                            array([Dynamics[:, 4], Dynamics[:, 5], Dynamics[:, 6]]).dump(fileSymb)
                            fileCoral.flush()
                            fileTrait.flush()
                            fileSymb.flush()
                            fileCoral.close()
                            fileTrait.close()
                            fileSymb.close()
          
