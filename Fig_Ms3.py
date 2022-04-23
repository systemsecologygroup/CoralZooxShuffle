# -*- coding: utf-8 -*-
from __future__ import division

from scipy import linspace, array, zeros, mean, load, sin, pi, amax, ones, exp
from numpy import concatenate, arange
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

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

#### Model with a dynamics on 3 types of symbiont population using odeint solver + bleaching + sinusoidal temperature #####
K_C = 5125112.94092e10 # earth coral carying capacity
Ksmax = 3e6 # healthy measure of carying capacity of symbiont per host biomass Gamma

color_list = ["#fbceb1", "#d99058", "#964b00"] # for the symbionts and trait invested in symbiont types

fsizeT = 18
fsize = 15
width = 0.4

beta = 1e2
import pylab
pylab.rcParams['xtick.major.pad']='8'

# Temperature forcing function for first forcing, mid duration is included so that the first
# phase of each forcing are identical but t1 is not needed for Forcing1 since temperature is not recovering (it's just I had t1 in other versions and I forgot to remove it) 
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
        
F_list = [Forcing1, Forcing2]
Forc = ["Forcing1", "Forcing2"]  

cases = ["case1/", "case2/"]

# plot cases together for the all temperature
Fig_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

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
Tk = [(0.25/12)/14] # half saturation of sigmoid is 0.5 day after t0, the same for all forcing (see forcing functions)
Tl = [2] # power of the sigmoid
# list of deviations from tbar0
ab_steps = 0.1
A_B = arange(0, 4+ab_steps, ab_steps)

F_list = [Forcing1, Forcing2] # forcing list
Forc = ["Forcing1", "Forcing2"]  

A_Bstr =["Height%d"%i for i in xrange(len(A_B))] # filenams for each elements in the list of deviations from Tbar0
Tbar0_list  = [26, 27, 28]


t0 = 2 # starting time of temperature shift
dur_steps = 0.25/12  # steps of week i.e quater a month
dur_list = arange(0.25/12, 4+dur_steps, dur_steps) # list of duration of stress, by steps of dur_steps, minimun duration is one week 
dur_str=["Dur%d"%i for i in xrange(len(dur_list))]

# symbiont more adapted to mean temperature Tbar0 is the more abundant       
IndexInit_26 = (low, low, high)  # for Tbar0 = 26
IndexInit_27 = (low, high, low)  # for Tbar0 = 27
IndexInit_28 = (high, low, low) # for Tbar0 = 28
IndexInit_list = [IndexInit_26, IndexInit_27, IndexInit_28] 

                    

# Plot cases on the same plot for a particular Tbar0, duration and magnitude of stress for different time period
"""
plt.figure(figsize = (12, 9))
tb = 1 # Index of Tbar0 to plot
ik = 0 # index of Tk to plot
il = 0 # index of TL to plot

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


Time_min = 1  # considered as time 0 because we assume runs bellow Time_min is spine up
Time_max = 5  # 

# For tb = 0 i.e Tbar0 = 26°C
#dl_A = 30 # index of 1st duration of stress to plot 
#dl_B = 55 # index of 2nd duration of stress to plot 
#iAB = 10 # in xrange(len(A_B))] # Index of temperature deviations to plot

# For tb = 1  i.e Tbar0 = 27°C
dl_A = 54 # index of 1st duration of stress to plot 
dl_B = 75 # index of 2nd duration of stress to plot
iAB = 20 # in xrange(len(A_B))] # Index of temperature deviations to plot

# For tb = 2 i.e Tbar0 = 28°C, there is not shuffling under 28 since there are not symbiont with mean thermal tolerance higher than 28°C
#dl_A = 49 # index of 1st duration of stress to plot 
#dl_B = 75 # index of 2nd duration of stress to plot 
#iAB = 20 # in xrange(len(A_B))] # Index of temperature deviations to plot

lwForc = 2.5 # linewidth for temperature forcing
lwCase1 = 1.5 # linewidth for case1
alphaCase1 = 1 # transparency for case1
lwCase2 = 6# linewidth for case2
alphaCase2 = 1 # transparency for case2


# Case1 Forcing 2 # for first column of plot
fileCoral1_A = open(cases[0]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_A]+".dat", "r")
fileTrait1_A = open(cases[0]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_A]+".dat", "r")
fileSymb1_A = open(cases[0]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_A]+".dat", "r")     
Coral1_A = load(fileCoral1_A)
Trait1_A = load(fileTrait1_A)
Symb1_A = load(fileSymb1_A)
fileCoral1_A.close()
fileTrait1_A.close()
fileSymb1_A.close()  

# Case1 Forcing 2 # for second column of plot
fileCoral1_B = open(cases[0]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_B]+".dat", "r")
fileTrait1_B = open(cases[0]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_B]+".dat", "r")
fileSymb1_B = open(cases[0]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_B]+".dat", "r")     
Coral1_B = load(fileCoral1_B)
Trait1_B = load(fileTrait1_B)
Symb1_B = load(fileSymb1_B)
fileCoral1_B.close()
fileTrait1_B.close()
fileSymb1_B.close()  

# Case1 Forcing 1, for 3rd column of plot
fileCoral1 = open(cases[0]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[0]+A_Bstr[iAB]+".dat", "r")
fileTrait1 = open(cases[0]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[0]+A_Bstr[iAB]+".dat", "r")
fileSymb1 = open(cases[0]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[0]+A_Bstr[iAB]+".dat", "r")     
Coral1 = load(fileCoral1)
Trait1 = load(fileTrait1)
Symb1 = load(fileSymb1)
fileCoral1.close()
fileTrait1.close()
fileSymb1.close()   

# Case2 Forcing 2 # for 1st column of plot
fileCoral2_A = open(cases[1]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_A]+".dat", "r")
fileTrait2_A = open(cases[1]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_A]+".dat", "r")
fileSymb2_A = open(cases[1]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_A]+".dat", "r")     
Coral2_A = load(fileCoral2_A)
Trait2_A = load(fileTrait2_A)
Symb2_A = load(fileSymb2_A)
fileCoral2_A.close()
fileTrait2_A.close()
fileSymb2_A.close()  

# Case2 Forcing 2 # for 2nd column of plot
fileCoral2_B = open(cases[1]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_B]+".dat", "r")
fileTrait2_B = open(cases[1]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_B]+".dat", "r")
fileSymb2_B = open(cases[1]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[dl_B]+".dat", "r")     
Coral2_B = load(fileCoral2_B)
Trait2_B = load(fileTrait2_B)
Symb2_B = load(fileSymb2_B)
fileCoral2_B.close()
fileTrait2_B.close()
fileSymb2_B.close()  

# Case2 Forcing 1 # for 3rd column of plot
fileCoral2 = open(cases[1]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[0]+A_Bstr[iAB]+".dat", "r")
fileTrait2 = open(cases[1]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[0]+A_Bstr[iAB]+".dat", "r")
fileSymb2 = open(cases[1]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[0]+A_Bstr[iAB]+".dat", "r")     
Coral2 = load(fileCoral2)
Trait2 = load(fileTrait2)
Symb2 = load(fileSymb2)
fileCoral2.close()
fileTrait2.close()
fileSymb2.close()   

# Fig Labels
ax0Lab = ["a", "b", "c"]
ax1Lab = ["d", "e", "f"]
ax2Lab = ["g", "h", "i"]

for k in xrange(3):
    ax0 = plt.subplot(3, 3, 1+k) # for temperature forcing
    
    ax1 = plt.subplot(3, 3, 4+k) # for corals
    ax2 = plt.subplot(3, 3, 7+k) # for symbiont

    if 1+k != 1 and 4+k != 4 and 7+k!=7:
        ax0.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
    else: 
        ax0.set_ylabel("Temperature forcing\n"+u"(\N{DEGREE SIGN}C)", fontsize = fsizeT)
        ax1.set_ylabel("Coral abundance\n(normalized)", fontsize = fsizeT)
        ax2.set_ylabel("Symbiont abundance\n($10^6$ cells per cm$^2$)", fontsize = fsizeT)
        ax0.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize) 
    
    # Plot forcing
    if 1+k == 1:
        ax0.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax0.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax0.set_xlim((Time_min, Time_max))
        TimePer1 = Time[(Time<=Time_max)*Time>Time_min]
        forc = zeros(len(TimePer1))
        t1 = t0 + dur_list[dl_A] 
        for n in xrange(len(TimePer1)):
            forc[n] = Forcing2(TimePer1[n], t0, t1, Tk[ik], Tl[il], Tbar0, A_B[iAB]) 
        ax0.plot(TimePer1, forc, linewidth = lwForc, color="#4F628E")   
        
    if 1+k == 2:
        ax0.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax0.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax0.set_xlim((Time_min, Time_max))
        TimePer2 = Time[(Time<=Time_max)*Time>Time_min]
        forc2B = zeros(len(TimePer2)) # for 2A and 2C it's the same
        t1B = t0 + dur_list[dl_B] # dl3A = dl3C
        for n in xrange(len(TimePer2)):
            forc2B[n] = Forcing2(TimePer2[n], t0, t1B, Tk[ik], Tl[il], Tbar0, A_B[iAB]) 
        ax0.plot(TimePer2, forc2B, linewidth = lwForc, color = "#4F628E")
                                        
    if 1+k == 3:
        ax0.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax0.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax0.set_xlim((Time_min, Time_max))
        TimePer2 = Time[(Time<=Time_max)*Time>Time_min]
        forc2B = zeros(len(TimePer2))
        t1B = t0 + dur_list[dl_B] # this does not matter for Forcing 1, it's just I forgot to remove it from the function from previous versions
        for n in xrange(len(TimePer2)):
            forc2B[n] = Forcing1(TimePer2[n], t0, t1B, Tk[ik], Tl[il], Tbar0, A_B[iAB]) 
        ax0.plot(TimePer2, forc2B, linewidth = lwForc, color = "#4F628E")
        
    # Plot coral biomass        
    if 4+k == 4:
        ax1.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax1.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax1.set_xlim((Time_min, Time_max))
        ax1.plot(Time, Coral1_A/K_C, linewidth = lwCase1, color="#0F5738", alpha = alphaCase1)
        ax1.plot(Time, Coral2_A/K_C, linewidth = lwCase2, color="#0F5738", alpha = alphaCase2)

    if 4+k == 5:
        ax1.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax1.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax1.set_xlim((Time_min, Time_max))
        ax1.plot(Time, Coral1_B/K_C, linewidth = lwCase1, color="#0F5738", alpha = alphaCase1)
        ax1.plot(Time, Coral2_B/K_C, linewidth = lwCase2, color="#0F5738", alpha = alphaCase2)
        
    if 4+k == 6:
        ax1.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax1.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax1.set_xlim((Time_min, Time_max))
        ax1.plot(Time, Coral1/K_C, linewidth = lwCase1, color="#0F5738", alpha = alphaCase1)
        ax1.plot(Time, Coral2/K_C,linewidth = lwCase2, color="#0F5738", alpha = alphaCase2)
    # plot Symbiont biomass    
    if 7+k == 7:
        ax2.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax2.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax2.set_xlim((Time_min, Time_max))
        
        ax2.plot(Time, (Symb1_A[2,:]/Coral1_A)/1e6, linewidth = lwCase1, color = "#fbceb1", alpha = alphaCase1)
        ax2.plot(Time, (Symb1_A[1,:]/Coral1_A)/1e6, linewidth = lwCase1, color = "#d99058", alpha = alphaCase1)
        ax2.plot(Time, (Symb1_A[0,:]/Coral1_A)/1e6, linewidth = lwCase1, color = "#964b00", alpha = alphaCase1)

        ax2.plot(Time, (Symb2_A[2,:]/Coral2_A)/1e6, linewidth = lwCase2, color = "#fbceb1", alpha = alphaCase2)
        ax2.plot(Time, (Symb2_A[1,:]/Coral2_A)/1e6, linewidth = lwCase2, color = "#d99058", alpha = alphaCase2)
        ax2.plot(Time, (Symb2_A[0,:]/Coral2_A)/1e6, linewidth = lwCase2, color = "#964b00", alpha = alphaCase2)
        
        print "H1", sum((Symb1_A[2,:] < Symb1_A[1,:])+(Symb1_A[2,:] < Symb1_A[0,:]))
        print "H2", sum((Symb2_A[2,:] < Symb2_A[1,:])+(Symb2_A[2,:] < Symb2_A[0,:]))

        
    if 7+k == 8:
        ax2.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax2.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax2.set_xlim((Time_min, Time_max))
    
        
        ax2.plot(Time, (Symb1_B[2,:]/Coral1_B)/1e6, linewidth = lwCase1, color = "#fbceb1", alpha = alphaCase1)
        ax2.plot(Time, (Symb1_B[1,:]/Coral1_B)/1e6, linewidth = lwCase1, color = "#d99058", alpha = alphaCase1)
        ax2.plot(Time, (Symb1_B[0,:]/Coral1_B)/1e6, linewidth = lwCase1, color = "#964b00", alpha = alphaCase1)

        ax2.plot(Time, (Symb2_B[2,:]/Coral2_B)/1e6,  linewidth = lwCase2, color = "#fbceb1", alpha = alphaCase2)
        ax2.plot(Time, (Symb2_B[1,:]/Coral2_B)/1e6, linewidth = lwCase2, color = "#d99058", alpha = alphaCase2)
        ax2.plot(Time, (Symb2_B[0,:]/Coral2_B)/1e6, linewidth = lwCase2, color = "#964b00", alpha = alphaCase2)

    if 7+k == 9:
        ax2.set_xticks(arange(Time_min, Time_max+0.5, 0.5))
        ax2.set_xticklabels(["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"])#set_xticklabels(["0"]+["%.1f"%(ti-1) for ti in arange(Time_min, Time_max+0.5, 0.5)[1:]])
        ax2.set_xlim((Time_min, Time_max))
        
        ax2.plot(Time, (Symb1[2,:]/Coral1)/1e6, linewidth = lwCase1, color = "#fbceb1", alpha = alphaCase1)
        ax2.plot(Time, (Symb1[1,:]/Coral1)/1e6,linewidth = lwCase1, color = "#d99058", alpha = alphaCase1)
        ax2.plot(Time, (Symb1[0,:]/Coral1)/1e6,linewidth = lwCase1, color = "#964b00", alpha = alphaCase1)

        ax2.plot(Time, (Symb2[2,:]/Coral2)/1e6, linewidth = lwCase2, color = "#fbceb1", alpha = alphaCase2)
        ax2.plot(Time, (Symb2[1,:]/Coral2)/1e6, linewidth = lwCase2, color = "#d99058", alpha = alphaCase2)
        ax2.plot(Time, (Symb2[0,:]/Coral2)/1e6, linewidth = lwCase2, color = "#964b00", alpha = alphaCase2)
        
    if 7+k in (7, 8, 9):
        ax0.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelbottom = True, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.set_xlabel("Time (years)", fontsize = fsizeT)
        
    else:
        ax0.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        
    ax0.set_yticks([25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31])
    ax0.set_yticklabels(["%.1f"%temp for temp in [25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31]])
    #ax0.set_ylim((25, Tbar0+max(A_B)+1))
    #ax0.set_ylim((25, 28)) # 26°C
    ax0.set_ylim((26, 30)) # 27°C
    #ax0.set_ylim((27, 31)) # 28°C
    if k != 0:
        ax0.text(1-0.2,30, ax0Lab[k], fontproperties = font, fontsize = fsize)
        ax1.text(1-0.2, 1, ax1Lab[k], fontproperties = font, fontsize = fsize)
        ax2.text(1-0.2, 3.5, ax2Lab[k], fontproperties = font, fontsize = fsize)
    else:
        ax0.text(1-0.85,30, ax0Lab[k], fontproperties = font, fontsize = fsize)
        ax1.text(1-0.85, 1, ax1Lab[k], fontproperties = font, fontsize = fsize)
        ax2.text(1-0.85, 3.5, ax2Lab[k], fontproperties = font, fontsize = fsize)

    ax1.set_yticks(arange(0, 1+0.02, 0.02))
    ax1.set_yticklabels(["%.2f"%temp for temp in arange(0, 1+0.02, 0.02)])
    #ax1.set_ylim((0.88, 1)) # 26°C
    ax1.set_ylim((0.82, 1)) # 27°C
    #ax1.set_ylim((0.78, 1)) # 28°C
    
    ax2.set_yticks(arange(0, 3.5+0.5, 0.5))
    ax2.set_yticklabels(["%.2f"%temp for temp in arange(0, 3.5+0.5, 0.5)])
    ax2.set_ylim((0, 3.5))

ax2.plot(Time, (-20)*ones(len(Time)), linewidth = lwCase1, color = "black", label = "H1")
ax2.plot(Time, (-20)*ones(len(Time)), linewidth = lwCase2, color = "black", label = "H2")
ax2.legend(loc = (0.025, 0.08), ncol = 1, fontsize = fsize-1, frameon = False)
ax2.text(1.1, 1.25, "$S_1$", color = "#fbceb1", fontsize = fsize)
ax2.text(1.5, 1.25, "$S_2$", color = "#d99058", fontsize = fsize)
ax2.text(1.9, 1.25, "$S_3$", color = "#964b00", fontsize = fsize)

plt.subplots_adjust(bottom = 0.06, right = 0.95, left = 0.1, top = 0.98, wspace = 0.15, hspace = 0.15)
"""

# For forcing 2 only, plot coral and symbiont contourplot density within axis magnitude and duration of stress 
#for a particular Tbar0 and for the time period within which shuffling might happen and after within which recovery to pre-excrusion might happen                 


# Creating contourplot data, save the data then comment this because it takes a while
"""
tb = 2 # Index of Tbar0, 0 for 26°C, 1 for 27°C, and 2 for 28°C
ik = 0 # index of Tk 
il = 0 # index of TL 
Tbar0 = Tbar0_list[tb]

Time = arange(0, t0+tmax+step, step)
period0 = (Time>=1)*(Time<=2) # the first year, year starts at 1 (which we consider year 0 in the manuscript since we have a one year spine up phase)
period1 = (Time>=2)*(Time<=4) # time within which shuffling might happen, estimated from previous plots
period2 = (Time>=4.)*(Time<=10)# time within which recovery might happen

Coral_Case1_Period0 = zeros((len(A_B), len(dur_list)))
Coral_Case1_Period1 = zeros((len(A_B), len(dur_list)))
Coral_Case1_Period2 = zeros((len(A_B), len(dur_list)))

Symb1_Case1_Period0 = zeros((len(A_B), len(dur_list)))
Symb2_Case1_Period0 = zeros((len(A_B), len(dur_list)))
Symb3_Case1_Period0 = zeros((len(A_B), len(dur_list)))

Symb1_Case1_Period1 = zeros((len(A_B), len(dur_list)))
Symb2_Case1_Period1 = zeros((len(A_B), len(dur_list)))
Symb3_Case1_Period1 = zeros((len(A_B), len(dur_list)))

Symb1_Case1_Period2 = zeros((len(A_B), len(dur_list)))
Symb2_Case1_Period2 = zeros((len(A_B), len(dur_list)))
Symb3_Case1_Period2 = zeros((len(A_B), len(dur_list)))

Coral_Case2_Period1 = zeros((len(A_B), len(dur_list)))
Coral_Case2_Period2 = zeros((len(A_B), len(dur_list)))

Symb1_Case2_Period1 = zeros((len(A_B), len(dur_list)))
Symb2_Case2_Period1 = zeros((len(A_B), len(dur_list)))
Symb3_Case2_Period1 = zeros((len(A_B), len(dur_list)))

Symb1_Case2_Period2 = zeros((len(A_B), len(dur_list)))
Symb2_Case2_Period2 = zeros((len(A_B), len(dur_list)))
Symb3_Case2_Period2 = zeros((len(A_B), len(dur_list)))

for iAB in xrange(len(A_B)):
    for iDur in xrange(len(dur_list)):
        print (iAB, len(A_B)), (iDur, len(dur_list))
        fileCoral1 = open(cases[0]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[iDur]+".dat", "r")
        #fileTrait1 = open(cases[0]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[iDur]+".dat", "r")
        fileSymb1 = open(cases[0]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[iDur]+".dat", "r")     
        Coral_Case1 = load(fileCoral1)
        #Trait2_Case1 = load(fileTrait1)
        Symb_Case1 = load(fileSymb1)
        fileCoral1.close()
        #fileTrait1.close()
        fileSymb1.close()
        
        Coral_Case1_Period0[iAB, iDur] = mean(Coral_Case1[period0]/K_C)
        
        NormSymb10 = Coral_Case1[period0]*1e6
        
        Symb1_Case1_Period0[iAB, iDur] = mean(Symb_Case1[2, period0]/NormSymb10)
        Symb2_Case1_Period0[iAB, iDur] = mean(Symb_Case1[1, period0]/NormSymb10)
        Symb3_Case1_Period0[iAB, iDur] = mean(Symb_Case1[0, period0]/NormSymb10)
        
        Coral_Case1_Period1[iAB, iDur] = mean(Coral_Case1[period1]/K_C)
        
        NormSymb11 = Coral_Case1[period1]*1e6
        
        Symb1_Case1_Period1[iAB, iDur] = mean(Symb_Case1[2, period1]/NormSymb11)
        Symb2_Case1_Period1[iAB, iDur] = mean(Symb_Case1[1, period1]/NormSymb11)
        Symb3_Case1_Period1[iAB, iDur] = mean(Symb_Case1[0, period1]/NormSymb11)
        
        Coral_Case1_Period2[iAB, iDur] = mean(Coral_Case1[period2]/K_C)
        
        NormSymb12 = Coral_Case1[period2]*1e6
        
        Symb1_Case1_Period2[iAB, iDur] = mean(Symb_Case1[2, period2]/NormSymb12)
        Symb2_Case1_Period2[iAB, iDur] = mean(Symb_Case1[1, period2]/NormSymb12)
        Symb3_Case1_Period2[iAB, iDur] = mean(Symb_Case1[0, period2]/NormSymb12)
        
        fileCoral2 = open(cases[1]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[iDur]+".dat", "r")
        #fileTrait2 = open(cases[1]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[iDur]+".dat", "r")
        fileSymb2 = open(cases[1]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(ik, il)+Forc[1]+A_Bstr[iAB]+dur_str[iDur]+".dat", "r")     
        Coral_Case2 = load(fileCoral2)
        #Trait_Case2 = load(fileTrait2)
        Symb_Case2 = load(fileSymb2)
        fileCoral2.close()
        #fileTrait2.close()
        fileSymb2.close()  

        Coral_Case2_Period1[iAB, iDur] = mean(Coral_Case2[period1]/K_C)
        
        NormSymb21 = Coral_Case2[period1]*1e6
        
        Symb1_Case2_Period1[iAB, iDur] = mean(Symb_Case2[2, period1]/NormSymb21)
        Symb2_Case2_Period1[iAB, iDur] = mean(Symb_Case2[1, period1]/NormSymb21)
        Symb3_Case2_Period1[iAB, iDur] = mean(Symb_Case2[0, period1]/NormSymb21)

        Coral_Case2_Period2[iAB, iDur] = mean(Coral_Case2[period2]/K_C)
        
        NormSymb22 = Coral_Case2[period2]*1e6
        
        Symb1_Case2_Period2[iAB, iDur] = mean(Symb_Case2[2, period2]/NormSymb22)
        Symb2_Case2_Period2[iAB, iDur] = mean(Symb_Case2[1, period2]/NormSymb22)
        Symb3_Case2_Period2[iAB, iDur] = mean(Symb_Case2[0, period2]/NormSymb22)

file_Coral_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Coral_Case1_Period0.dat", "wr")
Coral_Case1_Period0.dump(file_Coral_Case1_Period0)
file_Coral_Case1_Period0.flush()

file_Symb1_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Symb1_Case1_Period0.dat", "wr")
file_Symb2_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Symb2_Case1_Period0.dat", "wr")
file_Symb3_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Symb3_Case1_Period0.dat", "wr")

Symb1_Case1_Period0.dump(file_Symb1_Case1_Period0)
file_Symb1_Case1_Period0.flush()
Symb2_Case1_Period0.dump(file_Symb2_Case1_Period0)
file_Symb2_Case1_Period0.flush()
Symb3_Case1_Period0.dump(file_Symb3_Case1_Period0)
file_Symb3_Case1_Period0.flush()

file_Coral_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Coral_Case1_Period1.dat", "wr")
Coral_Case1_Period1.dump(file_Coral_Case1_Period1)
file_Coral_Case1_Period1.flush()

file_Symb1_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Symb1_Case1_Period1.dat", "wr")
file_Symb2_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Symb2_Case1_Period1.dat", "wr")
file_Symb3_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Symb3_Case1_Period1.dat", "wr")

Symb1_Case1_Period1.dump(file_Symb1_Case1_Period1)
file_Symb1_Case1_Period1.flush()
Symb2_Case1_Period1.dump(file_Symb2_Case1_Period1)
file_Symb2_Case1_Period1.flush()
Symb3_Case1_Period1.dump(file_Symb3_Case1_Period1)
file_Symb3_Case1_Period1.flush()

file_Coral_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Coral_Case1_Period2.dat", "wr")
Coral_Case1_Period2.dump(file_Coral_Case1_Period2)
file_Coral_Case1_Period2.flush()

file_Symb1_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Symb1_Case1_Period2.dat", "wr")
file_Symb2_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Symb2_Case1_Period2.dat", "wr")
file_Symb3_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Symb3_Case1_Period2.dat", "wr")

Symb1_Case1_Period2.dump(file_Symb1_Case1_Period2)
file_Symb1_Case1_Period2.flush()
Symb2_Case1_Period2.dump(file_Symb2_Case1_Period2)
file_Symb2_Case1_Period2.flush()
Symb3_Case1_Period2.dump(file_Symb3_Case1_Period2)
file_Symb3_Case1_Period2.flush()

file_Coral_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Coral_Case2_Period1.dat", "wr")
Coral_Case2_Period1.dump(file_Coral_Case2_Period1)
file_Coral_Case2_Period1.flush()

file_Symb1_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Symb1_Case2_Period1.dat", "wr")
file_Symb2_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Symb2_Case2_Period1.dat", "wr")
file_Symb3_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Symb3_Case2_Period1.dat", "wr")

Symb1_Case2_Period1.dump(file_Symb1_Case2_Period1)
file_Symb1_Case2_Period1.flush()
Symb2_Case2_Period1.dump(file_Symb2_Case2_Period1)
file_Symb2_Case2_Period1.flush()
Symb3_Case2_Period1.dump(file_Symb3_Case2_Period1)
file_Symb3_Case2_Period1.flush()

file_Coral_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Coral_Case2_Period2.dat", "wr")
Coral_Case2_Period2.dump(file_Coral_Case2_Period2)
file_Coral_Case2_Period2.flush()

file_Symb1_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Symb1_Case2_Period2.dat", "wr")
file_Symb2_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Symb2_Case2_Period2.dat", "wr")
file_Symb3_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Symb3_Case2_Period2.dat", "wr")

Symb1_Case2_Period2.dump(file_Symb1_Case2_Period2)
file_Symb1_Case2_Period2.flush()
Symb2_Case2_Period2.dump(file_Symb2_Case2_Period2)
file_Symb2_Case2_Period2.flush()
Symb3_Case2_Period2.dump(file_Symb3_Case2_Period2)
file_Symb3_Case2_Period2.flush()
"""


## Sensitivity to temperature forcing - Symbiont abundance in million cell per cm^2 coral surface ###

tb = 0 # Index of Tbar0, 0 for 26°C, 1 for 27°C, and 2 for 28°C
ik = 0 # index of Tk to plot
il = 0 # index of TL to plot 
Tbar0 = Tbar0_list[tb]

k = 0

file_Coral_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Coral_Case1_Period0.dat", "r")
Coral_Case1_Period0 = load(file_Coral_Case1_Period0)
file_Coral_Case1_Period0.close()

file_Symb1_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Symb1_Case1_Period0.dat", "r")
file_Symb2_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Symb2_Case1_Period0.dat", "r")
file_Symb3_Case1_Period0 = open("ContourData/"+str(Tbar0)+"/Symb3_Case1_Period0.dat", "r")

Symb1_Case1_Period0 = load(file_Symb1_Case1_Period0)
file_Symb1_Case1_Period0.close()
Symb2_Case1_Period0 = load(file_Symb2_Case1_Period0)
file_Symb2_Case1_Period0.close()
Symb3_Case1_Period0 = load(file_Symb3_Case1_Period0)
file_Symb3_Case1_Period0.close()

file_Coral_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Coral_Case1_Period1.dat", "r")
Coral_Case1_Period1 = load(file_Coral_Case1_Period1)
file_Coral_Case1_Period1.close()

file_Symb1_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Symb1_Case1_Period1.dat", "r")
file_Symb2_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Symb2_Case1_Period1.dat", "r")
file_Symb3_Case1_Period1 = open("ContourData/"+str(Tbar0)+"/Symb3_Case1_Period1.dat", "r")

Symb1_Case1_Period1 = load(file_Symb1_Case1_Period1)
file_Symb1_Case1_Period1.close()
Symb2_Case1_Period1 = load(file_Symb2_Case1_Period1)
file_Symb2_Case1_Period1.close()
Symb3_Case1_Period1 = load(file_Symb3_Case1_Period1)
file_Symb3_Case1_Period1.close()

file_Coral_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Coral_Case1_Period2.dat", "r")
Coral_Case1_Period2 = load(file_Coral_Case1_Period2)
file_Coral_Case1_Period2.close()

file_Symb1_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Symb1_Case1_Period2.dat", "r")
file_Symb2_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Symb2_Case1_Period2.dat", "r")
file_Symb3_Case1_Period2 = open("ContourData/"+str(Tbar0)+"/Symb3_Case1_Period2.dat", "r")

Symb1_Case1_Period2 = load(file_Symb1_Case1_Period2)
file_Symb1_Case1_Period2.close()
Symb2_Case1_Period2 = load(file_Symb2_Case1_Period2)
file_Symb2_Case1_Period2.close()
Symb3_Case1_Period2 = load(file_Symb3_Case1_Period2)
file_Symb3_Case1_Period2.close()

file_Coral_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Coral_Case2_Period1.dat", "r")
Coral_Case2_Period1 = load(file_Coral_Case2_Period1)
file_Coral_Case2_Period1.close()

file_Symb1_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Symb1_Case2_Period1.dat", "r")
file_Symb2_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Symb2_Case2_Period1.dat", "r")
file_Symb3_Case2_Period1 = open("ContourData/"+str(Tbar0)+"/Symb3_Case2_Period1.dat", "r")

Symb1_Case2_Period1 = load(file_Symb1_Case2_Period1)
file_Symb1_Case2_Period1.close()
Symb2_Case2_Period1 = load(file_Symb2_Case2_Period1)
file_Symb2_Case2_Period1.close()
Symb3_Case2_Period1 = load(file_Symb3_Case2_Period1)
file_Symb3_Case2_Period1.close()

file_Coral_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Coral_Case2_Period2.dat", "r")
Coral_Case2_Period2 = load(file_Coral_Case2_Period2)
file_Coral_Case2_Period2.close()

file_Symb1_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Symb1_Case2_Period2.dat", "r")
file_Symb2_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Symb2_Case2_Period2.dat", "r")
file_Symb3_Case2_Period2 = open("ContourData/"+str(Tbar0)+"/Symb3_Case2_Period2.dat", "r")

Symb1_Case2_Period2 = load(file_Symb1_Case2_Period2)
file_Symb1_Case2_Period2.close()
Symb2_Case2_Period2 = load(file_Symb2_Case2_Period2)
file_Symb2_Case2_Period2.close()
Symb3_Case2_Period2 = load(file_Symb3_Case2_Period2)
file_Symb3_Case2_Period2.close()

#plot cases separately
fig = plt.figure(figsize = (12, 9))
#plt.margins(0.01, 0.01)
ax = fig.add_subplot(111)
#cbaxes1 = fig.add_axes([0.89, 0.3, 0.04, 0.98]) # [left, bottom, width, height]
#cbaxes2 = fig.add_axes([0.89, 0.1, 0.04, 0.98])
#cbaxes3 = fig.add_axes([0.89, -0.1, 0.04, 0.98])
#cbaxes4 = fig.add_axes([0.89, -0.3, 0.04, 0.98])
cbaxes1 = fig.add_axes([0.76, 0.4, 0.04, 0.95]) # [left, bottom, width, height]
cbaxes2 = fig.add_axes([0.76, 0.175, 0.04, 0.95])
cbaxes3 = fig.add_axes([0.76, -0.0525, 0.04, 0.95])
cbaxes4 = fig.add_axes([0.76, -0.277, 0.04, 0.95])
# colorbar label pad
lp = 10
# Figure labels
ax1Lab = ["a", "b", "c"]
ax2Lab = ["d", "e", "f"]
ax3Lab = ["g", "h", "i"]
ax4Lab = ["j", "k", "l"]

case = 2  # 1 for H1 and 2 for H2
for k in xrange(3):
    ax1 = plt.subplot(4, 3, 1+k) # for coral
    ax2 = plt.subplot(4, 3, 4+k) # for low thermotolerant symbiont
    ax3 = plt.subplot(4, 3, 7+k) # for mid thermotolerant symbiont
    ax4 = plt.subplot(4, 3, 10+k) # for high thermotolerant symbiont

    if 1+k != 1 and 4+k != 4 and 7+k!=7 and 10+k!=10:
        ax1.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax3.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax4.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)

    else: 
        #ax1.set_ylabel("Magnitude\n"+u"(\N{DEGREE SIGN}C)", fontsize = fsize)
        #ax2.set_ylabel("Magnitude\n"+u"(\N{DEGREE SIGN}C)", fontsize = fsize)
        ax3.text(-1.35, 8.5, "Magnitude of temperature shift "+u"(\N{DEGREE SIGN}C)", rotation = "vertical", fontsize = fsizeT)
        #ax4.set_ylabel("Magnitude\n"+u"(\N{DEGREE SIGN}C)", fontsize = fsize)
        
        ax1.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize) 
        ax3.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax4.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize) 
    
    if 10+k in (10, 11, 12):
        ax1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax3.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax4.tick_params(labelbottom = True, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        if 10+k == 11:
            #ax4.set_xlabel("Duration of temperature shift (years)", fontsize = fsize)
            ax4.text(-1.35, -1.25, "Duration of temperature shift (years)", fontsize = fsizeT)
        
    else:
        ax1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax3.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax4.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        
    
    # Coral biomass for case1 and case2
    #levelCoral = linspace(0.80, 1, 100)
    minBio = 0.92 # 0.80 for 28°C (tb = 2), 0.84 for 27°C (tb = 1); 0.92 for 26°C (tb = 0) (checked visually)
    levelCoral = linspace(minBio, 1, 100)
    if 1+k == 1: # mean coral biomass within period0
        if case == 1 or case == 2: # period 0 is the same for all cases
            CS = ax1.contourf(dur_list, A_B, Coral_Case1_Period0, levelCoral, cmap = "YlGn")
            #plt.colorbar(CS, ax = ax1) 
    if 1+k == 2: # mean coral biomass within period1
        if case == 1:
            CS = ax1.contourf(dur_list, A_B, Coral_Case1_Period1, levelCoral, cmap = "YlGn")
            #plt.colorbar(CS, ax = ax1) 
        else:
            CS = ax1.contourf(dur_list, A_B, Coral_Case2_Period1, levelCoral, cmap = "YlGn") 
            #plt.colorbar(CS, ax = ax1) 
    elif 1+k == 3: # mean coral biomass within period2
        if case == 1:
            CS = ax1.contourf(dur_list, A_B, Coral_Case1_Period2, levelCoral, cmap = "YlGn") 
            #cbar=plt.colorbar(CS, ax = ax1)
            cbar = plt.colorbar(CS, ax = cbaxes1)
            cbar.set_label("Coral abundance\n(normalized)", fontsize = fsize, labelpad = lp)
            cbar.set_ticks(arange(minBio, 1+0.02, 0.02)[::1]) # ::1 for 26°C and 27°C, ::2 for 28°C
            cbar.ax.tick_params(labelsize=fsize) 
            cbar.set_ticklabels(["%.2f"%temp for temp in arange(minBio, 1+0.02, 0.02)[::1]]) # ::1 for 26°C and 27°C, ::2 for 28°C

        else:
            CS = ax1.contourf(dur_list, A_B, Coral_Case2_Period2, levelCoral, cmap = "YlGn")  
            #cbar=plt.colorbar(CS, ax = ax1)
            cbar = plt.colorbar(CS, ax = cbaxes1)
            cbar.set_label("Coral abundance\n(normalized)", fontsize = fsize, labelpad = lp)
            cbar.set_ticks(arange(minBio, 1+0.02, 0.02)[::1]) # ::1 for 26°C and 27°C, ::2 for 28°C
            cbar.ax.tick_params(labelsize=fsize) 
            cbar.set_ticklabels(["%.2f"%temp for temp in arange(minBio, 1+0.02, 0.02)[::1]]) # ::1 for 26°C and 27°C, ::2 for 28°C
    
    # Symbiont biomass for case1 and case2
    levelSymb1 = linspace(0, 3., 100)   # 3 for 26°C (tb = 0) and 27°C (tb = 1), 3. for 28°C tb = 2
    levelSymb2 = linspace(0, 3., 100)   # 3 for 26°C (tb = 0) and 27°C (tb = 1), 3. for 28°C tb = 2
    levelSymb3 = linspace(0, 3., 100)   # 3 for 26°C (tb = 0) and 27°C (tb = 1), 3. for 28°C tb = 2
    if 4+k == 4: # mean symbiont biomass withing period0
        if case == 1 or case == 2 : # the same for period 0
            CS1 = ax2.contourf(dur_list, A_B, Symb1_Case1_Period0, levelSymb1, cmap = "Oranges") 
            CS2 = ax3.contourf(dur_list, A_B, Symb2_Case1_Period0, levelSymb2, cmap = "Oranges") 
            CS3 = ax4.contourf(dur_list, A_B, Symb3_Case1_Period0, levelSymb3, cmap = "Oranges")
            #plt.colorbar(CS1, ax = ax2)
            #plt.colorbar(CS2, ax = ax3)
            #plt.colorbar(CS3, ax = ax4)
    if 4+k == 5: # mean symbiont biomass within period1
        if case == 1:
            CS1 = ax2.contourf(dur_list, A_B, Symb1_Case1_Period1, levelSymb1, cmap = "Oranges") 
            CS2 = ax3.contourf(dur_list, A_B, Symb2_Case1_Period1, levelSymb2, cmap = "Oranges") 
            CS3 = ax4.contourf(dur_list, A_B, Symb3_Case1_Period1, levelSymb3, cmap = "Oranges")
            #plt.colorbar(CS1, ax = ax2)
            #plt.colorbar(CS2, ax = ax3)
            #plt.colorbar(CS3, ax = ax4)
        else: 
            CS1 = ax2.contourf(dur_list, A_B, Symb1_Case2_Period1, levelSymb1, cmap = "Oranges")      
            CS2 = ax3.contourf(dur_list, A_B, Symb2_Case2_Period1, levelSymb2, cmap = "Oranges")      
            CS3 = ax4.contourf(dur_list, A_B, Symb3_Case2_Period1, levelSymb3, cmap = "Oranges")
            #plt.colorbar(CS1, ax = ax2)
            #plt.colorbar(CS2, ax = ax3)
            #plt.colorbar(CS3, ax = ax4)              
    elif 4+k == 6: # mean symbiont biomass within period2
        if case == 1:
            CS1 = ax2.contourf(dur_list, A_B, Symb1_Case1_Period2, levelSymb1, cmap = "Oranges") 
            CS2 = ax3.contourf(dur_list, A_B, Symb2_Case1_Period2, levelSymb2, cmap = "Oranges") 
            CS3 = ax4.contourf(dur_list, A_B, Symb3_Case1_Period2, levelSymb3, cmap = "Oranges")
            #cbar1 = plt.colorbar(CS1, ax = ax2)
            cbar1 = plt.colorbar(CS1, ax = cbaxes2)
            cbar1.set_label("$S_1$ abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize, labelpad = lp)
            #cbar2 = plt.colorbar(CS2, ax = ax3)
            cbar2 = plt.colorbar(CS2, ax = cbaxes3)
            cbar2.set_label("$S_2$ abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize, labelpad = lp)

            #cbar3 = plt.colorbar(CS3, ax = ax4) 
            cbar3 = plt.colorbar(CS3, ax = cbaxes4) 
            cbar3.set_label("$S_3$ abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize, labelpad = lp)

            cbar1.set_ticks(arange(0, 3+0.5, 0.5))
            cbar1.ax.tick_params(labelsize=fsize)
            cbar1.set_ticklabels(["%.2f"%temp for temp in arange(0, 3+0.5, 0.5)])
            
            cbar2.set_ticks(arange(0, 3+0.5, 0.5))
            cbar2.ax.tick_params(labelsize=fsize)
            cbar2.set_ticklabels(["%.2f"%temp for temp in arange(0, 3+0.5, 0.5)])
            
            cbar3.set_ticks(arange(0, 3+0.5, 0.5))
            cbar3.ax.tick_params(labelsize=fsize)
            cbar3.set_ticklabels(["%.2f"%temp for temp in arange(0, 3+0.5, 0.5)])
        else:       
            CS1 = ax2.contourf(dur_list, A_B, Symb1_Case2_Period2, levelSymb1, cmap = "Oranges")      
            CS2 = ax3.contourf(dur_list, A_B, Symb2_Case2_Period2, levelSymb2, cmap = "Oranges")       
            CS3 = ax4.contourf(dur_list, A_B, Symb3_Case2_Period2, levelSymb3, cmap = "Oranges")
            #cbar1 = plt.colorbar(CS1, ax = ax2)
            cbar1 = plt.colorbar(CS1, ax = cbaxes2)
            cbar1.set_label("$S_1$ abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize, labelpad = lp)
            #cbar2 = plt.colorbar(CS2, ax = ax3)
            cbar2 = plt.colorbar(CS2, ax = cbaxes3)
            cbar2.set_label("$S_2$ abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize, labelpad = lp)

            #cbar3 = plt.colorbar(CS3, ax = ax4) 
            cbar3 = plt.colorbar(CS3, ax = cbaxes4) 
            cbar3.set_label("$S_3$ abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize, labelpad = lp)

            cbar1.set_ticks(arange(0, 3+0.5, 0.5))
            cbar1.ax.tick_params(labelsize=fsize)
            cbar1.set_ticklabels(["%.2f"%temp for temp in arange(0, 3+0.5, 0.5)])
            
            cbar2.set_ticks(arange(0, 3+0.5, 0.5))
            cbar2.ax.tick_params(labelsize=fsize)
            cbar2.set_ticklabels(["%.2f"%temp for temp in arange(0, 3+0.5, 0.5)])
            
            cbar3.set_ticks(arange(0, 3+0.5, 0.5))
            cbar3.ax.tick_params(labelsize=fsize)
            cbar3.set_ticklabels(["%.2f"%temp for temp in arange(0, 3+0.5, 0.5)])
    
    ax1.set_yticks(arange(0, 4+1, 1))
    ax1.set_yticklabels(arange(0, 4+1, 1))
    ax2.set_yticks(arange(0, 4+1, 1)) 
    ax2.set_yticklabels(arange(0, 4+1, 1))
    ax3.set_yticks(arange(0, 4+1, 1))
    ax3.set_yticklabels(arange(0, 4+1, 1))
    ax4.set_yticks(arange(0, 4+1, 1))  
    ax4.set_yticklabels(arange(0, 4+1, 1))
    
    ax1.set_xticks(concatenate((array([0.25/12]), arange(1, 4+1, 1))))
    #ax1.set_xticklabels(["$d_0$\n (1 week)"]+["%d"%i for i in arange(1, 4+1, 1)])
    ax1.set_xticklabels(["  $d_0$"]+["%d"%i for i in arange(1, 4+1, 1)])

    ax2.set_xticks(concatenate((array([0.25/12]), arange(1, 4+1, 1)))) 
    ax2.set_xticklabels(["  $d_0$"]+["%d"%i for i in arange(1, 4+1, 1)])
    ax3.set_xticks(concatenate((array([0.25/12]), arange(1, 4+1, 1))))
    ax3.set_xticklabels(["  $d_0$"]+["%d"%i for i in arange(1, 4+1, 1)])
    ax4.set_xticks(concatenate((array([0.25/12]), arange(1, 4+1, 1))))  
    ax4.set_xticklabels(["  $d_0$"]+["%d"%i for i in arange(1, 4+1, 1)])
    
    if k!=0:
        ax1.text(0.25/12 - 0.4, 4, ax1Lab[k], fontproperties = font, fontsize = fsize)
        ax2.text(0.25/12 - 0.4, 4, ax2Lab[k], fontproperties = font, fontsize = fsize)
        ax3.text(0.25/12 - 0.4, 4, ax3Lab[k], fontproperties = font, fontsize = fsize)
        ax4.text(0.25/12 - 0.4, 4, ax4Lab[k], fontproperties = font, fontsize = fsize)
    else:
        ax1.text(0.25/12 - 0.65, 4, ax1Lab[k], fontproperties = font, fontsize = fsize)
        ax2.text(0.25/12 - 0.65, 4, ax2Lab[k], fontproperties = font, fontsize = fsize)
        ax3.text(0.25/12 - 0.65, 4, ax3Lab[k], fontproperties = font, fontsize = fsize)
        ax4.text(0.25/12 - 0.65, 4, ax4Lab[k], fontproperties = font, fontsize = fsize)
        
    
plt.subplots_adjust(bottom = 0.1, right = 0.775, left = 0.2, top = 0.97, wspace = 0.15, hspace = 0.15)


plt.savefig("TemporaryFig.pdf", bbox_inches = 'tight')
plt.show()


# Plot cases side by side for a particular Tbar0, duration and magnitude of stres
"""
fig = plt.figure()

Sty = ["-", ":", "--"]
lw = 2.5
dl = 191 # index of duration of stress to plot
tb = 1 # Index of Tbar0 to plot
ABindex = [40]#i for i in xrange(len(A_B))] # Index of temperature deviations to plot

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

k = 0
for cas in xrange(len(cases)):
    ax0 = plt.subplot(3, 2, 1+k) # for temperature forcing
    ax1 = plt.subplot(3, 2, 3+k) # for corals
    ax2 = plt.subplot(3, 2, 5+k) # for symbiont
    if 1+k != 1 and 4+k != 4 and 7+k !=7:
        ax0.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelleft = False, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)

    else: 
        ax0.set_ylabel("Temperature\nforcing ($^\circ$C)", fontsize = fsize)
        ax1.set_ylabel("Coral abundance\n(normalized)", fontsize = fsize)
        ax2.set_ylabel("Symbiont abundance\n($10^6$ cells per cm$^2$)", fontsize = fsize)
        ax0.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelleft = True, labelright = False, left = True, right = True, bottom = True, top = True, axis = "y", direction = "in", pad = 2, labelsize = fsize)
    
    if 5+k == 5 or 5+k == 6:
        print cases[cas]
        ax0.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelbottom = True, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.set_xlabel("Time(Year)", fontsize = fsize)
    else:
        ax0.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax1.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)
        ax2.tick_params(labelbottom = False, labeltop = False, bottom = True, top = True, axis = "x", direction = "in", pad = 2, labelsize = fsize)

            
    ax0.set_yticks([25, 26, 27, 28, 29, 30, 31])
    ax0.set_yticklabels([25, 26, 27, 28, 29, 30, 31])
    ax0.set_ylim((25, Tbar0+max(A_B)+1))
    #ax1.set_yticks(([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]))
    #ax1.set_yticklabels(([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]))
    ax1.set_yticks(arange(0, 1+0.1, 0.01)[::5])
    ax1.set_yticklabels(arange(0, 1+0.1, 0.01)[::5])
    ax1.plot(Time, zeros(len(Time)), linewidth = 2, color = "black")
    ax1.set_ylim((0.9, 1))
    #ax1.set_ylim((0, 1))
        
    ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax2.set_ylim((0, 3.5))
   
   
    ax0.set_xticks(arange(0, t0+tmax+1, 1))
    ax0.set_xticklabels(["%.d"%t for t in arange(0, t0+tmax+1, 1)])
    ax1.set_xticks(arange(0, t0+tmax+1, 1))
    ax1.set_xticklabels(["%.d"%t for t in arange(0, t0+tmax+1, 1)])
    ax2.set_xticks(arange(0, t0+tmax+1, 1))
    ax2.set_xticklabels(["%.d"%t for t in arange(0, t0+tmax+1, 1)])
    
    #ax0.set_xlim((t0, t0+tmax))
    #ax1.set_xlim((t0, t0+tmax))
    #ax2.set_xlim((t0, t0+tmax))
    ax0.set_xlim((0, t0+tmax))
    ax1.set_xlim((0, t0+tmax))
    ax2.set_xlim((0, t0+tmax))
    
    #ax0.set_xlim((3., 5))
    #ax1.set_xlim((3., 5))
    #ax2.set_xlim((3., 5))
          
    for f in xrange(1,2):#len(F_list)):            
        for i4 in xrange(len(Tk)):
            for i5 in xrange(len(Tl)):
                tk = Tk[i4]
                tl = Tl[i5]
                for ab in ABindex:
                    print Forc[f], A_B[ab]
                    AorB = A_B[ab]
                    if Forc[f] == "Forcing1" and dl == 0:  # There is not need to consider duration of stress for forcing 1
                        fileCoral = open(cases[cas]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+".dat", "r")
                        fileTrait = open(cases[cas]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+".dat", "r")
                        fileSymb = open(cases[cas]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+".dat", "r")        
                        Temperature = Forcing1(0, t0, tk, tl, Tbar0, AorB)
                        Coral = load(fileCoral)
                        Trait = load(fileTrait)
                        Symb = load(fileSymb)
                
                        fileCoral.close()
                        fileTrait.close()
                        fileSymb.close()    
                        # plot coral abundance as normalized biomass
                        #ax1.plot(Time, Coral/K_C, Sty[ABindex.index(ab)], linewidth = lw, color = "#0F5738")
                        ax1.plot(Time, Coral/K_C, linewidth = lw*(ab/max(ABindex)), color = "#0F5738")
                        
                        # plot normalized coral trait 
                        #nT = max(max(Trait[0, :]), max(Trait[1, :]), max(Trait[2, :]))
                        #ax1.plot(Time, (Trait[2, :])/nT, Sty[ABindex.index(ab)], linewidth = lw, color = "#fbceb1", label="$S_1$: Low")# thermotolerance")
                        #ax1.plot(Time, (Trait[1, :])/nT, Sty[ABindex.index(ab)], linewidth = lw, color = "#d99058", label="$S_2$: Mid")# thermotolerance")
                        #ax1.plot(Time, (Trait[0, :])/nT, Sty[ABindex.index(ab)], linewidth = lw, color = "#964b00", label="$S_3$: High")# thermotolerance")
        
                        
                        # plot symbiont abundance as 10^6 cells per cm2 coral surface
                    
                        #ax2.plot(Time, (Symb[2, :]/Coral)/1e6, Sty[ABindex.index(ab)], linewidth = lw, color = "#fbceb1", label="$S_1$: Low")# thermotolerance")
                        #ax2.plot(Time, (Symb[1, :]/Coral)/1e6, Sty[ABindex.index(ab)], linewidth = lw, color = "#d99058", label="$S_2$: Mid")# thermotolerance")
                        #ax2.plot(Time, (Symb[0, :]/Coral)/1e6, Sty[ABindex.index(ab)], linewidth = lw, color = "#964b00", label="$S_3$: High")# thermotolerance")  
                        ax2.plot(Time, (Symb[2, :]/Coral)/1e6, linewidth = lw*ab/max(ABindex), color = "#fbceb1", label="$S_1$: Low")# thermotolerance")
                        ax2.plot(Time, (Symb[1, :]/Coral)/1e6, linewidth = lw*ab/max(ABindex), color = "#d99058", label="$S_2$: Mid")# thermotolerance")
                        ax2.plot(Time, (Symb[0, :]/Coral)/1e6, linewidth = lw*ab/max(ABindex), color = "#964b00", label="$_3$: High")# thermotolerance")  
                    
                    elif Forc[f] == "Forcing2":
                        fileCoral = open(cases[cas]+str(Tbar0)+"/Coral/Coral_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+dur_str[dl]+".dat", "r")
                        fileTrait = open(cases[cas]+str(Tbar0)+"/Trait/Trait_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+dur_str[dl]+".dat", "r")
                        fileSymb = open(cases[cas]+str(Tbar0)+"/Symbiont/Symbiont_Tk%d_Tl%d_"%(i4, i5)+Forc[f]+A_Bstr[ab]+dur_str[dl]+".dat", "r")        
                        Temperature = Forcing2(0, t0, t1, tk, tl, Tbar0, AorB)
                        Coral = load(fileCoral)
                        Trait = load(fileTrait)
                        Symb = load(fileSymb)
                
                        fileCoral.close()
                        fileTrait.close()
                        fileSymb.close()    
                        # plot coral abundance as normalized biomass
                        #ax1.plot(Time, Coral/K_C, Sty[ABindex.index(ab)], linewidth = lw, color = "#0F5738")
                        ax1.plot(Time, Coral/K_C, linewidth = lw*(ab/max(ABindex)), color = "#0F5738")

                        # plot normalized coral trait 
                        #nT = max(max(Trait[0, :]), max(Trait[1, :]), max(Trait[2, :]))
                        #ax1.plot(Time, (Trait[2, :])/nT, Sty[ABindex.index(ab)], linewidth = lw, color = "#fbceb1", label="$S_1$: Low")# thermotolerance")
                        #ax1.plot(Time, (Trait[1, :])/nT, Sty[ABindex.index(ab)], linewidth = lw, color = "#d99058", label="$S_2$: Mid")# thermotolerance")
                        #ax1.plot(Time, (Trait[0, :])/nT, Sty[ABindex.index(ab)], linewidth = lw, color = "#964b00", label="$S_3$: High")# thermotolerance")
        
                        
                        # plot symbiont abundance as 10^6 cells per cm2 coral surface
                    
                        #ax2.plot(Time, (Symb[2, :]/Coral)/1e6, Sty[ABindex.index(ab)], linewidth = lw, color = "#fbceb1", label="$S_1$: Low")# thermotolerance")
                        #ax2.plot(Time, (Symb[1, :]/Coral)/1e6, Sty[ABindex.index(ab)], linewidth = lw, color = "#d99058", label="$S_2$: Mid")# thermotolerance")
                        #ax2.plot(Time, (Symb[0, :]/Coral)/1e6, Sty[ABindex.index(ab)], linewidth = lw, color = "#964b00", label="$S_3$: High")# thermotolerance")  
                        
                        ax2.plot(Time, (Symb[2, :]/Coral)/1e6, linewidth = lw*ab/max(ABindex), color = "#fbceb1", label="$S_1$: Low")# thermotolerance")
                        ax2.plot(Time, (Symb[1, :]/Coral)/1e6, linewidth = lw*ab/max(ABindex), color = "#d99058", label="$S_2$: Mid")# thermotolerance")
                        ax2.plot(Time, (Symb[0, :]/Coral)/1e6, linewidth = lw*ab/max(ABindex), color = "#964b00", label="$S_3$: High")# thermotolerance")  
                                     
                                                  
                    
        if k in (0, 1):
            print k, tk, tl, t0, t1, A_B[ab]
            for ab in ABindex:
                Forcing1A = concatenate((Tbar0*ones(len(Time[Time<t0])), Tbar0 + A_B[ab]*((Time[Time>=t0] - t0)**tl/(tk**tl + (Time[Time>=t0]-t0)**tl))))
                Forcing2A = zeros(len(Time))
                for t in xrange(len(Time)):
                    Forcing2A[t] = Forcing2(Time[t], t0, t1, tk, tl, Tbar0, A_B[ab])
                    
        
                #phase0 = Tbar0*ones(len(Time[Time<t0]))
                #phase1 = ((Time[(Time>=t0)*(Time<t1)] - t0)**tl/(tk**tl + (Time[(Time>=t0)*(Time<t1)]-t0)**tl))
                #phase2 = (1-((Time[Time>=t1] - t1)**tl/(tk**tl + (Time[Time>=t1]-t1)**tl)))
                
                #Forcing2A = concatenate((phase0, Tbar0+A_B[ab]*phase1, Tbar0+A_B[ab]*phase2))
                           
                if k in (0, 1, 3, 4, 5, 6):
                    #ax0.plot(Time, Forcing2A, Sty[ABindex.index(ab)], linewidth = lw, color="#4F628E")
                    ax0.plot(Time, Forcing2A, linewidth = lw*ab/max(ABindex), color="#4F628E")
                    
    if k in (0, 2, 4):
        ax0.text(-2, 29, Fig_labels[k], fontproperties=font, fontsize = fsize)
        ax1.text(-2, 1, Fig_labels[2+k], fontproperties=font, fontsize = fsize)
        ax2.text(-2, 3.5, Fig_labels[4+k], fontproperties=font, fontsize = fsize)
    else:
        ax0.text(-1, 29, Fig_labels[k], fontproperties=font, fontsize = fsize)
        ax1.text(-1, 1, Fig_labels[2+k], fontproperties=font, fontsize = fsize)
        ax2.text(-1, 3.5, Fig_labels[4+k], fontproperties=font, fontsize = fsize)            
    
    #if 5+k == 6:
        #ax2.legend(loc = (-1, -0.75), ncol = 3, fontsize = fsize)
    k += 1   
"""   