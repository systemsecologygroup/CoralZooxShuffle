from __future__ import division

from scipy import arange, concatenate, ones
from matplotlib import pyplot as plt

step = 0.001
t0 = 2
Time = arange(0, t0+10+step, step)

A = 0.4
B = 0.4

FParams = [(1.25, 20)]#, (1.5, 20), (1.75, 20)]
lw = 2.5
plt.figure()
for tk,tl in FParams:
    Forcing1A = concatenate((27*ones(len(Time[Time<t0])), 27 + A*((Time[Time>=t0] - t0)**tl/(tk**tl + (Time[Time>=t0]-t0)**tl))))
    Forcing1B = concatenate((27*ones(len(Time[Time<t0])), 27 + B*((Time[Time>=t0] - t0)**tl/(tk**tl + (Time[Time>=t0]-t0)**tl))))
    if FParams.index((tk, tl)) == 0:
        plt.plot(Time, Forcing1A, linewidth = lw, color = "#4F628E", label = "Forcing 1A")
        #plt.plot(Time, Forcing1B, "--", linewidth = lw, color = "#4F628E", label = "Forcing 1B")
    else:
        plt.plot(Time, Forcing1A, linewidth = lw, color = "#4F628E")
        #plt.plot(Time, Forcing1B, "--", linewidth = lw, color = "#4F628E")
plt.legend()
plt.xlim((t0, t0+10))
#plt.figure()

dur_steps = 0.5/12  # step of half a month
dur_list = arange(0, 4+dur_steps, dur_steps) # list of duration of stress, by steps of half a month 
t1 = t0+1#2.8333333333

for tk,tl in FParams:
    #phase0 = 27*ones(len(Time[Time<t0]))
    phase1 = Forcing1A[(Time<t1)]#((Time[(Time>=t0)*(Time<t1)] - t0)**tl/(tk**tl + (Time[(Time>=t0)*(Time<t1)]-t0)**tl))
    if (tk,tl) == (1.25, 20):
        for t in xrange(len(phase1)):
           print t, t0, t1, phase1[t]
    #phase2 = (1-((Time[Time>=t1] - t1)**tl/(tk**tl + (Time[Time>=t1]-t1)**tl)))
    phase2 = 27+A*(1-((Time - t1)**tl/(tk**tl + (Time-t1)**tl)))
    #if (tk,tl) == (1.25, 20):
        #for t in xrange(len(phase2)):
           #print t, t0, t1, phase2[t]

    #Forcing2A = concatenate((phase0, 27+A*phase1, 27+A*phase2))
    mark = t1+0.5#+t1#+0.05
    Forcing2A = concatenate((Forcing1A[Time<mark], phase2[Time>=mark]))#concatenate((phase1, 27+A*phase2))
    #Forcing2B = concatenate((phase0, 27+B*phase1, 27+B*phase2))
    if FParams.index((tk, tl)) == 0:
        plt.plot(Time[Time<mark], Forcing1A[Time<mark], linewidth = lw, color = "red")
        plt.plot(Time[Time>=mark], phase2[Time>=mark], linewidth = lw, color = "green")
        
        plt.plot(Time, Forcing2A, "--", linewidth = lw, color = "#4F628E", label = "Forcing 2A")
        #plt.plot(Time[(Time>=t0)*(Time<t1)], 27*ones(len(Time[(Time>=t0)*(Time<t1)])), linewidth = lw)
        #plt.plot(Time, Forcing2B, "--", linewidth = lw, color = "#4F628E", label = "Forcing 2B")
    else:
        plt.plot(Time, Forcing2A, linewidth = lw, color = "#4F628E")
        #plt.plot(Time, Forcing2B, "--", linewidth = lw, color = "#4F628E")

plt.xlim((t0, t0+10))

#plt.xlim((4, 7))
plt.legend()
plt.show()
