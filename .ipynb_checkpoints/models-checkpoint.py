from scipy.interpolate import interp1d
import numpy as np
def fopdt(u,k,tau,dt):
    y = 0
    ys = []
    ts = range(len(u))
    uf= interp1d(ts,u)
    for t in ts:
    
        if (t-dt) < 0:
            dydt=0
        else:
            dydt =  (k*uf(t-dt) - y)/tau # look at the brackets  , instead of square brackets. 
            
        y += dydt
        ys.append(y)
    y_arr = np.array(ys)
    return y_arr

# for a ramp process , the ramp gain itself can follow a first order process. 
# for example , when you inrease a flow set point , the flow PV can follow a first order proess
# so a term dynamic gain is used (dyn_k) .  The "k" is the steady state gain per min. 
def ramp(u,k,tau,dt):
    y = 0
    dyn_k = 0 
    ys = []
    ts = range(len(u))
    uf= interp1d(ts,u)
    for t in ts:
    
        if (t-dt) < 0:
            roc_dydt=0
        else:
            dkdt =  (k*uf(t-dt) - dyn_k )/tau
            dyn_k += dkdt
            
        y += dyn_k
        ys.append(y)
    y_arr = np.array(ys)
    return y_ar

# in a first order system , the rate of change is proportional to driving force. 
# in a second order system , there are two driving forces 
# first  driving force is between dydt and d2ydt2
# second driving force is between dydt and y 

def secondorder(u,k,tau1,tau2,dt):
    y  = 0
    dy2dt2 = 0
    dydt = 0
    ys = []
    ts = range(len(u))
    u_int = interp1d(ts,u)
    for t in ts :
        if (t-dt ) < 0:
            dy2dt2 = 0
            dydt = 0
        else:
            dy2dt2 = (k*u_int(t-dt) - y - tau1*dydt) /(tau2*tau2)  # driving force for d2ydt2
            
            dydt = dydt + dy2dt2
            
        y += dydt
        ys.append(y)
    y_arr = np.array(ys)
    return y_arr
 
def secondorder2(u,k1,k2,tau1,tau2,dt):
    y1  = 0
    y2 = 0
    dy1dt =0
    dy2dt = 0
    dydt = 0
    ys = []
    ts = range(len(u))
    u_int = interp1d(ts,u)
    for t in ts :
        if (t-dt ) < 0:
            dy1dt = 0
            dy2dt =0
            dydt = 0
        else:
            dy1dt = (k1*u_int(t-dt) - y1) /tau1  # driving force for dy1dt
            y1 += dy1dt
            dy2dt = (k2*u_int(t-dt) - y2) /tau2
            y2 += dy2dt
            
            
        y += (dy1dt+dy2dt)
        ys.append(y)
    y_arr = np.array(ys)
    return y_arr    