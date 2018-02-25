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

def ramp(u,k,tau,dt):
    y = 0
    ys = []
    ts = range(len(u))
    uf= interp1d(ts,u)
    for t in ts:
    
        if (t-dt) < 0:
            dydt=0
        else:
            dydt =  (k*uf(t-dt))/tau # look at the brackets  , instead of square brackets. 
            
        y += dydt
        ys.append(y)
    y_arr = np.array(ys)
    return y_arr

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
 