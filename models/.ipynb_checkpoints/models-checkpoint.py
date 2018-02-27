from scipy.interpolate import interp1d
import numpy as np

class Model(object):
    def __init__(self):
        pass
    def output(self):
        pass
    def step(self):
        pass
 
class FirstOrder(Model):
    """ 
    This is First order  process, described by differential equation dydt = (ku-y)/tau.  In a first order process ,  the rate of change is           directly proportional to the driving force with the proportionality constant being 1/tau. The driving force is (ku - y) . As the  'y' gets       closer to 'ku' and driving force keeps on decreasing finally leading to zero , and the process reaches the steady state. 
    """ 
    def __init__(self,k,tau,dt):
        self.k = k
        self.tau = tau
        self.dt = dt
    
    def sim(self,u):
        y = 0
        ys = []
        ts = range(len(u))
        uf= interp1d(ts,u)
        for t in ts:

            if (t-self.dt) < 0:
                dydt=0
            else:
                dydt =  (self.k*uf(t-self.dt) - y)/self.tau  

            y += dydt
            ys.append(y)
        y_arr = np.array(ys)
        return y_arr 
    
    def step():
        """
        By default, a step signal , with length equal to 5 times the time contant is used to generate the output
        """
        step_signal = np.zeros(int(5*self.tau))
        step_signal[1:] = 1
        return step_signal



class Ramp(Model):
    """ 
    This is First order Ramp process,  y = ku t  where the ramp gain follows a first order process.  
    For example , flow vs level is a ramp process. When you inrease a flow set point , the flow PV itself may  follow a first order proess ,
    so a term dynamic gain is used (dyn_k). This dyn_k reaches the steady state value "k". The "first order" dynamics of the gain  is not           obviously visible in the step response or simulation . However , this is clearly visible in the impulse response.  
    """ 
    def __init__(self,u,k,tau,dt):
        self.k =k
        self.tau = tau
        self.dt = dt

    def sim(self,u):
        y = 0
        dyn_k = 0 
        ys = []
        ts = range(len(u))
        uf= interp1d(ts,u)
        for t in ts:

            if (t-self.dt) < 0:
                dkdt = 0
            else:
                dkdt =  (self.k*uf(t-self.dt) - dyn_k )/self.tau
                dyn_k += dkdt

            y += dyn_k
            ys.append(y)
        y_arr = np.array(ys)
        return y_arr

class SecondOrder(Model):
    """
    This is a second order process described differential equation tau2^2 d2ydt2 + tau1 dydt = (ku -y)
    This model is used to define the systems with underdamped , critically damped, overdamped systems. To define systems with inverese               response,and overshoot response use secondorder2 
    """    
    def __init__(self,k,tau1,tau2,dt):
        self.k = k
        self.tau1 = tau1
        self.tau2 = dt
        self.dt = dt
    
    def sim(self,u):
        y  = 0
        dy2dt2 = 0
        dydt = 0
        ys = []
        ts = range(len(u))
        u_int = interp1d(ts,u)
        for t in ts :
            if (t-self.dt ) < 0:
                dy2dt2 = 0
                dydt = 0
            else:
                dy2dt2 = (self.k*u_int(t-self.dt) - y - self.tau1*dydt) /(self.tau2*self.tau2)  

                dydt = dydt + dy2dt2

            y += dydt
            ys.append(y)
        y_arr = np.array(ys)
        return y_arr

class SecondOrder2(Model):
    """
    This is a second order process described by two parallel processs described by two differential equations
    dy1dt1 = (k1u -y1 ) /tau1
    dy2dt2 = (k2u - y2) /tau2
    y = dy1dt1 + dy2dt2
    This allows to define inverse response systems , when k1 and k2 are in opposite directions and tau1 is very short,
    compared to tau2.
    Eg: In  columns with material balance control scheme , where the accumulator level is controlled by manipulating the reflux,
    the relationship between bottom temperatre setpoint and the top product quality  follow the inverse response. As the bottom temperature         increased ,     the vapor  carries more heavier content.So the impurity in distillate increases first. However , as the accumulator level       increases due increased vapor traffic, this increase the reflux flow. As the sharpness of separation increases , the impurity level             decreases and reaches a new         steady state value lower than intitial value. 
    This can be also used to define the overshoot response system, if both k1 and k2 are in same direction. 
    """ 
    def __init__(self,k1,k2,tau1,tau2,dt):
        
        self.k1 = k1
        self.k2 = k2
        self.tau1 = tau1
        self.tau2 = tau2
        self.dt = dt
        self.k = self.k1 + self.k2
    
    def sim(self,u):
        
        y1  = 0
        y2 = 0
        dy1dt =0
        dy2dt = 0
        dydt = 0
        ys = []
        ts = range(len(u))
        u_int = interp1d(ts,u)
        for t in ts :
            if (t-self.dt ) < 0:
                dy1dt = 0
                dy2dt =0
                dydt = 0
            else:
                dy1dt = (self.k1*u_int(t-self.dt) - y1) /self.tau1  
                y1 += dy1dt
                dy2dt = (self.k2*u_int(t-self.dt) - y2) /self.tau2
                y2 += dy2dt
            y += (dy1dt+dy2dt)
            ys.append(y)
        y_arr = np.array(ys)
        return y_arr 
