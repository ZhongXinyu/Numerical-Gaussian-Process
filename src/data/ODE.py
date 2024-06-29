
# implement the ODE solver using Forward Euler method, and Backward Euler method and Runge-Kutta method
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import jax.numpy as jnp

class ODE_solver:
    def __init__(self) -> None:
        pass

    
    def forward_euler(self, f, y0, t):
        """
        Forward Euler method to solve ODE

        Args:
        f: function, the right hand side of the ODE
        y0: array, the initial value of the ODE
        t: array, the time steps to solve the ODE

        Returns:
        y: array, the solution of the ODE
        """
        y = np.zeros((len(t), y0.shape[1]))
        y[0] = y0
        for i in range(1, len(t)):
            y[i] = y[i-1] + f(y[i-1])*(t[i] - t[i-1])
        return y
    
    def backward_euler(self, f, y0, t):
        """
        Backward Euler method to solve ODE

        Args:
        f: function, the right hand side of the ODE
        y0: array, the initial value of the ODE
        t: array, the time steps to solve the ODE

        Returns:
        y: array, the solution of the ODE
        """
        y = np.zeros((len(t), y0.shape[1]))
        print (y.shape)
        # initial value 
        y[0] = y0
        for i in range(1, len(t)):
            print (i)
            y[i] = fsolve(lambda x: x - y[i-1] - f(x)*(t[i] - t[i-1]), y[i-1])
        return y
    
    def trapezoidal(self, f, y0, t):
        """
        Trapezoidal method to solve ODE

        Args:
        f: function, the right hand side of the ODE
        y0: array, the initial value of the ODE
        t: array, the time steps to solve the ODE

        Returns:
        y: array, the solution of the ODE
        """
        y = np.zeros((len(t), y0.shape[1]))
        y[0] = y0
        for i in range(1, len(t)):
            h = t[i] - t[i-1]
            y[i] = fsolve(lambda x: x - y[i-1] - h/2*(f(x) + f(y[i-1])), y[i-1])
        return y

    def Runge_Kutta(self, f, y0, t):
        """
        Runge-Kutta method to solve ODE

        Args:
        f: function, the right hand side of the ODE
        y0: array, the initial value of the ODE
        t: array, the time steps to solve the ODE

        Returns:
        y: array, the solution of the ODE
        """
        y = np.zeros((len(t), y0.shape[1]))
        y[0] = y0
        for i in range(1, len(t)):
            h = t[i] - t[i-1]
            k1 = f(y[i-1])
            # check if k1 is infinite
            if not np.all(np.isfinite(k1)):
                print (f"k1 is infinite at time step {i}")
                break
            k2 = f(y[i-1] + h/2*k1)
            k3 = f(y[i-1] + h/2*k2)
            k4 = f(y[i-1] + h*k3)
            y[i] = y[i-1] + h/6*(k1 + 2*k2 + 2*k3 + k4)
        return y