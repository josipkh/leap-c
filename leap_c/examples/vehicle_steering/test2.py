import numpy as np

m = 1.0     # [kg] mass
L = 1.0     # [m] half of the field size
dt = 0.1    # [s] sampling time
b = 1.0     # [kg/s] friction coefficient 

# continuous-time matrices
Ac = np.array([
    [0.0, 0.0,  1.0,  0.0],
    [0.0, 0.0,  0.0,  1.0],
    [0.0, 0.0, -b/m,  0.0],
    [0.0, 0.0,  0.0, -b/m],
])
Bc = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1/m, 0.0],
    [0.0, 1/m]
])

# discretization using Euler's method
Ad = np.eye(4)+Ac*dt
Bd = Bc*dt

# T = np.array([
    
# ])