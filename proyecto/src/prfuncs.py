import numpy as np
from copy import copy
from pyquaternion import Quaternion

pi = np.pi

# Matriz de transformación homogénea a partir de parámetros de Denavit-Hartenberg
def dh(d, theta, a, alpha):
 sth = np.sin(theta)
 cth = np.cos(theta)
 sa  = np.sin(alpha)
 ca  = np.cos(alpha)
 T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
               [sth,  ca*cth, -sa*cth, a*sth],
               [0.0,      sa,      ca,     d],
               [0.0,     0.0,     0.0,   1.0]])
 return T


# Cinemática directa de un robot de 6GDL a partir de transformaciones de DH
def fkine(q):
 T1 = dh(0.3231,   pi/2+q[0],  0.0,    pi/2)
 T2 = dh(0.0,      pi/2+q[1],  0.2105, 0.0)
 T3 = dh(0.0,      pi/2+q[2],  0.0,    pi/2)
 T4 = dh(0.2641,   pi+q[3],    0.0,    pi/2)
 T5 = dh(0.0,      pi+q[4],    0.0,    pi/2)
 T6 = dh(0.1636,   q[5],       0.0,    0.0)
 T = T1 @ T2 @ T3 @ T4 @ T5 @ T6
 return T

def TF2xyzquat(T):
 quat = Quaternion(matrix=T[0:3,0:3])
 return np.array([T[0,3], T[1,3], T[2,3], quat.w, quat.x, quat.y, quat.z])