import numpy as np

def convert_to_ellipticity(rho, tau):

    a = 0.05333
    d_a = 0.03357
    d_p = 1.628
    d_w = 0.2951
    g_a = 0.0901
    g_p = 0.4553
    g_w = 0.4132

    return rho * (a + d_a*(np.arctan((tau - d_p)/d_w)) + g_a*np.exp(-((tau - g_p)**2.0)/(2*g_w**2.0)))

