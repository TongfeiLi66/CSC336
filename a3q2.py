#CSC 336 Summer 2020 A3Q2 starter code

#Note: you may use the provided code or write your own, it is your choice.

#some general imports
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt

def get_dn(fn):
    """
    Returns dn, given a value for fn, where Ad = b + f (see handout)

    The provided implementation uses the solve_banded approach from A2,
    but feel free to modify it if you want to try to more efficiently obtain
    dn.

    Note, this code uses a global variable for n.
    """

    #the matrix A in banded format
    diagonals = [np.hstack([0,0,np.ones(n-2)]),#zeros aren't used, so can be any value.
                 np.hstack([0,-4*np.ones(n-2),-2]),
                 np.hstack([9,6*np.ones(n-3),[5,1]]),
                 np.hstack([-4*np.ones(n-2),-2,0]),#make sure this -2 is in correct spot
                 np.hstack([np.ones(n-2),0,0])] #zeros aren't used, so can be any value.
    A = np.vstack(diagonals) * n**3

    b = -(1/n) * np.ones(n)

    b[-1] += fn #b + f

    sol = solve_banded((2, 2), A, b)
    dn = sol[-1]
    return dn



def q2b():
    
    return brentq(get_dn, -100, 100)
    

def q2c():
    
    return fsolve(get_dn, np.array([2]))[0]


def q2e():
    x = 5
    x = x - get_dn(x)/(get_dn(x) - get_dn(x-1))
    return x



if __name__ == "__main__":
    #experiment code
    exp = 3/8
    print("|                                  Accuracy and computational efficiency of b c and e                                   |")
    print("|                    b                       |                   c                 |                  e                 |")
    print("size |    runtime    |          error        |    runtime    |         error       |    runtime    |        error       |")    
    for i in range(5,17):
        n = 2**i
        #your code here
        
        start_time = time.perf_counter()
        b = q2b()
        time_taken = (time.perf_counter() - start_time)*1000  
        tb = time_taken  
        rel_errb = np.divide(abs(exp - b), exp)
        
        start_time = time.perf_counter()
        c = q2c()
        time_taken = (time.perf_counter() - start_time)*1000  
        tc = time_taken    
        rel_errc = np.divide(abs(exp - c), exp)
        
        start_time = time.perf_counter()
        e = q2e()
        time_taken = (time.perf_counter() - start_time)*1000
        te = time_taken  
        rel_erre = np.divide(abs(exp - e), exp)
        
        print(" {} |    {:.2f}ms    |     {:.10e}   |     {:.2f}ms   |    {:.10e}    |   {:.2f}ms   |    {:.10e}   |".format(n,tb,rel_errb,tc,rel_errc,te,rel_erre))
        
    