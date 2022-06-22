import scipy.linalg as sla
import numpy as np
import numpy.linalg as LA
from scipy.linalg import solve, inv
from scipy.stats import ortho_group
import time

def a_inv(A, atol = 1e-14):
    x = A.transpose() / np.multiply(LA.norm(A,1),LA.norm(A,np.inf))
    err = 1
    size = len(A[0])
    while(err > atol):
        r = np.identity(size) - A @ x
        x = x + x @ r
        err = np.abs(LA.norm(r,np.inf))
    return x, err


def q1():
    print("|        Accuracy and efficiency for getting A inverse          |")
    print("|             Newton's              |           sla.inv         |")
    print("size |    runtime   |     error     |    runtime   |    error   |")
    for i in range(1,10):
        size = 50 * i
        A = ortho_group.rvs(size)
        
        start_time = time.perf_counter()
        a_inv1, err1 = a_inv(A)
        time_taken = (time.perf_counter() - start_time)*1000  
        t1 = time_taken
        
        
        start_time = time.perf_counter()
        a_inv2 = sla.inv(A)
        time_taken = (time.perf_counter() - start_time)*1000  
        t2 = time_taken  
        err2 = np.abs(LA.norm(np.identity(size) - A @ a_inv2,np.inf))
        
        print(" {}  |    {:.2f}ms    |    {:.2e}   |    {:.2f}ms   |    {:.2e}   |".format(size,t1,err1,t2,err2))
    return
        
    