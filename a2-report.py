#CSC336 Assignment #2 starter code for the report question

#These are some basic imports you will probably need,
#but feel free to add others here if you need them.
import numpy as np
import numpy.linalg as LA
from scipy.sparse import diags
import scipy.linalg as sla
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.linalg import solve, solve_triangular, solve_banded
import time
import matplotlib
import matplotlib.pyplot as plt

"""
See the examples in class this week or ask on Piazza if you
aren't sure how to start writing the code
for the report questions.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.
"""

"""
timing code sample (feel free to use timeit if you find it easier)
#you might want to run this in a loop and record
#the average or median time taken to get a more reliable timing
start_time = time.perf_counter()
#your code here to time
time_taken = time.perf_counter() - start_time
"""

def create_matrix(n):
    x = [9] + [6]*(n-3) + [5, 1]
    y = [-4]*(n-2) + [-2]
    z = [1]*(n-2)
    diagonals = [z,y,x,y,z]
    md = diags(diagonals, [-2,-1,0,1,2]).toarray()
    m = diags(diagonals, [-2,-1,0,1,2], format = "csr")
    return md, m


def create_b(n):
    b = [-1/n**4]*n
    return b


def LU(n):
    A = create_matrix(n)[0]
    b = create_b(n)
    start_time = time.perf_counter()
    x = sla.solve(A,b)
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken


def banded_LU(n):
    ab = np.array([[0,0] + [1]*(n-2), [0] + [-4]*(n-2) + [-2], [9] + [6]*(n-3) + [5, 1], [-4]*(n-2) + [-2] + [0], [1]*(n-2) + [0,0]])
    b = create_b(n)
    start_time = time.perf_counter()
    x = sla.solve_banded((2, 2), ab, b)
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken    


def sparse_LU(n):
    A = create_matrix(n)[1]
    b = create_b(n)
    start_time = time.perf_counter()
    x = spsolve(A,b)
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken    


def prefactored(n):
    b = create_b(n)
    #make R
    x = [2] + [1]*(n-1)
    y = [-2]*(n-1)
    z = [1]*(n-2)
    diagonals = [x,y,z]
    R = diags(diagonals, [0,1,2]).toarray()
    RT = R.transpose()
    start_time = time.perf_counter()
    y = sla.solve_triangular(R,b, lower = False)
    x = sla.solve_triangular(RT,y,lower = True)    
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken


def banded_pre(n):
    b = create_b(n)   
    rb = np.array([[0,0] + [1]*(n-2), [0] +[-2]*(n-1), [2] + [1]*(n-1)])
    rtb = np.array([[2] + [1]*(n-1), [-2]*(n-1) + [0], [1]*(n-2) + [0,0]])
    start_time = time.perf_counter()
    y = sla.solve_banded((0, 2), rb, b)
    x = sla.solve_banded((2, 0), rtb, y)
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken    


def sparse_pre(n):
    x = [2] + [1]*(n-1)
    y = [-2]*(n-1)
    z = [1]*(n-2)
    diagonals = [x,y,z]
    R = diags(diagonals, [0,1,2], format = "csr")
    RT = R.transpose()
    b = create_b(n)
    start_time = time.perf_counter()
    y = spsolve_triangular(R,b, lower = False)
    x = spsolve_triangular(RT,y,lower = True)   
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken    


def cho(n):
    A = create_matrix(n)[0]
    b = create_b(n)
    start_time = time.perf_counter()
    c, low = sla.cho_factor(A, lower=False, overwrite_a=False, check_finite=True)
    x = sla.cho_solve((c,low), b, overwrite_b=False, check_finite=True)
    time_taken = (time.perf_counter() - start_time)*1000
    return x, time_taken    
    
   

def exp():
    print('\         |    |        |         |       |         |         |      |')
    print(' \ method | LU | banded |sparse LU|   R   | banded R| sparse R| chol |')
    print('n \       |    |        |         |       |         |         |      |')
    print('------------------------------------------------------------------')
    for i in range(1,9):
        n = 200*i 
        lu = LU(n)[1]
        blu = banded_LU(n)[1]
        slu = sparse_LU(n)[1]
        r = prefactored(n)[1]
        br = banded_pre(n)[1]
        spr = sparse_pre(n)[1]
        chol = cho(n)[1]
        print(" {}  | {:.2f}ms | {:.2f}ms | {:.2f}ms | {:.2f}ms | {:.2f}ms | {:.2f}ms | {:.2f}ms".format(n, lu, blu, slu, r, br, spr, chol))






def get_true_sol(n):
    """
    returns the true solution of the continuous model on the mesh,
    x_i = i / n , i=1,2,...,n.
    """
    
    x = np.linspace(1/n,1,n)
    d = (1/24)*(-(1-x)**4 + 4*(1-x) -3)
    return d



def compare_to_true(d):
    """
    produces plot similar to the handout,
    the input is the solution to the n x n banded linear system,
    this is one way to visually check if your code is correct.
    """
    dtrue = get_true_sol(100) #use large enough n to make plot look smooth

    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 14})
    plt.title("Horizontal Cantilevered Bar")
    plt.xlabel("x")
    plt.ylabel("d(x)")

    xtrue = np.linspace(1/100,1,100)
    plt.plot(xtrue,dtrue,'k')

    n = len(d)
    x = np.linspace(0,1,n+1)
    plt.plot(x,np.hstack([0,d]),'--r')

    plt.legend(['exact',str(n)])
    plt.grid()
    plt.show()
    

def q2():   
    plt.figure()
    size = []
    b = []
    br = []
    for i in range(4,17):
        n = 2**i
        size.append(n)
        true = get_true_sol(n)
        banded = banded_LU(n)[0]
        bandedr = banded_pre(n)[0]
        rel_errs = np.divide(sla.norm(np.abs(banded-true),np.inf), sla.norm(true, np.inf))
        b.append(rel_errs)
        rel_errsr = np.divide(sla.norm(np.abs(bandedr-true),np.inf), sla.norm(true, np.inf)) 
        br.append(rel_errsr)
    plt.loglog(size,b,basex = 2)
    plt.loglog(size,br,'--r',basex=2)    
    plt.title("Relative error table")
    plt.xlabel("size of A")
    plt.ylabel("relative error")  
    plt.legend(['banded LU','banded prefactored'])
    plt.show()
    plt.savefig("q2.png")



    

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    d = np.zeros(8)
    compare_to_true(sparse_pre(100)[0])