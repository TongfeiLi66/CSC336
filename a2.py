#CSC336 Assignment #2 starter code

import numpy as np
import scipy.linalg as sla
import time
#Q4a
def p_to_q(p):
    """
    return the permutation vector, q, corresponding to
    the pivot vector, p.
    >>> p_to_q(np.array([2,3,2,3]))
    array([2, 3, 0, 1])
    >>> p_to_q(np.array([2,4,8,3,9,7,6,8,9,9]))
    array([2, 4, 8, 3, 9, 7, 6, 0, 1, 5])
    """
    n = len(p)
    q = np.arange(n) #replace with your code
    for i in range (n-1):
        ind = p[i]
        tgt = q[ind]
        q[ind] = q[i]
        q[i] = tgt
    return q

#Q4b
def solve_plu(A,b):
    """
    return the solution of Ax=b. The solution is calculated
    by calling scipy.linalg.lu_factor, converting the piv
    vector using p_to_q, and solving two triangular linear systems
    using scipy.linalg.solve_triangular.
    """
    lu,piv = sla.lu_factor(A)
    q = p_to_q(piv)
    n = len(b)
    l,u = np.tril(lu,k=-1) + np.eye(n),np.triu(lu)
    Pb = []
    for i in range(n):
        Pb.append(b[q[i]])
    y = sla.solve_triangular(l,Pb,lower = True)
    x = sla.solve_triangular(u,y,lower = False)   
    #replace with your code
    print(x)
    return x


def table3():
    print('  n   | run time 1 | run time 2 | operation count 1 | operation count 2 | time efficiency of 2 to 1 | operation efficiency of 2 to 1')
    for i in range(1,9):
        n = 200*i
        A = np.random.random((n, n))
        B = np.random.random((n, n)) 
        C = np.random.random((n, n))
        b = np.random.random((n, 1))
        start_time = time.perf_counter()
        D = 2 * A + np.identity(n) #D = 2A+1
        Binv = sla.inv(B)
        Y = Binv.dot(D) #Y = B-1(2A+1)
        Cinv = sla.inv(C)
        Z = Cinv + A #Z=C-1 + A
        Z = Y.dot(Z)
        x = Z.dot(b)
        time_taken1 = (time.perf_counter() - start_time)*1000
        
        start_time = time.perf_counter()
        y = sla.solve(C,b)
        D = 2 * A + np.identity(n) #D = 2A+1
        Z = A.dot(b)
        Z = y + Z #y+Ab
        Z = D.dot(Z) #(2A + I)(y + Ab)
        x = sla.solve(B,Z)#solve Bx = (2A + I)(y + Ab)
        time_taken2 = (time.perf_counter() - start_time)*1000
        op_count1 = 4 * n**3 + 4 * n**2
        op_count2 = 2/3 * n**3 + 6 * n**2 + n
        print(" {}  |   {:.2f}ms   |   {:.2f}ms   |   {:.2e}ops   |    {:.2e}ops    |       {:.2f}times       |    {:.2f}times".format(n, time_taken1,time_taken2,op_count1,op_count2,time_taken1/time_taken2,op_count1/op_count2)) 
        
        
        
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    #test your solve_plu function on a random system
    n = 10
    A = np.random.uniform(-1,1,[n,n])
    b = np.random.uniform(-1,1,n)
    xtrue = sla.solve(A,b)
    x = solve_plu(A,b)
    print("solve_plu works:",np.allclose(x,xtrue,rtol=1e-10,atol=0))