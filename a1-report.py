#CSC336 Assignment #1 starter code for the report question
import numpy as np

"""
See the examples in class this week if you
aren't sure how to start writing the code
to run the experiment and produce the plot for Q1.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.

A few things you'll likely find useful:

import matplotlib.pyplot as plt

hs = np.logspace(-15,-1,15)
plt.figure()
plt.loglog(hs,rel_errs)
plt.show() #displays the figure
plt.savefig("myplot.png")



a_cmplx_number = 1j #the j in python is the complex "i"

try to reuse code where possible ("2 or more, use a for")

"""

#example function header, you don't have to use this
def fd(f, x, h):
    """
    Return the forward finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    return np.divide(f(x + h) - f(x), h)


def cd(f, x, h):
    return np.divide(f(x + h) - f(x - h), 2*h)

def csm(f, x, h):
    return np.divide(np.imag(f(x+np.multiply(1j, h))), h)

def fun(x):
    return np.divide(np.exp(x), np.sqrt(np.sin(x)**3 + np.cos(x)**3))

 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    hs = np.logspace(-15,-1,15)
    plt.figure()
    act = 4.05342789389862
    e1 = []
    e2 = []
    e3 = []
    
    for h in hs:
        exp1 = fd(fun, 1.5, h)
        exp2 = cd(fun, 1.5, h)
        exp3 = csm(fun, 1.5, h)  
        
        rel_errs1 = np.divide(np.abs(act-exp1), exp1)
        rel_errs2 = np.divide(np.abs(act-exp2), exp2)
        rel_errs3 = np.divide(np.abs(act-exp3), exp3)
        e1.append(rel_errs1)
        e2.append(rel_errs2)
        e3.append(rel_errs3)
        
    plt.loglog(hs,e1)
    plt.loglog(hs,e2)
    plt.loglog(hs,e3)
    
    plt.xlabel('h')
    plt.ylabel('relative error')
    plt.title('hs against relative error')
    plt.show() #displays the figure
    plt.savefig("myplot.png")        
    
    
    #import doctest
    #doctest.testmod()
    pass