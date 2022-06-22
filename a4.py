import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import CubicSpline
import time

years = np.linspace(1900,1980,9)
years10 = np.linspace(1900,1990,10)
y1 = years
y2 = years - 1900
y3 = years - 1940
y4 = (years - 1940)/40
V1 = np.vander(y1)
V2 = np.vander(y2)
V3 = np.vander(y3)
V4 = np.vander(y4)

pops = np.array([76212168,
                 92228496,
                 106021537,
                 123202624,
                 132164569,
                 151325798,
                 179323175,
                 203302031,
                 226542199])
def qa():
    return [LA.cond(V1),LA.cond(V2),LA.cond(V3),LA.cond(V4)]
    
def qb():
    plt.plot(years,pops,'*k')
    a = np.linalg.solve(V4,pops)
    xs = np.array(range(int(years[0]),int(years[-1])+1))
    plt.plot(xs,np.polyval(a,(xs - 1940)/40),'--r')  
    plt.show()
    
    
def qc():
    dydx = np.zeros(pops.shape)
    for i in range(8):
        if i == 0:
            dydx[i] = (pops[i+1] - pops[i])/(years[i+1]-years[i])
        elif i == 8:
            dydx[i] = (pops[i] - pops[i-1])/(years[i]-years[i-1])
        else:
            dydx[i] = (pops[i+1] - pops[i-1])/(years[i+1]-years[i-1])
    chs = CubicHermiteSpline(years,pops,dydx)
    tts = np.array(range(int(years[0]),int(years[-1])+1))
    yys = chs(tts)
    
    plt.scatter(years,pops)
    plt.plot(tts,yys,"-.r")
    plt.grid(True)
    plt.xlabel("years")
    plt.ylabel("population")
    plt.title("cubic hermite spline")  
    plt.show()
    
def qd():
    cs = CubicSpline(years,pops)
    tts = np.array(range(int(years[0]),int(years[-1])+1))
    cys = cs(tts)
    
    plt.scatter(years,pops)
    plt.plot(tts,cys,"-.r")
    plt.xlabel("years")
    plt.ylabel("population")
    plt.title("cubic spline")
    plt.show()

def qe():
    a = np.linalg.solve(V1,pops)
    y1 = np.polyval(a,1990)
    
    dydx = np.zeros(pops.shape)
    for i in range(8):
        if i == 0:
            dydx[i] = (pops[i+1] - pops[i])/(years[i+1]-years[i])
        elif i == 8:
            dydx[i] = (pops[i] - pops[i-1])/(years[i]-years[i-1])
        else:
            dydx[i] = (pops[i+1] - pops[i-1])/(years[i+1]-years[i-1])
    chs = CubicHermiteSpline(years,pops,dydx)
    y2 = chs(1990)
    
    cs = CubicSpline(years,pops)
    y3 = cs(1990)   
    
    print("  method   |    estimate    |   true value   |")
    print(" polynomial|    {:.0f}   |  {:.0f}  |  ".format(y1,248709873))
    print("  Hermite  |    {:.0f}   |  {:.0f}  |  ".format(y2,248709873))
    print("   cubic   |    {:.0f}   |  {:.0f}  |  ".format(y3,248709873))
    
def qf():
    tl = 0
    tc = 0
    tp = 0
    xs = np.array(range(int(years[0]),int(years[-1])+1))   
    
    for i in range(100):
        p = interp.lagrange(years,pops)
        start_time = time.perf_counter()
        yl = p(xs)
        time_taken = (time.perf_counter() - start_time)*1000
        tl+=time_taken
        
        cs = CubicSpline(years,pops)
        start_time = time.perf_counter()
        y3 = cs(xs)           
        time_taken = (time.perf_counter() - start_time)*1000
        tc+=time_taken        
        
        a = np.linalg.solve(V1,pops)
        start_time = time.perf_counter()
        yp = np.polyval(a,xs)
        time_taken = (time.perf_counter() - start_time)*1000
        tp+=time_taken     
        
    print("|   method   |    time    |")
    print("|  lagrange  |  {:.2f}ms  |".format(tl/100))
    print("|   cubic    |  {:.2f}ms  |".format(tc/100))
    print("|   horner   | {:.2f}ms  |".format(tp/100))
    
def create_m():
    mat = np.zeros(V1.shape)
    for i in range(9):
        mat[i][0] = 1
        if i == 0:
            continue
        for k in range(1,i+1):
            to_add = 1
            c = k
            l = i
            while c != 0:
                to_add *= 10*l
                l-=1
                c-=1
            mat[i][k] = to_add
    return mat

def p8(x, t):
    ret = 0
    for i in range(9):
        to_add = x[i]
        k = i
        while k != 0:
            to_add = to_add * (t-years[k-1])
            k-=1
        ret = ret + to_add
    return ret

def pi(t):
    ret = 1
    for i in range(9):
        ret = ret * (t - years[i])
    return ret

def p9(x,t):
    x9 = (248709873 - p8(x,1990))/pi(1990)
    return p8(x,t) + x9 * pi(t)
    

def qg():
    mat = create_m()
    x8 = np.linalg.solve(mat,pops)
    xs = np.array(range(int(years[0]),int(years[-1])+1))
    xss = np.array(range(int(years10[0]),int(years10[-1])+1))
    y8 = []
    y9 = []
    for y in xss:
        y1 = p8(x8,y)
        y8.append(y1)
        y2 = p9(x8,y)
        y9.append(y2)
    plt.plot(years,pops,'*k')
    plt.plot(xss,y8)
    plt.plot(xss,y9)
    plt.legend(["accurate","degree 8","degree 9"])
    plt.show()
    

def qh():
    xs = np.array(range(int(years[0]),int(years[-1])+1))
    ph = np.round(pops,-6)
    c1 = np.linalg.solve(V4,pops)
    c2 = np.linalg.solve(V4,ph)
    print("coe1:",c1)
    print("coe2:",c2)
    print("relative error:",np.linalg.norm(abs(c1-c2))/np.linalg.norm(c1))
    plt.plot(years,pops,'*k')
    plt.plot(xs,np.polyval(c1,(xs - 1940)/40),'--r')
    plt.plot(xs,np.polyval(c2,(xs - 1940)/40),'--b')  
    plt.legend(["accurate","not rounded","rounded"])
    plt.show()    
       