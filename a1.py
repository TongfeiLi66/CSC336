#CSC336 Assignment #1 starter code
import numpy as np

#Q2a
def alt_harmonic(fl=np.float16):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series.

    The floating point type fl is used in the calculations.
    """
    sum = 0
    n = 1
    summ = fl(sum) + fl(1)/fl(1)
    while sum != summ:
        n += 1
        sum = summ
        if n%2 == 0:
            summ = fl(summ) - fl(1)/fl(n)
        else:
            summ = fl(summ) + fl(1)/fl(n)
    
    return [n, summ]


#Q2b
#add code here as stated in assignment handout
exp = np.log(2)
ret = alt_harmonic(np.float16)
act = ret[1]
q2b_rel_error = np.divide(np.abs(float(act)-float(exp)), float(exp))

#Q2c
def alt_harmonic_given_m(m, fl=np.float16):
    """
    Returns the sum of the first m terms of the alternating
    harmonic series. The sum is performed in an appropriate
    order so as to reduce rounding error.

    The floating point type fl is used in the calculations.
    """
    sum = 0
    n = m
    while n != 0:
        if n%2 == 0:
            sum = fl(sum) - fl(1)/fl(n)
        else:
            sum = fl(sum) + fl(1)/fl(n)
        n -= 1   
    return sum


#Q3a
def alt_harmonic_v2(fl=np.float32):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series (using
    the formula in Q3a, where terms are paired).

    The floating point type fl is used in the calculations.
    """
    n = 1
    sum = 0
    # 1/(2n-1) - 1/2n
    to_be_add = np.divide(fl(1), np.multiply(fl(2),fl(n)) - fl(1)) - np.divide(1, np.multiply(fl(2),fl(n)))
    summ = fl(sum) + fl(to_be_add)
    while summ != sum:
        n += 1
        to_be_add = np.divide(fl(1), np.multiply(fl(2),fl(n)) - fl(1)) - np.divide(1, np.multiply(fl(2),fl(n)))
        sum = summ
        summ = fl(summ) + fl(to_be_add)
    return [n, summ]

#Q3b
#add code here as stated in assignment handout
ret = alt_harmonic_v2(np.float32)
act = ret[1]
exp = np.log(2)
q3b_rel_error = np.divide(np.abs(float(act)-float(exp)), float(exp))


#Q4b
def hypot(a, b):
    """
    Returns the hypotenuse, given sides a and b.
    """
    ma = max(a,b)
    mi = min(a,b)
    k = np.divide(mi,ma)
    return np.multiply(ma, np.sqrt(1+k**2))
    

#Q4c
q4c_input = [float(1e+300), float(1e+300), float(1e+300), float(1e+300)] #see handout for what value should go here.


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    pass