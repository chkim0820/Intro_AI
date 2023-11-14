from scipy.stats import binom 
import matplotlib.pyplot as plt 


def problem2C():
    # setting the values of n and p 
    p = 0.5
    # defining list of r values 
    r_values = ['1']
    # list of pmf values 
    dist = [binom.pmf(1, 1, p)] 
    # plotting the graph 
    plt.bar(r_values, dist) 
    plt.show()

def problem2B() :
    # setting the values 
    # of n and p 
    n = 4
    p = 0.75
    # defining list of r values 
    r_values = list(range(n + 1)) 
    # list of pmf values 
    dist = [binom.pmf(r, n, p) for r in r_values ] 
    # plotting the graph  
    plt.bar(r_values, dist) 
    plt.show()