import math
import random
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.asarray([5,15,25,35])
    y = np.asarray([34,30,25,24])

    plt.axis([0, 40, 0, 45])

    plt.plot(x,y,color="blue",linewidth=2,linestyle=':',label='Jay income', marker='o')

    plt.xlabel('m')
    plt.ylabel('mg/m^3')
    plt.show()



