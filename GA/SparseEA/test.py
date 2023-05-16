import numpy as np
import pandas as pd
a = np.asarray([[1,2,3,4],[1,2,3,4]])
b = np.asarray([0,0,0,0])
c = a[:,b==1]
d = c.shape[1]
print(c)