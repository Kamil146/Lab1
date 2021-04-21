import numpy as np
import matplotlib.pyplot as plt

def mod(x):
    y=np.copy(x)

    for i in range(np.size(x)):

        if(x[i]<0 and x[i]>=-10): y[i]=np.sin(x[i])
        if(x[i]>=0 and x[i]<=10): y[i]=np.sqrt(x[i])
        if(x[i]<-10 or x[i]>10): break

    plt.plot(x,y,'+',color='green')
    return y




