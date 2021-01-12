import os
import numpy as np
# ini=[271.0400085449219, 129.489990234375, 1.5]
# goal=[249.41009471,134.07869706,1.085471]
ini=[271.03924560546875, 129.48997497558594,1.5]
goal=[258.67194411,128.33213598,1.0854]
aa=ini[0]
dec=abs(ini[1]-goal[1])/abs(aa-(goal[0]))
for i in range(int(goal[0]),int(aa)):
    
    
    # print(int(goal[0])-i)
    if abs((int(aa)-i))<10:
        
        ini=[ini[0]-1,ini[1]-dec,ini[2]-0.05]
    else:
        ini=[ini[0]-1,ini[1]-dec,ini[2]]
    print(*ini,sep=',')
    

    