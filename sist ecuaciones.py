# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:59:32 2018

@author: OttoF
"""

import numpy as np

## A * x=b
## x= A**-1 * b

A = np.matrix([[0.36,-0.91],
               [-0.37,0.27]]  )

b = np.matrix([[-0.12],
               [0.23] ] ) 


x = A**(-1) * b


print(x)


