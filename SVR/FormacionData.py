# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:02:16 2021

@author: Gustavo
"""

import numpy as np
import os
import csv

PATH = "." 

RoR_max=4.0
RoR_min=1.0
m = (70.0-100.0)/(RoR_max-RoR_min)
n_u = 107.575
n = 100.0 - m*RoR_min
n2 = 72.6 - m*3.5
n3 = 95 - m*1.75
n4 = 91.25 - m*1.63
print(n)
print(n2)
print(n3)
print(n4)

RoR = [(96-n_u)/m, (97-n_u)/m, (98-n_u)/m,(99-n_u)/m]
SpaO2_real = [96,97,98,99]

f = open(PATH+'/spaO2_dataset.csv', 'w+', encoding="utf-8")
f.close()
for i in range(100):
    if ( os.path.exists(PATH+'/spaO2_dataset.csv')): #si existe 
        print("se escribio en consola")
        f = open(PATH+'/spaO2_dataset.csv', 'a+', encoding="utf-8")
        writer = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        noise = np.random.normal(0,0.05/3,4)
        for j in range(0,4):
            escribir = [RoR[j]+noise[j], SpaO2_real[j]]
            escribir_str = [str(escribir[0]), str(escribir[1])]
            print(escribir)
            print( (RoR[j]+noise[j])*m+n_u )
            writer.writerow(escribir_str)
        f.close()