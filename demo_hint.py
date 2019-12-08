import random
import math
import numpy as np
from fullFORCE import fullFORCE 

#This script will train and test a rate model using the full-FORCE
#algorightm. It will train it once using hints, and once without using
#hints on the "ready-set-go" task.

p = {'DTRLS':1, 'dt':1e-3, 'taux': 1e-2, 'eta':0}
#dt: time step, taux: decay time 
#DTRLS: num timestep between updates

T = {'RLS': 1000, 'test': 500, 'init':10, 'data':100}

#gains
g = {'r':1.5, 'fout':1, 'fin':1, 'fhint':1}

N={'N':1000, 'out':1, 'in':2, 'hint':1}

random.seed(0)

ran={'J': 1/math.sqrt(N['N']) * np.random.normal(size=N['N']),\
    'fout': (-1 + 2 * np.random.uniform(size=[N['N'], N['out']])),\
    'fin':(-1 + 2 * np.random.uniform(size=[N['N'], N['in']])), \
    'fhint': (-1 + 2 * np.random.uniform(size=[N['N'], N['hint']]))}

V=np.eye(N['N'])
lrn = fullFORCE('train', g, N, p, ran, T['RLS'], T['init'], 'ready_set_go', 'nohint', V)
ERR = fullFORCE('test', g, N, p, ran, T['test'], T['init'], \
    'ready_set_go', 'nohint', V, lrn)

print(ERR)



