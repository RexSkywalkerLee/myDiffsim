import os
import sys
import time
import json
import copy

try:
    procs = sys.argv[1]
    cudas = sys.argv[2]
except:
    print("Usage: python3 exp_parallel_runner.py <cudas>")
    exit(-1)
path = 'rotor_out/'+time.ctime()
os.mkdir(path)
#conf = json.load(open('conf.json'))
#task = conf['experiment']['task']
#proc = conf['training']['process']
#subpath = path+'/{0}_{1}_{2}_{3}_{4}_{5}'.format(lr,m,betas,op,sc,proc) 
#os.mkdir(subpath)
command = 'mpiexec -n {0} python3 mpi_rotor_d.py {1} \'{2}/\''.format(procs,cudas,path)
ret = os.system(command)
if ret != 0:
    exit()
#print('Experiment:{0}_{1}_{2}_{3}_{4}_{5}'.format(lr,m,betas,op,sc,proc))
