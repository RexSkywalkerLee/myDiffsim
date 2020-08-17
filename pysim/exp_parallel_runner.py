import os
import sys
import time

try:
	cudas = sys.argv[1]
	task  = sys.argv[2]
except:
	print("Usage: python3 exp_parallel_runner.py <cudas> <task>")
	exit(-1)
path = 'results/'+task+"_"+time.ctime()
os.mkdir(path)
n = len(os.listdir('conf/rigidcloth/paralldrag')) + 1
command = 'mpiexec -n {0} python3 exp_on_parallel.py {1} {2} \'{3}\''.format(n,task,cudas,path)
os.system(command)