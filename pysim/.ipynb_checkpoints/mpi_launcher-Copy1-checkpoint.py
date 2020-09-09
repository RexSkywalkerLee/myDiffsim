import os
import sys
import time
import json
import copy

try:
	cudas = sys.argv[1]
except:
	print("Usage: python3 exp_parallel_runner.py <cudas>")
	exit(-1)
path = 'results/'+time.ctime()
os.mkdir(path)
conf = json.load(open('experiments/conf.json'))
task = conf['experiment']['task']
for lr in conf['training']['lr']:
	for m in conf['training']['momentum']:
		for betas in conf['training']['betas']:
			for op in conf['training']['optimizer']:
				for sc in conf['training']['scheduler']:
					for proc in conf['training']['process']:
						subpath = path+'/{0}_{1}_{2}_{3}_{4}_{5}'.format(lr,m,betas,op,sc,proc) 
						os.mkdir(subpath)
						cf = copy.deepcopy(conf)
						cf['training']['lr']        = lr
						cf['training']['momentum']  = m
						cf['training']['betas']     = betas
						cf['training']['optimizer'] = op
						cf['training']['scheduler'] = sc
						del cf['training']['process']
						n                           = proc
						json.dump(cf,open(subpath+'/conf.json','w'))
						command = 'mpiexec -n {0} python3 mpi_cloth_rotate.py {1} \'{2}\''.format(n,cudas,subpath)
						ret = os.system(command)
						if ret != 0:
							exit()
						print('Experiment:{0}_{1}_{2}_{3}_{4}_{5}'.format(lr,m,betas,op,sc,proc))