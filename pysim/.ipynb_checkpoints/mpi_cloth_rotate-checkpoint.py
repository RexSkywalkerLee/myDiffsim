import torch
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import sys
import gc
import os

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from mpi.mpi_pytorch import setup_pytorch_for_mpi, sync_params,mpi_avg_grads
from mpi.mpi_tools import mpi_fork, mpi_avg ,num_procs,proc_id

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

rank = proc_id() # Defnine rank 0 as server and rest are  runner
size = num_procs()

center = 62
node_number = 0
using_whole = True
print(sys.argv)
try:
	cuda = sys.argv[1].split(',')[(rank)%(len(sys.argv[1].split(',')))]
	os.environ['CUDA_VISIBLE_DEVICES'] = cuda
	print(cuda)
	default_path = sys.argv[2]
except:
	print("Error in Usage!")
print("cuda device:",cuda)
os.mkdir(default_path+'/default_out'+str(rank))
out_path = default_path+'/default_out'+str(rank)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU is:",torch.cuda.is_available())

# === Visualize Using Tensorboard ===
writer = SummaryWriter(out_path+'/exp_loss')
# ===================================

if rank == 0:
	setup_pytorch_for_mpi()

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

exp = json.load(open(default_path+"/conf.json"))
activation_dict = {'relu':F.relu,'linear':F.linear}

# === Extract global parameters from json ===
net_name = exp['network']['type']
net_shape= exp['network']['shape']
net_activations = [activation_dict[i] for i in exp['network']['activation']]
lr = exp['training']['lr']
momentum = exp['training']['momentum']
betas = exp['training']['betas']
net_optimizer = exp['training']['optimizer']
net_scheduler = exp['training']['scheduler']
handles   = exp['experiment']['handles']
ref_points= exp['experiment']['refs']
epochs    = exp['experiment']['epochs']
steps     = exp['experiment']['steps']
# ============================================
losses = []

class Net(nn.Module):
	def __init__(self, n_input, n_output):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_input, 50).double()
		self.fc2 = nn.Linear(50, 200).double()
		self.fc3 = nn.Linear(200, n_output).double()
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class CNNNet(nn.Module):
	def __init__(self,nwidth,nheight,n_output):
		super(CNNNet,self).__init__()
		self.cv1 = nn.Conv2d(6 ,16,kernel_size=3,stride = 1).double()
		self.cv1.weight.data.fill_(0.001)
		self.cv2 = nn.Conv2d(16,32,kernel_size=3,stride = 1).double()
		self.cv2.weight.data.fill_(0.001)
		self.cv3 = nn.Conv2d(32,64,kernel_size=3,stride = 1).double()
		self.cv3.weight.data.fill_(0.001)
		self.dropout1 = nn.Dropout2d(0.05)
		self.dropout2 = nn.Dropout(0.05)
		def conv2d_size_out(size,kernel_size = 3,stride = 1):
			return (size - (kernel_size -1) - 1)//stride +1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(nwidth)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(nheight)))
		linear_input_size = convw * convh * 64
		self.fc1 = nn.Linear(linear_input_size+1+12,256).double()
		self.fc1.weight.data.fill_(0.001)
		self.fc2 = nn.Linear(256,512).double()
		self.fc2.weight.data.fill_(0.001)
		self.fc3 = nn.Linear(512,n_output).double()
		self.fc3.weight.data.fill_(0.001)


	def forward(self,x,t,last_output):
		x = torch.reshape(x,(1,6,9,9))
		x = F.relu(self.cv1(x))
		x = F.relu(self.cv2(x))
		x = F.relu(self.cv3(x))
		x = self.dropout1(x)
		x = torch.flatten(x)
		x = torch.cat([x,t,last_output])
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.dropout2(x)
		x = self.fc3(x)
		return  x

class FCNet(nn.Module):
	def __init__(self, nwidth, nheight, n_output):
		super(FCNet, self).__init__()
		self.fc1 = nn.Linear(nwidth*nheight*6+1, net_shape[0]).double()
		self.fc2 = nn.Linear(net_shape[0], net_shape[1]).double()
		self.fc3 = nn.Linear(net_shape[1], net_shape[2]).double()
		self.fc4 = nn.Linear(net_shape[2], n_output).double()

	def forward(self, x,t):
		x = x.double()
		t = t.double()
		#last_output = last_output.double()
		x = torch.cat([x,t])
		x = net_activations[0](self.fc1(x))
		x = net_activations[1](self.fc2(x))
		x = net_activations[2](self.fc3(x))
		x = self.fc4(x)
		return x

net_dict = {'FCN':FCNet,'CNN':CNNNet}

def save_config(jfile):
	with open('conf/rigidcloth/paralldrag/'+jfile,'r') as f:
		config = json.load(f)
	with open(out_path+'/conf.json','w') as f:
		json.dump(config, f)


file_list = os.listdir('conf/rigidcloth/paralldrag')
jfile = 'init{0}.json'.format(rank+1)
save_config(jfile)

def reset_sim(sim, epoch=0,rank=0):
	if epoch%5 ==0:
		arcsim.init_physics(out_path+'/conf.json', out_path+'/rank'+str(rank)+'_out%d'%epoch,False)
	else:
		arcsim.init_physics(out_path+'/conf.json', out_path+'/rank'+str(rank)+'_out',False)

def get_intm_loss(sim,steps):
	loss = 0
	for i in [25, 60, 30, 54]:
		v = sim.cloths[0].mesh.nodes[i].v
		r = sim.cloths[0].mesh.nodes[i].x - torch.tensor([0.5,0.5,0])
		loss -= torch.cross(v,r)[2]
	loss /= steps
	return loss

def get_loss(sim,intm_losses):
	loss = 0
	loss += torch.norm(sim.cloths[0].mesh.nodes[center].x
					- torch.tensor([0.5,0.5,0]))**2
	for i in [25, 60, 30, 54]:
		d = (sim.cloths[0].mesh.nodes[i].x[2])**2
		loss += d
	for p in handles:
		loss += sim.cloths[0].mesh.nodes[p].x[2]**2
	loss += torch.sum(intm_losses)
	loss = loss.to(device)
	return loss

def run_sim(steps, sim, net,goal):
	intm_loss = torch.zeros(steps,dtype=torch.float32)
	last_output = torch.zeros(12,dtype = torch.float64).to(device)
	for step in range(steps):
		print(rank,':',step)
		remain_time = torch.tensor([step/steps],dtype=torch.float64)

		if using_whole:
			x1,x2,x3,v1,v2,v3 = [],[],[],[],[],[]
			for i in range(node_number):
				x1.append(sim.cloths[0].mesh.nodes[i].x[0])
				x2.append(sim.cloths[0].mesh.nodes[i].x[1])
				x3.append(sim.cloths[0].mesh.nodes[i].x[2])
				v1.append(sim.cloths[0].mesh.nodes[i].v[0])
				v2.append(sim.cloths[0].mesh.nodes[i].v[1])
				v3.append(sim.cloths[0].mesh.nodes[i].v[2])
			net_input = torch.cat([torch.tensor(x1),
								torch.tensor(x2),
								torch.tensor(x3),
								torch.tensor(v1),
								torch.tensor(v2),
								torch.tensor(v3)]).to(device)
			t = remain_time.to(device)
			net_output = net(net_input,t)
			last_output.data = net_output.data
			last_output = last_output.to(device)

		else:
			net_input = []
			for i in range(len(handles)):
				net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
				net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
			net_input.append(remain_time)
			net_output = net(torch.cat(net_input).to(device))
		net_output = torch.clamp(net_output,-50,50)

		for i in range(len(handles)):
			sim_input = torch.cat([net_output[i*3].view([1]),
								net_output[i*3+1].view([1]),
								net_output[i*3+2].view([1])]).cpu()
			sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 

		arcsim.sim_step()
		intm_loss[step] = get_intm_loss(sim,steps)

	
	loss = get_loss(sim,intm_loss)
	return loss

def visualize_loss(losses,dir_name,rank):
	plt.plot(losses)
	plt.plot(losses.argmin(),losses.min(),'ro')
	plt.text(losses.argmin(),losses.min(),"Minimum :{0} at {1}".format(losses.argmin(),losses.min()))
	plt.title('losses')
	plt.xlabel('epochs')
	plt.ylabel('losses')
	plt.savefig(dir_name+'/'+'loss'+str(rank)+'.jpg')

def server_task(optimizer,scheduler,net,sim): # This is a loop
	epoch = 0
	loss_prev = 0
	global steps,epochs
	while True:
		#if steps <= 80 and epoch >= 20 and losses[epoch-1] < -0.3:
		#   steps += 1

		st = time.time()
		optimizer.zero_grad()
		# ==== Synchronize Parameters ====
		sync_params(net)
		# ================================

		# ==== Run the training on server ====
		goal = 0.0
		net = net.to(device)
		#print(next(net.parameters()).grad)
		reset_sim(sim,epoch,rank)
		loss = run_sim(steps,sim,net,goal)
		loss.backward()
		net = net.cpu()
		# ====================================

		# ==== Get Gradient From Runner ====
		mpi_avg_grads(net)
		# ==================================
		# ==== Average the loss ====
		loss = float(mpi_avg(float(loss)))
		# ==========================

		# Unpack the gradients and losses
		'''
		gradients = [result.gradient for result in results]
		loss = torch.mean(torch.tensor([result.loss for result in results],dtype=torch.float32))
		'''
		#loss = results[1].loss
		en0 = time.time()
		writer.add_scalar('training loss',loss,epoch)

		en1 = time.time()
		print("=======================================")
		#f.write('epoch {}: loss={}\n'.format(epoch, loss.data))
		print('epoch {}: loss={}\n'.format(epoch, loss))
		#print('epoch {}: loss={}  ans={}\n'.format(epoch, loss.data, ans.data))

		print('forward tim = {}'.format(en0-st))
		print('backward time = {}'.format(en1-en0))


		if epoch % 5 == 0:
			torch.save(net.state_dict(), torch_model_path)
		if loss < -1.5 and abs(loss_prev - loss) <= 0.02:
			break
		if loss < -3:
			break
		loss_prev = loss
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		p = 0
		for param in net.parameters():
			param.grad.data.clamp_(-1,1)
		optimizer.step()
		if net_scheduler == 'cos':
			scheduler.step(epoch)
		elif net_scheduler == 'reduceonplateau':
			scheduler.step(loss)
		if epoch>=epochs:
			break
		epoch += 1
		losses.append(loss)
		# break

def runner_task(sim):# This is loop
	epoch = 0
	net = net_dict[net_name](9,9,3*len(handles))
	while True:
		sync_params(net)
		net = net.to(device)
		goal = 0.0
		#print(next(net.parameters()).grad)
		reset_sim(sim,epoch,rank)
		loss = run_sim(steps,sim,net,goal)
		loss.backward()
		net = net.cpu()
		#grads = [param.grad.data for param in net.parameters()]
		#print('Worker{0}: '.format(rank),grads[1])

		# ==== Share the gradient ====
		#pack  = ServerPack(gradient=grads,loss = loss)
		#_ = comm.gather(pack,root = 0)
		mpi_avg_grads(net)
		# ============================

		# ==== Average the loss ====
		loss = mpi_avg(float(loss))
		# ==========================

		for param in net.parameters():
			param.grad.data.zero_()
		losses.append(loss)
		if epoch >= epochs:
			break
		epoch += 1

if rank != 0:
	sim=arcsim.get_sim()
	reset_sim(sim,rank=rank)
	node_number = len(sim.cloths[0].mesh.nodes)
	runner_task(sim)
else:
	if using_whole:
		net = net_dict[net_name](9, 9, 3*len(handles)).cpu()
	else:
		net = Net(25,12).cpu()
	if net_optimizer == 'SGD':
		optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum)
	elif net_optimizer == 'Adam':
		optimizer = torch.optim.Adam(net.parameters(),lr=lr,betas = (betas[0],betas[1]))
	if net_scheduler == 'cos':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2,eta_min=0.0001)
	elif net_scheduler == 'reduceonplateau':
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
	sim = arcsim.get_sim()
	reset_sim(sim,rank=rank)
	node_number = len(sim.cloths[0].mesh.nodes)
	server_task(optimizer,scheduler,net,sim)
	goal = 0.0
	net = net.to(device)
	with torch.no_grad():
		run_sim(steps*2,sim,net,goal)
	torch.save(net.state_dict(),default_path+'/net')
losses = np.array(losses,dtype=np.float32)
np.save(default_path+'/'+net_name+str(rank)+'_loss',losses)
visualize_loss(losses,default_path,rank)
print("Done!:",rank)
exit(0)
