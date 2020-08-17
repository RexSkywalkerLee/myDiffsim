import torch
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Define rank 0 as server and rest are  runner
size = comm.Get_size()
writer = SummaryWriter('exp_rotate_cloth_runs/exp_'+str(rank))
handles = [10, 57]
ref_points = [25, 60, 30, 54]
center = 62
node_number = 0
using_whole = True
net_name = 'FCN'
steps = 5
epochs= 5
print(sys.argv)
try:
	cuda = sys.argv[2].split(',')[(rank-1)%(len(sys.argv[2].split(',')))]
	os.environ['CUDA_VISIBLE_DEVICES'] = cuda
	print(cuda)
	task = sys.argv[1]
	default_path = sys.argv[3]
except:
	print("Error in Usage!")
print("cuda device:",cuda)
os.mkdir(default_path+'/default_out'+str(rank))
out_path = default_path+'/default_out'+str(rank)

jfile = 'drag'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU is:",torch.cuda.is_available())

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

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
		self.fc1 = nn.Linear(linear_input_size+1,256).double()
		self.fc1.weight.data.fill_(0.001)
		self.fc2 = nn.Linear(256,512).double()
		self.fc2.weight.data.fill_(0.001)
		self.fc3 = nn.Linear(512,n_output).double()
		self.fc3.weight.data.fill_(0.001)


	def forward(self,x,t):
		x = torch.reshape(x,(1,6,9,9))
		x = F.relu(self.cv1(x))
		x = F.relu(self.cv2(x))
		x = F.relu(self.cv3(x))
		x = self.dropout1(x)
		x = torch.flatten(x)
		x = torch.cat([x,t])
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.dropout2(x)
		x = self.fc3(x)
		return  x

class FCNet(nn.Module):
	def __init__(self, nwidth, nheight, n_output):
		super(FCNet, self).__init__()
		self.fc1 = nn.Linear(nwidth*nheight*6+1, 128).double()
		#self.fc1.weight.data.fill_(0.001)
		self.fc2 = nn.Linear(128, 256).double()
		#self.fc2.weight.data.fill_(0.001)
		self.fc3 = nn.Linear(256, 512).double()
		#self.fc3.weight.data.fill_(0.001)
		self.fc4 = nn.Linear(512, n_output).double()
		#self.fc4.weight.data.fill_(0.001)

	def forward(self, x,t):
		x = torch.cat([x,t])
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

net_dict = {'FCN':FCNet,'CNN':CNNNet}

class RunPack:
	def __init__(self,goal,net_params):
		self.goal = goal
		self.net_params = net_params

class ServerPack:
	def __init__(self,loss,gradient):
		self.loss = loss
		self.gradient = gradient

def save_config(jfile):
	with open('conf/rigidcloth/paralldrag/'+jfile,'r') as f:
		config = json.load(f)
	with open(out_path+'/conf.json','w') as f:
		json.dump(config, f)

if rank != 0:
	file_list = os.listdir('conf/rigidcloth/paralldrag')
	jfile = file_list[rank-1]
	save_config(jfile)
else:
	jfile = 'init1.json'
	save_config(jfile)

def reset_sim(sim, epoch=0,rank=0):
	arcsim.init_physics(out_path+'/conf.json', out_path+'/rank'+str(rank)+'_out%d'%epoch,False)

def get_intm_loss(sim,steps):
	loss = 0
	for i in ref_points:
		v = sim.cloths[0].mesh.nodes[i].v
		r = sim.cloths[0].mesh.nodes[i].x - torch.tensor([0.5,0.5,0])
		loss -= torch.cross(v,r)[2]
	loss /= steps
	return loss

def get_rotation_loss(sim,intm_losses,goal):
	loss = torch.tensor(0,dtype=torch.float32)
	loss += torch.norm(sim.cloths[0].mesh.nodes[center].x
					- torch.tensor([0.5,0.5,0]))
	for i in ref_points:
		v = sim.cloths[0].mesh.nodes[i].v
		r = sim.cloths[0].mesh.nodes[i].x - sim.cloths[0].mesh.nodes[center].x
		w = torch.norm(torch.tensor([v[0],v[1]]))*(torch.cross(v,r)[2]/torch.abs(torch.cross(v,r)[2]))
		loss += (w-goal)**2
	loss += torch.sum(intm_losses)
	print(loss)
	loss = loss.to(device)
	return loss

def get_stableness_loss(sim):
	loss = 0
	loss += torch.norm(sim.cloths[0].mesh.nodes[center].x
					- torch.tensor([0.5,0.5,0]))**2
	for p in handles:
		loss += sim.cloths[0].mesh.nodes[p].x[2]**2
	loss = loss.to(device)
	return loss

def get_loss(sim,intm_losses,goal):
	loss = 0
	loss += torch.norm(sim.cloths[0].mesh.nodes[center].x
					- torch.tensor([0.5,0.5,0]))
	for i in ref_points:
		v = sim.cloths[0].mesh.nodes[i].v
		r = sim.cloths[0].mesh.nodes[i].x - sim.cloths[0].mesh.nodes[center].x
		w = torch.norm(torch.tensor([v[0],v[1]]))*(torch.cross(v,r)[2]/torch.abs(torch.cross(v,r)[2]))
		loss += (w-goal)**2
	for p in handles:
		loss += sim.cloths[0].mesh.nodes[p].x[2]**2
	loss += torch.sum(intm_losses)
	loss = loss.to(device)
	return loss

def get_invariant_loss(sim,intm_losses):
	loss = 0
	loss += torch.norm(sim.cloths[0].mesh.nodes[center].x
					- torch.tensor([0.5,0.5,0]))
	for i in ref_points:
		v = sim.cloths[0].mesh.nodes[i].v
		r = sim.cloths[0].mesh.nodes[i].x - sim.cloths[0].mesh.nodes[center].x
		w = torch.norm(torch.tensor([v[0],v[1]]))*(torch.cross(v,r)[2]/torch.abs(torch.cross(v,r)[2]))
		loss -= w
	for p in handles:
		loss += sim.cloths[0].mesh.nodes[p].x[2]**2
	loss += torch.sum(intm_losses)
	loss = loss.to(device)
	return loss


def run_sim(steps, sim, net,goal):
	intm_loss = torch.zeros(steps,dtype=torch.float32)
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
			t = torch.tensor(remain_time).to(device)
			net_output = net(net_input,t)
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

	if task == 'pure_rotation':
		loss = get_rotation_loss(sim,intm_loss,goal)
	elif task == 'pure_stable':
		loss = get_stableness_loss(sim)
	elif task == 'both':
		loss = get_loss(sim,intm_loss,goal)
	elif task == 'invariant':
		loss = get_invariant_loss(sim,intm_loss)
	return loss

def visualize_loss(losses,dir_name,rank):
	plt.plot(losses)
	plt.title('losses')
	plt.xlabel('epochs')
	plt.ylabel('losses')
	plt.savefig(dir_name+'/'+'loss'+str(rank)+'.jpg')

def server_task(optimizer,scheduler,net): # This is a loop
	epoch = 0
	loss_prev = 0
	global steps,epochs
	while True:
		if steps <= 80 and epoch >= 20 and losses[epoch-1] < -0.3:
			steps += 1

		sigma = 0.05
		runner_packs = [] 
		for i in range(size):
			goal_w = np.random.normal(0,scale=sigma) + 2.0 
			net_params = net.state_dict()
			runner_packs.append(RunPack(goal_w,net_params))

		st = time.time()
		_ = comm.scatter(runner_packs,root=0)
		results = comm.gather(None,root=0)[1:]
		# Unpack the gradients and losses
		gradients = [result.gradient for result in results]
		loss = torch.mean(torch.tensor([result.loss for result in results],dtype=torch.float32))
		en0 = time.time()
		optimizer.zero_grad()
		writer.add_scalar('training loss',loss,epoch)

		en1 = time.time()
		print("=======================================")
		f.write('epoch {}: loss={}\n'.format(epoch, loss.data))
		print('epoch {}: loss={}\n'.format(epoch, loss.data))
		#print('epoch {}: loss={}  ans={}\n'.format(epoch, loss.data, ans.data))

		print('forward tim = {}'.format(en0-st))
		print('backward time = {}'.format(en1-en0))


		if epoch % 5 == 0:
			torch.save(net.state_dict(), torch_model_path)

		if loss < -5 and abs(loss_prev - loss) <= 0.02:
			break
		loss_prev = loss
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
		for i,param in enumerate(net.parameters()):
			param.grad = torch.sum(torch.stack([gradients[j][i] for j in range(size-1)],dim=0),dim=0)/(size-1)
			param.grad.data.clamp_(-1,1)
		optimizer.step()
		scheduler.step(loss)
		if epoch>=epochs:
			break
		epoch = epoch + 1
		losses.append(loss.data)
		# break

def runner_task(sim):# This is loop
	epoch = 0
	while True:
		command  = comm.scatter(None,root = 0)
		goal = command.goal
		net = net_dict[net_name](9,9,12).to(device)
		net.load_state_dict(command.net_params)
		reset_sim(sim,epoch,rank)
		loss = run_sim(steps,sim,net,goal)
		loss.backward()
		grads = [param.grad.data for param in net.parameters()]
		pack  = ServerPack(gradient=grads,loss = loss)
		_ = comm.gather(pack,root = 0)
		for param in net.parameters():
			param.grad.data.zero_()
		epoch += 1
		losses.append(loss)



with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	if rank != 0:
		sim=arcsim.get_sim()
		reset_sim(sim,rank=rank)
		node_number = len(sim.cloths[0].mesh.nodes)
		#lr = 0.01
		#momentum = 0.9
		#f.write('lr={} momentum={}\n'.format(lr,momentum))
			#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)
		#optimizer = torch.optim.Adam(net.parameters(),lr=lr)
		#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
			# optimizer = torch.optim.Adadelta([density, stretch, bend])
		runner_task(sim)
		#arcsim.init_physics(out_path+'/conf.json', out_path+'/test',False)
		#run_sim(40,sim,net)
	else:
		if using_whole:
			if net_name == 'CNN':
				net = CNNNet(9, 9, 12).to(device)
			if net_name == 'FCN':
				net = FCNet(9, 9, 12).to(device)
		else:
			net = Net(25,12).to(device)
		lr = 0.01
		momentum = 0.9
		f.write('lr={} momentum={}\n'.format(lr,momentum))
		optimizer = torch.optim.Adam(net.parameters(),lr=lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
		server_task(optimizer,scheduler,net)
		sim = arcsim.get_sim()
		reset_sim(sim,rank=rank)
		goal = np.random.normal(0,scale=0.05) + 2.0
		run_sim(10,sim,net,goal)
		torch.save(net.state_dict(),default_path+'/'+task)
	losses = np.array(losses,dtype=np.float32)
	np.save(default_path+'/'+net_name+str(rank)+'_loss',losses)
	visualize_loss(losses,default_path,rank)
print("done")
