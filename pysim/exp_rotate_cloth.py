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
from mpi4py import MPI
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('exp_rotate_cloth_runs/exp2')
running_loss = 0.0

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank() # Define rank 0 as server and rest are  runner
# size = comm.Get_size()

handles = [10, 51, 41, 57]
ref_points = [25, 60, 30, 54]
center = 62
node_number = 0
using_whole = True
net_name = 'CNN'

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU is:",torch.cuda.is_available())

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

losses = []


class Net(nn.Module):
	def __init__(self, n_input, n_output):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_input, 100).double()
		self.fc2 = nn.Linear(100, 200).double()
		self.fc3 = nn.Linear(200, n_output).double()
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = x.to(device)
		# x = torch.clamp(x, min=-5, max=5)
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
		self.fc1.weight.data.fill_(0.001)
		self.fc2 = nn.Linear(128, 256).double()
		self.fc2.weight.data.fill_(0.001)
		self.fc3 = nn.Linear(256, 512).double()
		self.fc3.weight.data.fill_(0.001)
		self.fc4 = nn.Linear(512, n_output).double()
		self.fc4.weight.data.fill_(0.001)
		self.dropout = nn.Dropout(0.05)

	def forward(self, x,t):
		x = torch.cat([x,t])
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.dropout(x)
		x = self.fc4(x)
		return x

# class RunPack:
# 	def __init__(self,goal,net_params):
# 		self.goal = goal
# 		self.net_params = net_params

# class ServerPack:
# 	def __init__(self,loss,gradient):
# 		self.loss = loss
# 		self.gradient = gradient
		
with open('conf/rigidcloth/drag/drag_cloth.json','r') as f:
	config = json.load(f)
    
# matfile = config['cloths'][0]['materials'][0]['data']
# with open(matfile,'r') as f:
# 	matconfig = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')


torch.set_num_threads(20)
spf = config['frame_steps']
	
scalev=1

def reset_sim(sim, epoch):
	if epoch % 5==0:
		arcsim.init_physics(out_path+'/conf.json', out_path+'/r_out%d'%epoch,False)
	#text_name = out_path+'/out%d'%epoch + "/goal.txt"
	#np.savetxt(text_name, goal[3:6], delimiter=',')
	else:
		arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)
	#print(sim.obstacles[0].curr_state_mesh.dummy_node.x)



# def get_loss(ans, goal):
# 	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
# 	dif = ans - goal
# 	loss = torch.norm(dif.narrow(0, 3, 3), p=2)

# 	return loss

def get_loss(sim):
	ans = []
	for node in ref_points:
		ans.append(sim.cloths[0].mesh.nodes[node].x[2])
	print(ans)
	ans = torch.stack(ans).to(device)
	loss = torch.norm(ans, p=2)
	
	return loss.to(device)
	

def get_rotation_loss(sim):
	loss = 0
	# Need to include: 
	# 	loss for angular velocity on x-y plane
	for i in ref_points:
		v = torch.norm(torch.tensor([sim.cloths[0].mesh.nodes[i].v[0],
									  sim.cloths[0].mesh.nodes[i].v[1],0]))
		r = torch.norm(torch.tensor([sim.cloths[0].mesh.nodes[i].x[0]-sim.cloths[0].mesh.nodes[center].x[0],
						             sim.cloths[0].mesh.nodes[i].x[1]-sim.cloths[0].mesh.nodes[center].x[1],0]))
		loss -= v/(r+1e-4)
	# 	loss for change in center's position
	loss += torch.norm(sim.cloths[0].mesh.nodes[center].x
					   - torch.tensor([0.5,0.5,0]))
	#	loss for change in handles z coord
	for p in handles:
		loss += sim.cloths[0].mesh.nodes[p].x[2]**2
	loss = loss.to(device)
	return loss




def run_sim(steps, sim, net):

	#for obstacle in sim.obstacles:
	#	for node in obstacle.curr_state_mesh.nodes:
	#		node.m    *= 0.2

	#sim.obstacles[0].curr_state_mesh.dummy_node.x = torch.tensor([0.0000, 0.0000, 0.0000,
	#np.random.random(), np.random.random(), -np.random.random()],dtype=torch.float64)
	#print("obstcale_pose:",sim.obstacles[0].curr_state_mesh.dummy_node.x)
	for step in range(steps):
		print(step)
# 		remain_time = torch.tensor([step/steps],dtype=torch.float64)
            
# 		if using_whole:
# 			x1,x2,x3,v1,v2,v3 = [],[],[],[],[],[]
# 			for i in range(node_number):
# 				x1.append(sim.cloths[0].mesh.nodes[i].x[0])
# 				x2.append(sim.cloths[0].mesh.nodes[i].x[1])
# 				x3.append(sim.cloths[0].mesh.nodes[i].x[2])
# 				v1.append(sim.cloths[0].mesh.nodes[i].v[0])
# 				v2.append(sim.cloths[0].mesh.nodes[i].v[1])
# 				v3.append(sim.cloths[0].mesh.nodes[i].v[2])
# 			net_input = torch.cat([torch.tensor(x1),
#                                    torch.tensor(x2),
#                                    torch.tensor(x3),
#                                    torch.tensor(v1),
#                                    torch.tensor(v2),
#                                    torch.tensor(v3)]).to(device)
# 			t = torch.tensor(remain_time).to(device)
# 			net_output = net(net_input,t)
# 		else:
# 			net_input = []
# 			for i in range(len(handles)):
# 				net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
# 				net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
# 				# dis = sim.obstacles[0].curr_state_mesh.dummy_node.x - goal
# 				# net_input.append(dis.narrow(0, 3, 3))
# 			net_input.append(remain_time)
# 			net_output = net(torch.cat(net_input).to(device))
		

		# outputs = net_output.view([4, 3])
		
		net_input = []
		for i in range(len(ref_points)):
			net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].x)
			net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].v)
		for i in range(len(handles)):
			net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
			net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
		net_output = net(torch.cat(net_input).to(device))
		
		for i in range(len(handles)):
			sim_input = torch.cat([net_output[i*2].view([1]).cpu(),
                                   net_output[i*2+1].view([1]).cpu(),
                                   torch.tensor([0], dtype=torch.float64)])
			sim.cloths[0].mesh.nodes[handles[i]].v += sim_input
			sim.cloths[0].mesh.nodes[handles[i]].v[2] = torch.tensor([0], dtype=torch.float64)
			
			sim.cloths[0].mesh.nodes[center].v = torch.tensor([0,0,0], dtype=torch.float64)

		arcsim.sim_step()
	

	loss = get_loss(sim)

	return loss

def do_train(cur_step,optimizer,sim,net):
	epoch = 0
	loss_prev = 0
	while True:
		# steps = int(1*15*spf)
		steps = 30

		reset_sim(sim, epoch)

		st = time.time()
		loss = run_sim(steps, sim, net)
		en0 = time.time()
		
		optimizer.zero_grad()

		loss.backward()
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

# 		if loss < -0.3 and abs(loss_prev - loss) <= 0.02:
# 			break
# 		loss_prev = loss
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
# 		for param in net.parameters():
# 			param.grad.data.clamp_(-0.05,0.05)
		optimizer.step()
		if epoch>=200:
			break
		epoch = epoch + 1
		losses.append(loss.data)
		# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	
	tot_step = 1
	sim=arcsim.get_sim()
	arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)

	node_number = len(sim.cloths[0].mesh.nodes)
	#param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
# 	if using_whole:
# 		if net_name == 'CNN':
# 			net = CNNNet(9, 9, 12).to(device)
# 		if net_name == 'FCN':
# 			net = FCNet(9, 9, 12).to(device)
# 	else:
# 		net = Net(25,12).to(device)
	net = Net(48, 8).to(device)
	if os.path.exists(torch_model_path):
		net.load_state_dict(torch.load(torch_model_path))
		print("load: %s\n success" % torch_model_path)

	lr = 0.01
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,net)
	arcsim.init_physics(out_path+'/conf.json', out_path+'/test',False)
	run_sim(99,sim,net)
	losses = np.array(losses,dtype=np.float32)
	np.save(net_name+'_test',losses)

print("done")
