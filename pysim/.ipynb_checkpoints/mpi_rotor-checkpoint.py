import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
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

handles = [10, 51, 41, 57]
ref_points = [25, 60, 30, 54]
center = 62

cloth = [25, 24, 3, 70, 7, 6, 58, 31, 30, 71, 26, 4, 5, 80, 8, 17, 19, 66, 55, 28, 29, 50, 49, 27, 18, 32, 68,
         42, 2, 1, 51, 76, 57, 56, 33, 67, 43, 72, 0, 61, 62, 45, 44, 63, 65, 77, 75, 9, 10, 12, 41, 78, 64, 79,
         73, 74, 34, 20, 11, 39, 40, 47, 46, 59, 23, 22, 15, 14, 37, 36, 48, 52, 60, 38, 21, 16, 13, 69, 35, 53, 54]

losses = []

try:
    cuda = sys.argv[1].split(',')[(rank)%(len(sys.argv[1].split(',')))]
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    print(cuda)
    default_path = sys.argv[2]
except:
    print('Error in usage!')

print("cuda device:",cuda)

steps = 20
epochs = 100

if os.path.exists(default_path+str(rank)):
    pass
else:
    os.mkdir(default_path+str(rank))
out_path = default_path+str(rank)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU is:",torch.cuda.is_available())

writer = SummaryWriter(out_path+'/exp_loss')

if rank == 0:
    setup_pytorch_for_mpi()

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        log_std = -0.5 * np.ones(n_output, dtype=np.float64)
        log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.std = torch.exp(log_std).double()
        self.fc1 = nn.Linear(n_input, 64).double()
        self.fc2 = nn.Linear(64, 64).double()
        self.fc3 = nn.Linear(64, n_output).double()
        self.dropout = nn.Dropout(0.05)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        a = Normal(x, self.std.to(device)).sample()
        return x

with open('conf/rigidcloth/rotor/rotor.json','r') as f:
    config = json.load(f)

goal = []
with open('meshes/rigidcloth/rotor/flag_90deg.obj','r') as f:
    for line in f:
        if 'v ' in line:
            pos = [float(i) for i in line[2:].split()]
            new_pos = torch.tensor(pos, dtype=torch.float64).to(device)
            goal.append(new_pos)
goal = torch.stack(goal).to(device)

def save_config(config, file):
    with open(file,'w') as f:
        json.dump(config, f)

save_config(config, out_path+'/conf.json')

spf = config['frame_steps']

scalev=1

def reset_sim(sim, epoch=0, rank=0):
    if epoch % 5==0:
        arcsim.init_physics(out_path+'/conf.json', out_path+'/rank'+str(rank)+'_out%d'%epoch,False)
    else:
        arcsim.init_physics(out_path+'/conf.json',out_path+'/rank'+str(rank)+'_out',False)

def visualize_loss(losses, dir_name, rank):
    plt.plot(losses)
    plt.plot(losses.argmin(),losses.min(),'ro')
    plt.text(losses.argmin(),losses.min(), "Minimum :{0} at {1}".format(losses.argmin(),losses.min()))
    plt.title('losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(dir_name+'/'+'loss'+str(rank)+'.jpg')

def get_rotation_loss(ans):
    loss = torch.norm(ans[-1,:] - torch.tensor([0.500000, 0.502674, -0.000000], dtype=torch.float64).to(device))
    cnt = 0
    for i in ref_points:
        loss = loss + torch.norm(ans[cnt,:] - goal[i,:])
        cnt += 1
    for i in handles:
        loss = loss + torch.norm(ans[cnt,:] - goal[i,:])
        cnt += 1
   # loss = loss + torch.norm(ans[:,-1]) 
    loss = loss.to(device)
    return loss


def get_reference_loss(ans, goal):
    diff = ans - goal
    loss = torch.norm(diff)
    loss = loss.to(device)
    return loss	


def run_sim(steps, sim, net):

    for step in range(steps):
        print(rank,':',step)
        
        net_input = []
        for i in range(len(ref_points)):
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].v)
        net_input.append(sim.obstacles[0].curr_state_mesh.dummy_node.x)
        net_input.append(sim.obstacles[0].curr_state_mesh.dummy_node.v)
        net_input.append(torch.tensor(step/steps, dtype=torch.float64).view(1))
        net_output = net(torch.cat(net_input).to(device))

        sim.obstacles[0].curr_state_mesh.dummy_node.v += net_output.cpu()

        arcsim.sim_step()

    ans = [ node.x.to(device) for node in sim.cloths[0].mesh.nodes ]
    ans = torch.stack(ans)
    ans = ans.to(device)

    loss = get_reference_loss(ans, goal)

    return loss.to(device)

def server_task(optimizer,scheduler,net,sim): # This is a loop
    epoch = 0
    global steps, epochs
    while True:
        st = time.time()
        optimizer.zero_grad()
        # ==== Synchronize Parameters ====
        sync_params(net)
        # ================================

        # ==== Run the training on server ====
        net = net.to(device)
        #print(next(net.parameters()).grad)
        reset_sim(sim,epoch,rank)
        loss = run_sim(steps,sim,net)
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
        print('epoch {}: loss={}\n'.format(epoch, loss))
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))


        if epoch % 5 == 0:
            torch.save(net.state_dict(), torch_model_path)
        optimizer.step()
        scheduler.step(epoch)
        
        if epoch >= epochs:
            break
        epoch += 1
        losses.append(loss)

def runner_task(sim):# This is loop
	epoch = 0
	net = Net(37,6)
	while True:
		sync_params(net)
		net = net.to(device)
		reset_sim(sim,epoch,rank)
		loss = run_sim(steps,sim,net)
		loss.backward()
		net = net.cpu()

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
    sim = arcsim.get_sim()
    reset_sim(sim,rank=rank)
    runner_task(sim)
else:
    net = Net(37,6).cpu()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,50,2,eta_min=0.0001)
    sim = arcsim.get_sim()
    reset_sim(sim,rank=rank)
    server_task(optimizer,scheduler,net,sim)
    net = net.to(device)
    with torch.no_grad():
        run_sim(steps*2,sim,net)
    torch.save(net.state_dict(),default_path+'/net')
losses = np.array(losses,dtype=np.float32)
np.save(default_path+'/'+str(rank)+'_loss',losses)
visualize_loss(losses,default_path,rank)
print("Done!:",rank)
exit(0)


