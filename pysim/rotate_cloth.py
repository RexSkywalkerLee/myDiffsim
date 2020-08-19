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
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter



handles = [10, 51, 41, 57]
ref_points = [25, 60, 30, 54]
center = 62

cloth = [25, 24, 3, 70, 7, 6, 58, 31, 30, 71, 26, 4, 5, 80, 8, 17, 19, 66, 55, 28, 29, 50, 49, 27, 18, 32, 68,
         42, 2, 1, 51, 76, 57, 56, 33, 67, 43, 72, 0, 61, 62, 45, 44, 63, 65, 77, 75, 9, 10, 12, 41, 78, 64, 79,
         73, 74, 34, 20, 11, 39, 40, 47, 46, 59, 23, 22, 15, 14, 37, 36, 48, 52, 60, 38, 21, 16, 13, 69, 35, 53, 54]

losses = []

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'rotate_out/exp7/'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

writer = SummaryWriter(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

if torch.cuda.is_available():
    dev = "cuda:2"
else:
    dev = "cpu"

torch.set_num_threads(8)
device = torch.device(dev)

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 128).double()
        self.fc2 = nn.Linear(128, 256).double()
        self.fc3 = nn.Linear(256, 512).double()
        self.fc4 = nn.Linear(512, n_output).double()
        self.dropout = nn.Dropout(0.05)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        # x = torch.clamp(x, min=-5, max=5)
        return x

class CNNet(nn.Module):
        def __init__(self, n_output):
                super(CNNet,self).__init__()
                self.cv1 = nn.Conv2d(6, 16, kernel_size=2, stride=1, padding=1).double()
                self.cv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1).double()
                self.maxpool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.05)
                self.fc1 = nn.Linear(5*5*32+1, 120).double()
                self.fc2 = nn.Linear(120, 84).double()
                self.fc3 = nn.Linear(84, n_output).double()

        def forward(self,x,t):
                x = torch.reshape(x,(1,6,9,9))
                x = F.relu(self.cv1(x))
                x = F.relu(self.cv2(x))
                x = self.maxpool(x)
                x = torch.flatten(x)
                x = torch.cat([x,t])
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                x = x.to(device)
                return  x

with open('conf/rigidcloth/drag/drag_cloth.json','r') as f:
	config = json.load(f)

goal = []
with open('meshes/rigidcloth/drag/rotated_big_flag.obj','r') as f:
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

def reset_sim(sim, epoch):
	if epoch % 5==0:
		arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
	#text_name = out_path+'/out%d'%epoch + "/goal.txt"
	#np.savetxt(text_name, goal, delimiter=',')
	else:
		arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)
        #print(sim.obstacles[0].curr_state_mesh.dummy_node.x)

def visualize_loss(losses,dir_name):
    plt.plot(losses)
    plt.title('losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(dir_name+'/'+'loss'+'.jpg')


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

    cum_loss = torch.tensor([0.0], dtype=torch.float64)
    cum_loss = cum_loss.to(device)

    for step in range(steps):
        print(step)
        		
        net_input = []
        for i in range(len(ref_points)):
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].v)
        for i in range(len(handles)):
            net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
        net_input.append(torch.tensor(step/steps, dtype=torch.float64).view(1))
        net_output = net(torch.cat(net_input).to(device))
        
       # net_outer = []
       # for i in range(9):
       #     net_inner = []
       #     net_inner.append(torch.cat([ sim.cloths[0].mesh.nodes[i].x[0].view(1,1,1) for i in cloth[9*i:9*i+9] ], dim=1))
       #     net_inner.append(torch.cat([ sim.cloths[0].mesh.nodes[i].x[1].view(1,1,1) for i in cloth[9*i:9*i+9] ], dim=1))
       #     net_inner.append(torch.cat([ sim.cloths[0].mesh.nodes[i].x[2].view(1,1,1) for i in cloth[9*i:9*i+9] ], dim=1))
       #     net_inner.append(torch.cat([ sim.cloths[0].mesh.nodes[i].v[0].view(1,1,1) for i in cloth[9*i:9*i+9] ], dim=1))
       #     net_inner.append(torch.cat([ sim.cloths[0].mesh.nodes[i].v[1].view(1,1,1) for i in cloth[9*i:9*i+9] ], dim=1))
       #     net_inner.append(torch.cat([ sim.cloths[0].mesh.nodes[i].v[2].view(1,1,1) for i in cloth[9*i:9*i+9] ], dim=1))
       #     net_inner = torch.cat(net_inner, dim=2)
       #     net_outer.append(net_inner)
       # net_outer = torch.cat(net_outer, dim=0)
       # net_outer = net_outer.to(device)
       ##print(net_outer.size())
       # 
       # time = torch.tensor([step/steps], dtype=torch.float64).to(device)

       # net_output = net(net_outer, time)

        for i in range(len(handles)):
            sim_input = net_output[3*i:3*i+3]
            sim_input = torch.clamp(sim_input, -1e5, 1e5)
            sim_input = sim_input.cpu()
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input

        arcsim.sim_step()

        ans = [] 
        ans.extend([ sim.cloths[0].mesh.nodes[i].x.to(device) for i in ref_points ])
        ans.extend([ sim.cloths[0].mesh.nodes[i].x.to(device) for i in handles ])
        ans.append(sim.cloths[0].mesh.nodes[center].x.to(device))
    
        ans = torch.stack(ans)
        ans = ans.to(device)

        loss = get_rotation_loss(ans)

       # ans = [ node.x.to(device) for node in sim.cloths[0].mesh.nodes ]
       # ans = torch.stack(ans)
       # ans = ans.to(device)

       # loss = get_reference_loss(ans, goal)

        cum_loss = cum_loss + loss * (step / steps)

    return cum_loss.to(device)

def do_train(optimizer,scheduler,sim,net):
    epoch = 1
    while True:
        #steps = int(1*15*spf)
        steps = 30
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(steps, sim, net)
        en0 = time.time()
        		
        optimizer.zero_grad()
        
        loss.backward()
        writer.add_scalar('training loss', loss, epoch)

        en1 = time.time()
        print("=======================================")
        f.write('epoch {}: loss={}\n'.format(epoch, loss.data))
       #print('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(epoch, loss.data, ans.data, goal.data))
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        losses.append(loss.data)

        print('forward time = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        
        if epoch % 5 == 0:
            torch.save(net.state_dict(), torch_model_path)
        
        #if loss<1e-3:
        #    break
        # dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
       
        for param in net.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        optimizer.step()
        scheduler.step()

        print(optimizer.param_groups[0]['lr'])
        		
        if epoch >= 200:
            break
        
        if epoch % 50 == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0, last_epoch=-1)
        
        epoch = epoch + 1


with open(out_path+'/log.txt','w',buffering=1) as f:
    sim=arcsim.get_sim()
    # reset_sim(sim)
    
   #param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
    net = Net(49, 12)
   #net = CNNet(12)
    net = net.to(device)
    if os.path.exists(torch_model_path):
        net.load_state_dict(torch.load(torch_model_path))
        print("load: %s\n success" % torch_model_path)
    
    lr = 0.01
    momentum = 0.9
    f.write('lr={} momentum={}\n'.format(lr,momentum))
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0, last_epoch=-1)
   #optimizer = torch.optim.Adam(net.parameters(),lr=lr)
   #optimizer = torch.optim.Adadelta([density, stretch, bend])
    do_train(optimizer,scheduler,sim,net)
    losses = np.array(losses,dtype=np.float32)
    np.save(out_path+'/'+'_loss',losses)
    visualize_loss(losses,out_path)
print("done")

