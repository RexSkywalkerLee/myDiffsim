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
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('lift_expert_out01/exp1')

handles = [25, 30] 

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'lift_expert_out01'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)


timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"

torch.set_num_threads(8)
device = torch.device(dev)

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 128).double()
       #self.fc1 = nn.DataParallel(self.fc1).cuda()
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
        
with open('conf/rigidcloth/fold_cloth/fold_cloth.json','r') as f:
	config = json.load(f)
# matfile = config['cloths'][0]['materials'][0]['data']
# with open(matfile,'r') as f:
# 	matconfig = json.load(f)


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



# def get_loss(ans, goal):
# 	#[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
# 	dif = ans - goal
# 	loss = torch.norm(dif.narrow(0, 3, 3), p=2)

# 	return loss
def get_loss(xcoords, ycoords, height_diff):

    epsilon = 1e-8

    x = xcoords - torch.mean(xcoords)
    y = ycoords - torch.mean(ycoords)

    corr = torch.sum(x * y) / (epsilon + (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))))

    loss = 1-torch.abs(corr)

    #print(loss)
    #print(height_diff)
    loss = loss + 1. * height_diff 

    loss = loss.to(device)
    
    #print(ans)
    #print(goal)
    #print(loss)
    
    return loss
    
def run_sim(steps, sim, net):

    for step in range(steps):
        print(step)
        remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64).to(device)
        		
        net_input = []
        for node in sim.cloths[0].mesh.nodes:
            net_input.append(node.x.to(device))
            net_input.append(node.v.to(device))

        # dis = sim.obstacles[0].curr_state_mesh.dummy_node.x - goal
        # net_input.append(dis.narrow(0, 3, 3))
        net_input.append(remain_time)
        #net_input = [t.to(device) for t in net_input]
        net_input = torch.cat(net_input).to(device)
        net_output = net(net_input)
        
        # outputs = net_output.view([4, 3])
        		
        for i in range(len(handles)):
            sim_input = net_output[3*i:3*i+3]
            sim_input = torch.clamp(sim_input, -100, 100)
            sim_input = sim_input.to(torch.device('cpu'))
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        
        arcsim.sim_step()

    xcoords = [ node.x[0].to(device) for node in sim.cloths[0].mesh.nodes ]
    ycoords = [ node.x[1].to(device) for node in sim.cloths[0].mesh.nodes ]
    xcoords = torch.stack(xcoords)
    ycoords = torch.stack(ycoords)
    xcoords = xcoords.to(device)
    ycoords = ycoords.to(device)

    ans_handle0 = sim.cloths[0].mesh.nodes[handles[0]].x[2]
    ans_handle1 = sim.cloths[0].mesh.nodes[handles[1]].x[2]
    height_diff = torch.norm(ans_handle0 - ans_handle1) 
    height_diff = height_diff.to(device)

    loss = get_loss(xcoords, ycoords, height_diff)
        
    return loss

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    while True:
        #steps = int(1*15*spf)
        steps = 20
        
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
        
        print('forward time = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        
        if epoch % 5 == 0:
            torch.save(net.state_dict(), torch_model_path)
        
        #if loss<1e-3:
        #    break
        # dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
       
        #for param in net.parameters():
        #    param.grad.data.clamp_(-0.1, 0.1)
        optimizer.step()
        		
        if epoch>=400:
            quit()
        		
        epoch = epoch + 1
        # break

with open(out_path+'/log.txt','w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    # reset_sim(sim)
    
    #param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
    net = Net(487, 6)
    net = net.to(device)
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
    
print("done")

