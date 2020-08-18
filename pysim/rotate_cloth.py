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


writer = SummaryWriter('rotate_out/exp4')

handles = [10, 51, 41, 57]
ref_points = [25, 60, 30, 54]
center = 62

losses = []

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'rotate_out/exp4/'
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
        
with open('conf/rigidcloth/drag/drag_cloth.json','r') as f:
	config = json.load(f)

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

#def get_loss(xcoords, ycoords, height_diff):
#
#    epsilon = 1e-3
#
#    x = xcoords - torch.mean(xcoords)
#    y = ycoords - torch.mean(ycoords)
#
#    corr = torch.sum(x * y) / ((torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))) + epsilon)
#
#    loss = -torch.abs(corr)
#
#    #print(loss)
#    #print(height_diff)
#    loss = loss + 10 * height_diff 
#
#    loss = loss.to(device)
#    
#    #print(ans)
#    #print(goal)
#    #print(loss)
#    
#    return loss

def visualize_loss(losses,dir_name):
    plt.plot(losses)
    plt.title('losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(dir_name+'/'+'loss'+'.jpg')


def get_loss(ans):

    loss = torch.norm(ans[-1,:] - torch.tensor([0.500000, 0.502674, -0.000000], dtype=torch.float64).to(device))
    loss = loss + torch.norm(ans[:,-1]) 
    loss = loss.to(device)

    return loss

def run_sim(steps, sim, net):

    cum_loss = torch.tensor([0.0], dtype=torch.float64)
    cum_loss = cum_loss.to(device)

    for step in range(steps):
        print(step)
       #remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64).to(device)
        		
        net_input = []
        for i in range(len(ref_points)):
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].v)
        for i in range(len(handles)):
            net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
        net_output = net(torch.cat(net_input).to(device))

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

        loss = get_loss(ans)

        cum_loss = cum_loss + loss

    return cum_loss.to(device)

def do_train(optimizer,scheduler,sim,net):
    epoch = 1
    while True:
        #steps = int(1*15*spf)
        steps = 40
        
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
    net = Net(48, 12)
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

