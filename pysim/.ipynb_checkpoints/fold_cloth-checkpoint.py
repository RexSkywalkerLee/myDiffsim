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

handles = [25, 60, 30, 54] 

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'fold_cloth_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)


timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

#if torch.cuda.is_available():
#    dev = "cuda:0"
#else:
#    dev = "cpu"
#    torch.set_num_threads(8)
#device = torch.device(dev)
#

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 500).double()
        self.fc2 = nn.Linear(500, 200).double()
        self.fc3 = nn.Linear(200, 50).double()
        self.fc4 = nn.Linear(50, n_output).double()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.cuda()
        # x = torch.clamp(x, min=-5, max=5)
        return x
        
with open('conf/rigidcloth/fold_cloth/fold_cloth.json','r') as f:
	config = json.load(f)
# matfile = config['cloths'][0]['materials'][0]['data']
# with open(matfile,'r') as f:
# 	matconfig = json.load(f)


goal = []
with open('meshes/rigidcloth/drag/cloth.obj','r') as f:
    for line in f:
        if 'v ' in line:
            pos = [float(i) for i in line[2:].split()]
            new_pos = torch.tensor(pos, dtype=torch.float64)
            goal.append(new_pos)
goal = torch.stack(goal)
goal = goal.cuda()

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')


spf = config['frame_steps']

scalev=1

def reset_sim(sim, epoch, goal):
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
def get_loss(ans, goal):
    #[0.0000, 0.0000, 0.0000, 0.7500, 0.6954, 0.3159
    diff = ans - goal
    diff[:,2] *= 10
    loss = torch.norm(diff)
    loss = loss.cuda()
    
    #print(ans)
    #print(goal)
    #print(loss)
    
    return loss
    
def run_sim(steps, sim, net, goal):

   #for obstacle in sim.obstacles:
       #for node in obstacle.curr_state_mesh.nodes:
           #node.m    *= 0.2

    #sim.obstacles[0].curr_state_mesh.dummy_node.x = torch.tensor([0.0000, 0.0000, 0.0000,
    #np.random.random(), np.random.random(), -np.random.random()],dtype=torch.float64)
    #print(sim.obstacles[0].curr_state_mesh.dummy_node.x)
    for step in range(steps):
        print(step)
        remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64)
        		
        net_input = []
        for node in sim.cloths[0].mesh.nodes:
            net_input.append(node.x)
            net_input.append(node.v)

        # dis = sim.obstacles[0].curr_state_mesh.dummy_node.x - goal
        # net_input.append(dis.narrow(0, 3, 3))
        net_input.append(remain_time)
		net_input = [t.cuda() for t in net_input]
        net_output = net(torch.cat(net_input))
        		
        
        # outputs = net_output.view([4, 3])
        		
        for i in range(len(handles)):
            sim_input = net_output[3*i:3*i+3]
            sim_input = sim_input.to(torch.device('cpu'))
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        
        arcsim.sim_step()

    #cnt = 0
    #ans1 = torch.tensor([0, 0, 0],dtype=torch.float64)

    ans = [ node.x for node in sim.cloths[0].mesh.nodes ]
    ans = torch.stack(ans)
    ans = ans.cuda()

    #ans1 /= cnt
    #ans1 = torch.cat([torch.tensor([0, 0, 0],dtype=torch.float64), ans1])
      
    #ans  = ans1
    #ans = sim.obstacles[0].curr_state_mesh.dummy_node.x
        
    loss = get_loss(ans, goal)
        
    return loss, ans

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    global goal
    while True:
        #steps = int(1*15*spf)
        steps = 20
        
        #sigma = 0.05
        #z = np.random.random()*sigma + 0.5
        #y = np.random.random()*sigma - sigma/2
        #x = np.random.random()*sigma - sigma/2
        
        
        #ini_co = torch.tensor([0.0000, 0.0000, 0.0000,0.4744, 0.4751, 0.0564], dtype=torch.float64)
        		
        #goal = torch.tensor([0.0000, 0.0000, 0.0000, 0, 0, z],dtype=torch.float64)
        #goal = goal + ini_co
        
        reset_sim(sim, epoch, goal)
        
        st = time.time()
        loss, ans = run_sim(steps, sim, net, goal)
        en0 = time.time()
        		
        optimizer.zero_grad()
        
        loss.backward()
        
        en1 = time.time()
        print("=======================================")
        f.write('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(epoch, loss.data, ans.data, goal.data))
       #print('epoch {}: loss={}\n  ans = {}\n goal = {}\n'.format(epoch, loss.data, ans.data, goal.data))
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward time = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        
        if epoch % 5 == 0:
            torch.save(net.state_dict(), torch_model_path)
        
        if loss<1e-3:
            break
        # dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])
        
        optimizer.step()
        		
        if epoch>=400:
            quit()
        		
        epoch = epoch + 1
        # break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    # reset_sim(sim)
    
    #param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
    net = Net(487, 12)
    net = nn.DataParallel(net)
    net = net.cuda()
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

