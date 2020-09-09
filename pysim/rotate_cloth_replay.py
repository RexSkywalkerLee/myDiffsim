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

#handles = [10, 51, 41, 57]
#ref_points = [25, 60, 30, 54]
center = 62

losses = []

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'rotate_out/exp11'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth2020-08-20_12:05:52')

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

def reset_sim(sim):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/replay',False)

def run_sim(steps, sim, net):

    for step in range(steps):
        print(step)
       #remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64).to(device)
        		
        r_steps = 15
        r_step = step % 15
        
        if step%60<15:
            r_handles = [10, 51, 41, 57]
            r_ref_points = [25, 60, 30, 54]
        elif step%60>=15 and step%60<30:
            r_handles = [51, 57, 10, 41]
            r_ref_points = [30, 25, 54, 60]
        elif step%60>=30 and step%60<45:
            r_handles = [57, 41, 51, 10]
            r_ref_points = [54, 30, 60, 25]
        else:
            r_handles = [41, 10, 57, 51]
            r_ref_points = [60, 54, 25, 30]


        net_input = []
        for i in range(len(r_ref_points)):
            net_input.append(sim.cloths[0].mesh.nodes[r_ref_points[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[r_ref_points[i]].v)
        for i in range(len(r_handles)):
            net_input.append(sim.cloths[0].mesh.nodes[r_handles[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[r_handles[i]].v)
        net_input.append(torch.tensor(r_step/r_steps, dtype=torch.float64).view(1))
        net_output = net(torch.cat(net_input).to(device))

        for i in range(len(r_handles)):
            sim_input = net_output[3*i:3*i+3]
            sim_input = torch.clamp(sim_input, -1e5, 1e5)
            sim_input = sim_input.cpu()
            sim.cloths[0].mesh.nodes[r_handles[i]].v += sim_input

        arcsim.sim_step()

with open(out_path+'/log.txt','r',buffering=1) as f:
    sim=arcsim.get_sim()
    # reset_sim(sim)
    
    #param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
    net = Net(49, 12)
    net = net.to(device)
    if os.path.exists(torch_model_path):
        net.load_state_dict(torch.load(torch_model_path, map_location=torch.device('cpu')))
        print("load: %s\n success" % torch_model_path)
    else:
        print("load fail!")
        quit()

    reset_sim(sim)
    run_sim(300,sim,net)
print("done")

