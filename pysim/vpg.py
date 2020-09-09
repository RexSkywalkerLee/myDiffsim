import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import scipy.signal
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
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs



handles = [10, 51, 41, 57]
ref_points = [25, 60, 30, 54]
center = 62

cloth = [25, 24, 3, 70, 7, 6, 58, 31, 30, 71, 26, 4, 5, 80, 8, 17, 19, 66, 55, 28, 29, 50, 49, 27, 18, 32, 68,
         42, 2, 1, 51, 76, 57, 56, 33, 67, 43, 72, 0, 61, 62, 45, 44, 63, 65, 77, 75, 9, 10, 12, 41, 78, 64, 79,
         73, 74, 34, 20, 11, 39, 40, 47, 46, 59, 23, 22, 15, 14, 37, 36, 48, 52, 60, 38, 21, 16, 13, 69, 35, 53, 54]

rets = []

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'algo_out/exp1/'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

writer = SummaryWriter(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path

# if torch.cuda.is_available():
#     dev = "cuda:0"
# else:
dev = "cpu"

torch.set_num_threads(20)
device = torch.device(dev)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers).to(device)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float64)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation).double()
        
    def _distribution(self, obs):
        mu = self.mu_net(obs.double())
        std = torch.exp(self.log_std).to(device)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1).to(device)
    
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation).double()

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs.double()), -1).to(device)
    
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64,64), activation=nn.Tanh):
        super().__init__()
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]
    
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class VPGBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float64)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float64)
        self.adv_buf = np.zeros(size, dtype=np.float64)
        self.rew_buf = np.zeros(size, dtype=np.float64)
        self.ret_buf = np.zeros(size, dtype=np.float64)
        self.val_buf = np.zeros(size, dtype=np.float64)
        self.logp_buf = np.zeros(size, dtype=np.float64)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs.detach().cpu()
        self.act_buf[self.ptr] = act.cpu()
        self.rew_buf[self.ptr] = rew.cpu()
        self.val_buf[self.ptr] = val.cpu()
        self.logp_buf[self.ptr] = logp.cpu()
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val.cpu())
        vals = np.append(self.val_buf[path_slice], last_val.cpu())
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    
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

def visualize_loss(rets,dir_name):
    plt.plot(rets)
    plt.title('returns')
    plt.xlabel('epochs')
    plt.ylabel('returns')
    plt.savefig(dir_name+'/'+'loss'+'.jpg')

def get_reference_loss(ans, goal):
    diff = ans - goal
    loss = torch.norm(diff)
    loss = loss.to(device)

    return loss

def compute_loss_pi(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    pi, logp = ac.pi(obs, act)
    loss_pi = -(logp * adv).mean().to(device)

    return loss_pi

# Set up function for computing value loss
def compute_loss_v(data):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean().to(device)

def run_sim(steps, sim, ac, buf):
    
    ret = 0
    
    net_input = []
    for i in range(len(ref_points)):
        net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].x)
        net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].v)
    for i in range(len(handles)):
        net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
        net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
    net_input.append(torch.tensor(0, dtype=torch.float64).view(1))
    o = torch.cat(net_input).to(device)

    for step in range(steps):
        print(step)
        
        a, v, logp = ac.step(o)

        for i in range(len(handles)):
            sim_input = a[3*i:3*i+3]
            sim_input = torch.clamp(sim_input, -1e5, 1e5)
            sim_input = sim_input.cpu()
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input
        arcsim.sim_step()
        
        ans = [ node.x.to(device) for node in sim.cloths[0].mesh.nodes ]
        ans = torch.stack(ans)
        ans = ans.to(device)
        r = -get_reference_loss(ans, goal)
        
        buf.store(o, a, r, v, logp)
        
        net_input = []
        for i in range(len(ref_points)):
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[ref_points[i]].v)
        for i in range(len(handles)):
            net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
            net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
        net_input.append(torch.tensor(step/steps, dtype=torch.float64).view(1))
        o = torch.cat(net_input).to(device)
        
        ret = ret + r
    
    _, v, _ = ac.step(o)
    buf.finish_path(v)
    ret = ret.to(device) / steps
    return ret

def do_train(pi_optimizer,pi_scheduler,vf_optimizer,vf_scheduler,sim,ac):
    epoch = 1
    
    while True:
        steps = 15
        train_v_iters = 20
        
        buf = VPGBuffer(obs_dim=49, act_dim=12, size=steps, gamma=0.99, lam=0.97)
        
        reset_sim(sim, epoch)
        
        st = time.time()
        ret = run_sim(steps, sim, ac, buf)
        en0 = time.time()
        data = buf.get()
        
        pi_l_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()
        
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        #mpi_avg_grads(ac.pi)  
        pi_optimizer.step()
        pi_scheduler.step()
        
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(ac.v)
            vf_optimizer.step()
            vf_scheduler.step()
        
        writer.add_scalar('training loss', ret, epoch)

        en1 = time.time()
        print("=======================================")
        f.write('epoch {}: return={}\n'.format(epoch, ret.data))
        print('epoch {}: return={}\n'.format(epoch, ret.data))
        
        rets.append(ret.data)

        print('forward time = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        
        if epoch % 5 == 0:
            torch.save(ac.pi.mu_net.state_dict(), torch_model_path+('/actor_weight.pth%s'%timestamp))
            torch.save(ac.v.v_net.state_dict(), torch_model_path+('/critic_weight.pth%s'%timestamp))
        
        #if loss<1e-3:
        #    break
        # dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])

        if epoch >= 200:
            break
        
        epoch = epoch + 1


with open(out_path+'/log.txt','w',buffering=1) as f:
    sim=arcsim.get_sim()
    # reset_sim(sim)
    
   #param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
   #net = Net(49, 12)
   #net = CNNet(12)
   #net = net.to(device)
    ac = MLPActorCritic(49, 12)
#     if os.path.exists(torch_model_path):
#         net.load_state_dict(torch.load(torch_model_path))
#         print("load: %s\n success" % torch_model_path)
    
    pi_lr = 1e-2
    pi_momentum = 0.9
    vf_lr = 1e-2
    vf_momentum = 0.9
    pi_optimizer = torch.optim.SGD(ac.pi.parameters(), lr=pi_lr, momentum=pi_momentum)
    vf_optimizer = torch.optim.SGD(ac.v.parameters(), lr=vf_lr, momentum=vf_momentum)
    pi_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(pi_optimizer, T_0=50, T_mult=1, eta_min=0, last_epoch=-1)
    vf_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(vf_optimizer, T_0=50, T_mult=1, eta_min=0, last_epoch=-1)
   #optimizer = torch.optim.Adam(net.parameters(),lr=lr)
   #optimizer = torch.optim.Adadelta([density, stretch, bend])
    do_train(pi_optimizer,pi_scheduler,vf_optimizer,vf_scheduler,sim,ac)
    rets = np.array(rets,dtype=np.float32)
    np.save(out_path+'/'+'_loss',rets)
    visualize_loss(rets,out_path)
print("done")

