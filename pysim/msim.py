import torch
import arcsim

with torch.autograd.profiler.profile() as prof:
        #arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/fold_cloth/folded_flag.json','out'])
    
        #arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/ground.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/multibody/multibody_make.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/manipulation/manipulation.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/clothbunny/clothbunny.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/bounce/bounce.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/game2/game2.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/q_rigid_gravity.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/scale/scale.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/multibody/multibody_make.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/multi/multi.json','out'])

	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/drag/drag_cloth.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/drag/gravity_cloth.json','out'])
        
        # arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/manipulation/manipulation_vid.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/demo_collision2.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/absparse/multibody_make.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/absparse/abqr_make.json','out'])
	# # arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/absparse/multi.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/absparse/multi_make.json','out'])
	
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/circular_domino/circular_domino_make.json','out'])
         arcsim.msim(3,['arcsim','replay','rotate_out/exp3/out200'])
        #arcsim.msim(3,['arcsim','replay','fold_expert_out/out100'])
	#arcsim.msim(3,['arcsim','replay','test/out1'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/circular_domino/debug.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/gravity.json','out'])
	#arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/domino.json','out'])
	# arcsim.msim(4,['arcsim','simulate','conf/rigidcloth/circular_domino/test.json','out'])
	#arcsim.msim(4,['arcsim','resume','conf/gravity.json','out'])
	
	#arcsim.msim(3,['arcsim','replay','rigidcloth/domino'])
	# arcsim.msim(3,['arcsim','replay','out'])
	# arcsim.msim(3,['arcsim','replay','200122_circulat_dominod'])
	# arcsim.msim(3,['arcsim','replay','200121_domino/out6'])
	
	# arcsim.msim(3,['arcsim','replay','rigidcloth/domino_jump'])
	# arcsim.msim(3,['arcsim','replay','200118_bunny/out'])
	#arcsim.msim(3,['arcsim','replay','rigidcloth/domino2'])
	#arcsim.msim(3,['arcsim','replay','rigidcloth/jump'])
	# arcsim.msim(3,['arcsim','replay','200118_bunny/out'])
	# arcsim.msim(3,['arcsim','replay','200205_circulat_dominod/out_damping9'])
	# arcsim.msim(3,['arcsim','replay','200118_bunny/out'])
	# arcsim.msim(3,['arcsim','replay','200118_bounce/out'])
print("adfsadf")
print(prof)
