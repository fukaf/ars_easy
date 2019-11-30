import numpy as np
import os
import matplotlib.pyplot as plt

env_name = "LunarLanderContinuous-v2"
param = "1000_1000_0.025_4_2_0.03_1_v2"
filename = os.path.join("exp/brs/trained_policy", env_name + "/"+param)
a = np.load(filename+".npz")
x = np.linspace(1,1000,1000)
y = a['reward']
plt.plot(x,y)
plt.title(env_name + " w/ " + param)
plt.ylabel("total_reward")
plt.xlabel("steps")
plt.savefig(filename+".png")