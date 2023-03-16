import gym
import numpy as np 
import random
#from gym_minigrid.wrappers import *

env = gym.make('MiniGrid-Unlock-v0',render_mode = 'human')
#env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
#window = Window("minigrid")
steps = []
rew = []
ep = []


env.reset()
alpha = 0.2
epsilon =1
decay_rate = 1.1
total_episodes = 300
actions = [0,1,2,5]
gamma=0.9


def hashing(a,b,c): 
     return(a +(3*b)+(9*c))
     
def epsilon_greedy(hash ,epsilon):
    
    
    #explore:
    if np.random.uniform(0,1) < epsilon:
        action = random.choice([0,1,2,5])
        
    #greedy policy:
    else:
        t= max(q[hash])
        action = q[hash].index(t)
    return action
s =[]


x,y =env.agent_pos
d =env.agent_dir
q  = { hashing(x,y,d):[0,0,0,0]}
for episode in range(150):
    env.reset()
    print(episode)
    x,y = env.agent_pos
    d  =env.agent_dir
    p=hashing(x,y,d)

    if q.get(p) is None :
        q[p]=[0,0,0,0]
    epsilon = epsilon/decay_rate
    done = False
    R=0
    S=0
    
    while not done:
        a1 =epsilon_greedy(p,epsilon)
        #env.render()
        obs,reward,done,info,_=env.step(a1)
        R=R+reward
        S=S+1
        x1, y1 = env.agent_pos
        d1 =env.agent_dir
        r = hashing(x1,y1,d1)
        if q.get(r) is None:
            q[r] = [0,0,0,0]
    
        a2 = epsilon_greedy(r, 0)


        q[p][a1]=  q[p][a1]+ alpha*((reward + gamma*(q[r][a2]))-q[p][a1])
        
    
    
        p=r
       # a1 = a2
    rew.append(R)
    steps.append(S)    
        #img = env.get_frame()
        # window.show_img(img)
#env.close()
print("   reward list :   ")
print(rew)
print("  steps  ")
print(  steps)
#print(" episode no. ")
#print(ep)
       