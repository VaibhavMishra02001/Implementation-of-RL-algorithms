import gym
import numpy as np 
import random
#from gym_minigrid.wrappers import *
#import gym_minigrid
#from gym_minigrid import Window
import matplotlib
#env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
env = gym.make('MiniGrid-Empty-8x8-v0')
#env = gym.make('MiniGrid-Empty-8x8-v0')
#window = Window("minigrid")
env.reset()

#x =env.agent_pos
#y= env.agent_pos
#print(x,y)
steps = []
rew = []
ep = []
Q = {(env.agent_pos[0],env.agent_pos[1]):{dire:{a:0 for a in range(3)}for dire in range (4) }}
policy= {(env.agent_pos[0],env.agent_pos[1]):{dire:0 for dire in range (4)}}

alpha = 0.4
epsilon =1
decay_rate = 1.1
total_episodes = 150
actions = [0,1,2]
gamma=0.9

for episodes in range (total_episodes):
    env.reset()
    #env.render()

   # state1 = (env.agent_pos[0],env.agent_pos[1])
    #d1 = env.agent_dir
    print(episodes+1)
    ep.append(episodes+1)
    epsilon = epsilon/decay_rate
    x1,y1 = env.agent_pos
    state1 =x1,y1
    d1 = env.agent_dir
    if Q.get(state1) is None:
        Q[state1]={dire:{a:0 for a in range (3)}for dire in range(4)}
    if policy.get(state1) is None:
        policy[state1]={dire:0 for dire in range (4)}
    
    #if(np.random.uniform(0,1)<epsilon):
     #    a1 = random.choice(actions)
    #else:
     #   a1 = policy[state1][d1]   
    R =0 
    S = 0
    done = False
    while (not done):
      
     
      
      if(np.random.uniform(0,1)<epsilon):
        a1 = random.choice(actions)
        policy[state1][d1]=a1
      else:
        a1 = policy[state1][d1] 

      obs,reward,done,info,_=env.step(policy[state1][d1])
      x2,y2 = env.agent_pos
      state2 = x2,y2
      d2=env.agent_dir
      R = R+reward
      S = S+1
      if Q.get(state2) is None:
            Q[state2]={dire:{a:0 for a in range(3)}for dire in range(4)}
         

      if policy.get(state2) is None:
            policy[state2]={dire:0 for dire in range (4)}


      a2=0
      if np.random.uniform(0,1)<epsilon:
            a2 = random.choice(actions)
            policy[state2][d2]=a2
      else:
            
            a2 = policy[state2][d2]
       
        #print(policy,d2,state2)

      Q[state1][d1][a1]= Q[state1][d1][a1]+ (alpha*(reward+ (gamma*Q[state2][d2][a2])-Q[state1][d1][a1]))
      state1 =state2
      d1=d2
      a1=a2
      
       
     
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