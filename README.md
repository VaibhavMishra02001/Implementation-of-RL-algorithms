# Implementation-of-RL-algorithms
Agent is trained in gym-Minigrid env.


gym-Minigrid https://github.com/maximecb/gym-minigrid


The Agent is trained using 3 algorithms:
1. Sarsa 
2. Sarsa Lambda
3. Q-Learning

The agent was trained on 4 environments:

1.6x6 Empty room
2.8x8 Empty room
## Observation Space
For discrete observation, we use agent_pos, which returns the grid number at which the agent is present and agent_dir which returns the the direction at which the agent is pointing.

## Action Space 
* 0	:Turn Left
* 1	:Turn Right
* 2	:Move Forward

## Reward Function 
Reward is 1 when agent reaches goal, elswhere 0.

## Hyperparameters
* Gamma = 0.9
* Alpha = 0.2
* Exploration 
    epsilon = epsilon/1.1
* Training Episodes = 150  
   



# 6x6 Empty Room 

Hyperparameters:
1. 
 
## Simulation -
![20230317010741](https://user-images.githubusercontent.com/109021179/225740672-c190c049-6d3c-4bdb-89eb-385cef883bbe.gif)

## Sarsa

![2 ](https://user-images.githubusercontent.com/109021179/225908985-6fae7b75-2e6d-415b-942b-796ecb5eff5b.png)  ![Figure_1](https://user-images.githubusercontent.com/109021179/225892794-a002fb91-fd8b-46c8-b98b-319b07ef198a.png )
## Sarsa lambda
![Figure_1](https://user-images.githubusercontent.com/109021179/225893486-8dc19a07-4042-4c33-80fd-ba95669fa585.png)

![2](https://user-images.githubusercontent.com/109021179/225892980-0232a28c-359a-4a56-9634-ae98fa75036f.png)
## Q-Learning

![Figure_1](https://user-images.githubusercontent.com/109021179/225893145-add080e1-27be-4bba-85e1-c4df0daa5085.png)
![2](https://user-images.githubusercontent.com/109021179/225894014-8c246efb-c8b1-4206-8e27-1389d5a57ed5.png)

## Comparison of algorithms:
![2](https://user-images.githubusercontent.com/109021179/225894395-0668564d-6a84-4d4a-bc09-68690ac1c307.png)

![Figure_1](https://user-images.githubusercontent.com/109021179/225894302-d72ad635-37e7-4980-a8df-b8db5e7b03fb.png)

# 8X8 Empty Room

## Simulation
![20230316233932](https://user-images.githubusercontent.com/109021179/225906097-33634596-6e56-412b-9bee-ab7649075a96.gif)

## Sarsa
![2](https://user-images.githubusercontent.com/109021179/225897340-c8ebe377-f3ea-4fab-a3a7-6e133167f211.png)
![Figure_1](https://user-images.githubusercontent.com/109021179/225897656-38c26c04-9a7b-4307-bafe-6334c729e120.png)
## Sarsa lambda
![2](https://user-images.githubusercontent.com/109021179/225898124-f15d632e-38af-4a12-8b1d-5edce4ebac45.png)
![Figure_1](https://user-images.githubusercontent.com/109021179/225897995-b0a464c5-9738-453d-a7a6-8c39d9404fd8.png)

## Q-Learning
![2](https://user-images.githubusercontent.com/109021179/225899132-52ce2dde-b16f-4652-b8a4-a972f86f5a32.png)

![Figure_1](https://user-images.githubusercontent.com/109021179/225898400-4038c8c1-666f-46c4-aa39-d66a9c8c88f5.png)

## Comparison
![3](https://user-images.githubusercontent.com/109021179/225898613-3f42cd1d-72f7-4957-b800-c01162b2b5c3.png)
![4](https://user-images.githubusercontent.com/109021179/225898677-82e1ac7b-43a1-4ae3-9334-8a4e248e99f6.png)

## Vatriation on changing Learning Rate 
![6](https://user-images.githubusercontent.com/109021179/225971816-e4ab977f-1974-4d73-b742-77231b4d5ecb.png)
![5](https://user-images.githubusercontent.com/109021179/225971942-63151659-30ce-4d02-b71b-554f4b688c11.png)

## Variation with changing Discount Factor(gamma)
![7](https://user-images.githubusercontent.com/109021179/225972153-83cd81d5-9ba1-4a19-a7b1-2c8b7988a253.png)
![8](https://user-images.githubusercontent.com/109021179/225972204-9d11ea60-1082-4fb2-a254-ecc1477455b8.png)






