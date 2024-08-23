import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

SIZE = 10           # area of the game environment
EPISODES = 30000    
SHOW_EVERY = 3000   # show image every 3000 episode

FOOD_REWARD = 25      
ENEMY_PENALITY = 300  
MOVE_PENALITY = 1     

epsilon = 0.6
EPS_DECAY = 0.9998
DISCOUNT = 0.95
LEARNING_RATE = 0.1

q_table = None
# player-blue food-green enemy-red 
d = {1:(255,0,0), # blue
     2:(0,255,0), # green
     3:(0,0,255)} # red

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# agent class, contain position and move function
class Cube:
     def __init__(self): # generate initial position randomly
         self.x = np.random.randint(0, SIZE-1)
         self.y = np.random.randint(0, SIZE-1)

     def __str__(self):
        return f'{self.x},{self.y}'
     
     def __sub__(self, other):
        return (self.x-other.x,self.y- other.y)
     
     def action(self,choise):
        if choise == 0:
            self.move(x=1,y=1)
        elif choise == 1:
            self.move(x=-1, y=1)
        elif choise == 2:
            self.move(x=1, y=-1)
        elif choise == 3:
            self.move(x=-1, y=-1)

     def move(self,x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x< 0:
            self.x = 0
        if self.x> SIZE -1:
            self.x = SIZE-1
        if self.y< 0:
            self.y = 0
        if self.y> SIZE -1:
            self.y = SIZE-1

# init q-table
if q_table is None:   # if there is no provided q-table, generate one randomly
    q_table = {}
    for x1 in range(-SIZE+1,SIZE): # x1: distance between food and agent on x direction 
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE): # x2: distance between enemy and agent on x direction
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1,y1),(x2,y2))] = [np.random.randint(-5,0) for i in range(4)]
else:                # use the provided q-table
    with open(q_table,'rb') as f:
        q_table= pickle.load(f)

# train an agent
episode_rewards = []  #init reward sequence
for episode in range (EPISODES):
    # init player, food, enemy
    player = Cube()
    food = Cube()
    enemy = Cube()

    # set show to true to display image every 3000 episodes
    if episode % SHOW_EVERY == 0:
        print('episode ',episode,'  epsilon:',epsilon)
        print('mean_reward:',np.mean(episode_rewards[-SHOW_EVERY:]))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)  # state
        # explore + exploit
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])   # choose an action with highes q value
        else:
            action = np.random.randint(0,4)    # choose an action randomly
        # print("player的位置：",player)
        # print("player的观测：",obs)
        # print("player的动作：",action)
        player.action(action)              # agent perform an action
        # food.move()
        # enemy.move()

        # print("player的下一步位置：",player)
        # reward
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif player.x == enemy.x and player.y == enemy.y:
            reward = - ENEMY_PENALITY
        else:
            reward = - MOVE_PENALITY
        # print('reward:',reward)
        # renew q-table
        current_q = q_table[obs][action]            # q-value of current state and action
        # print('current_q:',current_q)
        new_obs = (player-food,player-enemy)        # new state after action
        # print('new_obs:',new_obs)
        max_future_q = np.max(q_table[new_obs])     # biggest q value of new state
        # print('max_future_q:',max_future_q)
        # print('')
        if reward ==FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT* max_future_q)
        q_table[obs][action] = new_q

        # display the image
        if show:
            env = np.zeros((SIZE,SIZE,3),dtype= np.uint8)
            env[food.x][food.y] = d[FOOD_N]
            env[player.x][player.y] = d[PLAYER_N]
            env[enemy.x][enemy.y] = d[ENEMY_N]
            img = Image.fromarray(env,'RGB')
            img = img.resize((800,800))
            cv2.imshow('',np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALITY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
     
        episode_reward += reward

        if reward == FOOD_REWARD or reward ==ENEMY_PENALITY:
            break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY


moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY,mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean{SHOW_EVERY} reward')
plt.show()

with open(f'qtable_{int(time.time())}.pickle','wb') as f:
    pickle.dump(q_table,f)