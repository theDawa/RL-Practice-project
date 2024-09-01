import numpy as np
import gymnasium as gym
import gridworld 
import time

class Sarsa_Agent():
    """ Since the discrete actions have been redefined as {0,1,2,3} by using the wapper file, we can simply represent the action by a number. """
    
    def __init__(self,
                 status_n:int,
                 act_n:int,
                 e_greedy:float = 0.1,
                 lr:float = 0.1,
                 gamma:float = 0.9) -> None:
        
        self.status_n = status_n
        self.act_n = act_n
        self.Q = np.zeros((self.status_n,self.act_n))
        
        self.e_greedy = e_greedy
        self.lr = lr
        self.gamma = gamma

    # obtain action according to experience
    def predict(self,state:int) -> int:
        # action of determine policy by policy improvement
        Q_list = self.Q[state,:]
        # action = np.argmax(Q_list) # for this method, [0,0,0,0] will always choose action[0]
        action = np.random.choice(np.flatnonzero(Q_list==Q_list.max()))  # for this method, [0,0,0,0] will choose action[0,1,2,3] randomly. If the list has multiple max values, then choose the action randomly
        return action

    def get_action(self,state:int) -> int:
        # epsilon-greedy policy
        if np.random.uniform(0,1) < self.e_greedy:
            action = np.random.choice(self.act_n)
        else:
            # use improved policy
            action = self.predict(state)
            
        return action
    
    def policy_evaluation(self,
                          state:int,
                          action:int,
                          reward:float,
                          next_state:int,
                          next_action:int,
                          done:bool) -> None:
        current_Q = self.Q[state,action]
        # Note that if terminated is True, there will be no next_state and next_action. In this case, the target_Q is just reward
        TD_target = reward + (1-float(done)) * self.gamma * self.Q[next_state,next_action]
        self.Q[state,action] -= self.lr * (current_Q - TD_target)

# train an episode
def train_episode(env,agent,is_render:bool=False) -> float:
        total_reward = 0 # record total reward in one episode
        state,_ = env.reset() # reset env and get initial state
        action = agent.get_action(state) # get action using learned epsilon-greedy policy
        while True:
            
            next_state, reward, terminated, truncated, _ = env.step(action) # take action and get next_state, reward, done, info
            """In Gymnasium or Gym v0.26, done is True when terminated or truncated. 
                In Gym early version, there is done but not terminated, truncated here.
                You should modify the code according to the version of Gym you use."""
            done = terminated or truncated
            total_reward += reward
            # For Sarsa, we NEED obtain a' using the current policy        
            next_action = agent.get_action(next_state)
            # using data to do policy evaluation
            agent.policy_evaluation(state,action,reward,next_state,next_action,done)
            # update state and action
            state = next_state
            action = next_action     
            if is_render:
                env.render() # !! You can find the game window in the taskbar !!
                time.sleep(0.1) # set a speed for visualization
                
            if done:
                break
            
        return total_reward       

def test_episode(env, agent) -> float:
        total_reward = 0 # record total reward in one episode
        state,_ = env.reset() # reset env and get initial state
        while True:
            action = agent.predict(state) # get action using learned greedy policy
            next_state, reward, terminated, truncated, _= env.step(action) # take action and get next_state, reward, done, info
            done = terminated or truncated
            state = next_state
            total_reward += reward
            env.render()
            time.sleep(0.1)
            if done:
                break
            
        return total_reward

def train(env,episodes=500,lr=0.1,gamma=0.9,e_greed=0.1) -> None: 
        agent = Sarsa_Agent(status_n=env.observation_space.n, act_n=env.action_space.n)      
        is_render = False
        for e in range(episodes):
            episode_reward = train_episode(env, agent, is_render)
            print('Episode %s: Total Reward = %.2f'%(e,episode_reward)) 
            
            if e%50==0:
                is_render = True
            else:
                is_render = False
                
        test_reward = test_episode(env, agent)
        print('Test Total Reward = %.2f'%(test_reward))

if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")  # 0：up, 1: right, 2: down, 3: left
    env = gridworld.CliffWalkingWapper(env)
    train(env)