import numpy as np
import random

n_states=5
actions=[-1,1]
Q=np.zeros((n_states,len(actions)))
alpha=0.1
gamma=0.9
epsilon=0.1
episodes=1000

def step(state,action):
    next_state=state+action
    if next_state <0 or next_state >=n_states:
        return state,0
    reward = 1 if next_state ==4 else 0
    return next_state , reward

def choose_action(state):
    if random.random()<epsilon:
        return random.choice([0,1])
    else:
        return np.argmax(Q[state])
print("Train")
for _ in range(episodes):
    state=random.randint(0,3)
    while state!=4:
        action_idx=choose_action(state)
        action=actions[action_idx]
        next_state,reward=step(state,action)
        Q[state,action_idx]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action_idx])
        state=next_state

print("Learned Q-table:")
print(Q)

state=0
print("\nTest:")
while state!=4:
    action_idx=np.argmax(Q[state])
    action=actions[action_idx]
    next_state,reward=step(state,action)
    print(f"State:{state},Action:{action},Reward:{reward},NextState:{next_state}")
    state=next_state