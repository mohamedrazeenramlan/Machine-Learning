import numpy as np
import random

# ==========================================
# 1. THE ENVIRONMENT (The Grid & Rules)
# ==========================================
class GridWorld:
    def __init__(self):
        self.cols = 5
        self.rows = 3
        self.start_state = (0, 0)
        self.target_state = (4, 0)
        self.walls = [(2, 0), (2, 1), (3, 0)]
        self.current_state = self.start_state
        
        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = [0, 1, 2, 3]

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        x, y = self.current_state
        
        if action == 0:   # Up
            next_y = min(y + 1, self.rows - 1)
            next_state = (x, next_y)
        elif action == 1: # Down
            next_y = max(y - 1, 0)
            next_state = (x, next_y)
        elif action == 2: # Left
            next_x = max(x - 1, 0)
            next_state = (next_x, y)
        elif action == 3: # Right
            next_x = min(x + 1, self.cols - 1)
            next_state = (next_x, y)

        if next_state in self.walls:
            next_state = self.current_state

        self.current_state = next_state

        if self.current_state == self.target_state:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return self.current_state, reward, done

# ==========================================
# 2. THE Q-LEARNING ALGORITHM (Training)
# ==========================================
env = GridWorld()
q_table = np.zeros((env.cols, env.rows, len(env.actions)))

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0       
epsilon_decay = 0.99
epochs = 500

print("Training Agent... Please wait.")

for epoch in range(epochs):
    state = env.reset()
    done = False
    
    while not done:
        x, y = state
        
        # Epsilon-Greedy: Choose random action OR best known action
        if random.uniform(0, 1) < epsilon:
            action = random.choice(env.actions)
        else:
            action = np.argmax(q_table[x, y])
            
        next_state, reward, done = env.step(action)
        next_x, next_y = next_state
        
        # The Bellman Equation (Updating the Q-Table)
        old_value = q_table[x, y, action]
        next_max = np.max(q_table[next_x, next_y])
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[x, y, action] = new_value
        
        state = next_state
        
    epsilon = max(0.1, epsilon * epsilon_decay)

# ==========================================
# 3. THE REPORTING LOOP (Testing)
# ==========================================
print("\n--- Final Optimal Path Report ---")
state = env.reset()
done = False
action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

step_count = 1
while not done:
    x, y = state
    # Now we only exploit the learned Q-Table (no more random exploration)
    action = np.argmax(q_table[x, y]) 
    
    next_state, reward, done = env.step(action)
    
    print(f"Step {step_count}:")
    print(f"  Current State: {state}")
    print(f"  Action Taken:  {action} ({action_names[action]})")
    print(f"  Next State:    {next_state}")
    print(f"  Reward:        {reward}\n")
    
    state = next_state
    step_count += 1
    
    if step_count > 20: 
        print("Agent got stuck!")
        break
        
print("Target Reached!")