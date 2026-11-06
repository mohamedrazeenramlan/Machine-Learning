import numpy as np
import random
import time
import os


class VisualGridWorld:
    def __init__(self):
        self.cols = 8
        self.rows = 8
        self.start_state = (0, 0)   # Bottom-Left (A1)
        self.target_state = (7, 7)  # Top-Right (H8)
        
        # Place Rooks (Walls) to create a maze-like obstacle
        self.walls = [(2, 0), (2, 1), (2, 2), (2, 3), 
                      (5, 4), (5, 5), (5, 6), (5, 7)]
        
        self.current_state = self.start_state
        self.actions = [0, 1, 2, 3] # Up, Down, Left, Right

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        x, y = self.current_state
        
        if action == 0:   next_state = (x, min(y + 1, self.rows - 1)) # Up
        elif action == 1: next_state = (x, max(y - 1, 0))             # Down
        elif action == 2: next_state = (max(x - 1, 0), y)             # Left
        elif action == 3: next_state = (min(x + 1, self.cols - 1), y) # Right

        # Wall collision
        if next_state in self.walls:
            next_state = self.current_state

        self.current_state = next_state

        if self.current_state == self.target_state:
            return self.current_state, 0, True
        else:
            return self.current_state, -1, False

    def render(self, step_num, action_name):
        # Clears the terminal for a smooth animation frame
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- Step {step_num} | Action: {action_name} ---")
        
        # Print rows backwards (7 down to 0) so (0,0) is bottom-left
        for y in range(self.rows - 1, -1, -1):
            row_str = ""
            for x in range(self.cols):
                if (x, y) == self.current_state:
                    row_str += " ♙ " # Pawn (Agent)
                elif (x, y) == self.target_state:
                    row_str += " 🎯 " # Target
                elif (x, y) in self.walls:
                    row_str += " ♜ " # Rook (Wall)
                else:
                    row_str += " . " # Empty space
            print(row_str)
        print("-" * 27)
        time.sleep(0.15) # Pause so you can actually see the movement

# ==========================================
# 2. Q-LEARNING ALGORITHM & ANIMATION LOOP
# ==========================================
env = VisualGridWorld()
q_table = np.zeros((env.cols, env.rows, len(env.actions)))
action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0       
epsilon_decay = 0.99
epochs = 500

for epoch in range(epochs):
    state = env.reset()
    done = False
    step_count = 0
    
    # We only want to animate the very first epoch and the very last epoch
    animate_this_epoch = (epoch == 0 or epoch == epochs - 1)
    
    if animate_this_epoch:
        if epoch == 0:
            print("Starting Epoch 1: Complete random guessing (Exploration)...")
            time.sleep(2)
        else:
            print("Starting Epoch 500: Mastered optimal path (Exploitation)...")
            time.sleep(2)

    while not done:
        x, y = state
        
        # Epsilon-Greedy choice
        if random.uniform(0, 1) < epsilon:
            action = random.choice(env.actions)
        else:
            action = np.argmax(q_table[x, y])
            
        next_state, reward, done = env.step(action)
        
        # Visualize if it's the first or last epoch
        if animate_this_epoch:
            env.render(step_count, action_names[action])
        
        # Bellman Equation update
        next_x, next_y = next_state
        old_value = q_table[x, y, action]
        next_max = np.max(q_table[next_x, next_y])
        q_table[x, y, action] = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        
        state = next_state
        step_count += 1
        
        # Failsafe for Epoch 1 wandering infinitely
        if step_count > 100 and epoch == 0:
            break
            
    epsilon = max(0.1, epsilon * epsilon_decay)

print("\nTraining Complete! The Pawn successfully learned the maze.")