import pygame
from pygame.locals import *
from time import sleep
import numpy as np
import matplotlib.pyplot as plt


class GridWorldEnvironment:
    def __init__(self):
        self.gridSize = 40
        self.resolution = [600, 520]
        self.gridsInRow = self.resolution[0] // self.gridSize
        self.gridsInColumn = self.resolution[1] // self.gridSize
        self.grid = [[0 for _ in range(self.gridsInRow)] for _ in range(self.gridsInColumn)]
        self.actions = ['up', 'down', 'left', 'right']
        self.startX = np.random.randint(0, self.gridsInRow)
        self.startY = np.random.randint(0, self.gridsInColumn)
        self.goalX = np.random.randint(0, self.gridsInRow)
        self.goalY = np.random.randint(0, self.gridsInColumn)
        self.grid[self.goalY][self.goalX] = 2
        self.grid[self.startY][self.startX] = 1
        print(self.startX, self.startY)
        print(self.goalX, self.goalY)
        print(self.grid)

    def reset(self):
        self.startX = np.random.randint(0, self.gridsInRow)
        self.startY = np.random.randint(0, self.gridsInColumn)
        self.grid = [[0 for _ in range(self.gridsInRow)] for _ in range(self.gridsInColumn)]
        self.grid[self.startY][self.startX] = 1
        return self.startX, self.startY


    def step(self, action):
        if action == 'up':
            new_y = max(0, self.startY - 1)
            new_x = self.startX
        elif action == 'down':
            new_y = min(self.gridsInColumn - 1, self.startY + 1)
            new_x = self.startX
        elif action == 'left':
            new_x = max(0, self.startX - 1)
            new_y = self.startY
        elif action == 'right':
            new_x = min(self.gridsInRow - 1, self.startX + 1)
            new_y = self.startY

        self.grid[self.startY][self.startX] = 0
        self.startX, self.startY = new_x, new_y
        self.grid[new_y][new_x] = 1 

        distance_to_goal = self.manhattan_distance(self.startX, self.startY, self.goalX, self.goalY)
          
        if distance_to_goal == 0:
            reward = 1  
            done = True
        else:
            reward = 1 - distance_to_goal ** 0.5
            done = False

        next_state = (new_x, new_y)
        return next_state, reward, done, {}
    
    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def plotGrid(self, current_state):
        screen = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("GridWorld")
        screen.fill((255, 255, 255))

        grid_color = (245, 145, 100) 
        grid_size = self.gridSize
        screen_width, screen_height = screen.get_size()
        player_color = (0, 120, 120)
        apple_color = (255, 20, 0)

        player_x, player_y = current_state
        pygame.draw.rect(screen, player_color, (player_x * grid_size, player_y * grid_size, grid_size, grid_size))

        apple_x = self.goalX * grid_size
        apple_y = self.goalY * grid_size
        pygame.draw.rect(screen, apple_color, (apple_x, apple_y, grid_size, grid_size))

        # Draws vertical lines
        for x in range(0, screen_width, grid_size):
            pygame.draw.line(screen, grid_color, (x, 0), (x, screen_height))

        # Draws horizontal lines
        for y in range(0, screen_height, grid_size):
            pygame.draw.line(screen, grid_color, (0, y), (screen_width, y))

        pygame.display.flip()

class QLearner:
    def __init__(self, environment, a, eps, g, maxEpisodes):
        self.environment = environment
        self.a = a
        self.eps = eps
        self.g = g
        self.maxEpisodes = maxEpisodes
        self.numberOfActions = len(self.environment.actions)
        self.qTable = {}
        self.sumOfRewards = []

    def initialize_q_table(self):
        for x in range(self.environment.gridsInRow):
            for y in range(self.environment.gridsInColumn):
                self.qTable[(x, y)] = {action: 0 for action in self.environment.actions}

    def chooseAction(self, state, episodeIndex):
        p = np.random.uniform(0, 1)
        if episodeIndex > 20000:
            self.eps *= 0.999        

        if p < self.eps:
            return np.random.choice(self.environment.actions)
        else:
            return max(self.qTable[state], key=self.qTable[state].get)

    def simulateLearning(self):
        self.initialize_q_table()
        success_count = 0 


        for episodeIndex in range(self.maxEpisodes):
            state = self.environment.reset()
            done = False
            episodeRewards = 0

            while not done:

                action = self.chooseAction(state, episodeIndex)
                next_state, reward, done, _ = self.environment.step(action)
                episodeRewards += reward

                # Updates Q-value
                old_value = self.qTable[state][action]
                next_max = max(self.qTable[next_state].values())
                new_value = (1 - self.a) * old_value + self.a * (reward + self.g * next_max)
                self.qTable[state][action] = new_value

                state = next_state

            self.sumOfRewards.append(episodeRewards)
            #print(f'Episode: {episodeIndex + 1}, Total Reward: {episodeRewards}')

            if episodeRewards == 1:  
                success_count += 1

            if success_count >= 5000:
                print("Reached the goal 5000 times. Stopping the simulation.")
                break

    def visualize_last_run(self):
        state = self.environment.reset()
        done = False
        while not done:
            self.environment.plotGrid(state)  # Updates Pygame window based on the current state
            action = self.chooseAction(state, 1)
            next_state, reward, done, _ = self.environment.step(action)
            state = next_state
            sleep(1) 

    def plot_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.sumOfRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.show()


if __name__ == "__main__":
    env = GridWorldEnvironment()
    q_learner = QLearner(env, a=0.1, eps=0.1, g=0.99, maxEpisodes=30000)
    q_learner.simulateLearning()
    q_learner.plot_rewards()
    
    q_learner.visualize_last_run()
