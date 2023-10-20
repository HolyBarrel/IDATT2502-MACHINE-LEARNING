import pygame
from pygame.locals import *
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import math

class GridWorldEnvironment:
    def __init__(self):
        self.gridSize = 40
        self.resolution = [520, 520]
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
        # print(self.grid)

    def reset(self):
        self.startX = np.random.randint(0, self.gridsInRow)
        self.startY = np.random.randint(0, self.gridsInColumn)
        self.grid = [[0 for _ in range(self.gridsInRow)] for _ in range(self.gridsInColumn)]
        self.grid[self.startY][self.startX] = 1
        return self.startX, self.startY


    def step(self, action):
        if action == 'up':
            newY = max(0, self.startY - 1)
            newX = self.startX
        elif action == 'down':
            newY = min(self.gridsInColumn - 1, self.startY + 1)
            newX = self.startX
        elif action == 'left':
            newX = max(0, self.startX - 1)
            newY = self.startY
        elif action == 'right':
            newX = min(self.gridsInRow - 1, self.startX + 1)
            newY = self.startY

        self.grid[self.startY][self.startX] = 0
        self.startX, self.startY = newX, newY
        self.grid[newY][newX] = 1

        distanceToGoal  = self.manhattanDistance(self.startX, self.startY, self.goalX, self.goalY)
          
        if distanceToGoal  == 0:
            reward = 1  
            done = True
        else:
            reward = 1 - distanceToGoal ** 0.5
            #reward = math.exp(-distanceToGoal)
            done = False

        nextState = (newX, newY)
        return nextState, reward, done, {}
    
    def manhattanDistance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def plotGrid(self, currentState):
        screen = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("GridWorld")
        screen.fill((255, 255, 255))

        gridColor = (245, 145, 100)
        gridSize = self.gridSize
        screenWidth, screenHeight = screen.get_size()
        playerColor = (0, 120, 120)
        appleColor = (255, 20, 0)

        playerX, playerY = currentState
        pygame.draw.rect(screen, playerColor, (playerX * gridSize, playerY * gridSize, gridSize, gridSize))

        appleX = self.goalX * gridSize
        appleY = self.goalY * gridSize
        pygame.draw.rect(screen, appleColor, (appleX, appleY, gridSize, gridSize))

        # Draws vertical lines
        for x in range(0, screenWidth, gridSize):
            pygame.draw.line(screen, gridColor, (x, 0), (x, screenHeight))

        # Draws horizontal lines
        for y in range(0, screenHeight, gridSize):
            pygame.draw.line(screen, gridColor, (0, y), (screenWidth, y))

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

    def initializeQTable(self):
        for x in range(self.environment.gridsInRow):
            for y in range(self.environment.gridsInColumn):
                self.qTable[(x, y)] = {action: 0 for action in self.environment.actions}

    def chooseAction(self, state, episodeIndex):
        p = np.random.uniform(0, 1)

        if episodeIndex > 2000:
            self.eps *= 0.999        

        if p < self.eps:
            return np.random.choice(self.environment.actions)
        else:
            return max(self.qTable[state], key=self.qTable[state].get)

    def simulateLearning(self):
        self.initializeQTable()
        successCount = 0

        for episodeIndex in range(self.maxEpisodes):
            state = self.environment.reset()
            done = False
            episodeRewards = 0

            while not done:
                action = self.chooseAction(state, episodeIndex)
                nextState, reward, done, _ = self.environment.step(action)
                episodeRewards += reward

                oldValue = self.qTable[state][action]
                nextMax = max(self.qTable[nextState].values())
                newValue = (1 - self.a) * oldValue + self.a * (reward + self.g * nextMax)
                self.qTable[state][action] = newValue

                state = nextState

            self.sumOfRewards.append(episodeRewards)

            if episodeRewards == 1:
                successCount += 1

            if successCount >= 5000:
                print("Reached the goal 5000 times. Stopping the simulation.")
                break

    def visualizeLastRun(self):
        state = self.environment.reset()
        done = False
        while not done:
            self.environment.plotGrid(state)
            action = self.chooseAction(state, 1)
            nextState, reward, done, _ = self.environment.step(action)
            state = nextState
            sleep(1)

    def plotRewards(self):
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
    q_learner.plotRewards()
    
    q_learner.visualizeLastRun()
