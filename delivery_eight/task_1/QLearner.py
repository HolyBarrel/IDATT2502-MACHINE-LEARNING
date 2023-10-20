import numpy as np
import matplotlib.pyplot as plt

class QLearner:
    def __init__(self, environment, a, eps, g, maxEpisodes, numberOfBinsArray, lowerBoundsArray, upperBoundsArray):
        self.environment = environment
        self.a = a
        self.eps = eps
        self.g = g
        self.maxEpisodes = maxEpisodes
        self.numberOfBinsArray = numberOfBinsArray
        self.lowerBoundsArray = lowerBoundsArray
        self.upperBoundsArray = upperBoundsArray

        self.sumOfRewards = []
        self.numberOfActions = self.environment.action_space.n

        self.qTable = np.random.uniform(low=0, high=1, size=(
            self.numberOfBinsArray[0], self.numberOfBinsArray[1], self.numberOfBinsArray[2], self.numberOfBinsArray[3], self.numberOfActions
            ))


    def discretizedIndex(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        postionBin = np.linspace(self.lowerBoundsArray[0], self.upperBoundsArray[0], self.numberOfBinsArray[0])
        velocityBin = np.linspace(self.lowerBoundsArray[1], self.upperBoundsArray[1], self.numberOfBinsArray[1])
        angleBin = np.linspace(self.lowerBoundsArray[2], self.upperBoundsArray[2], self.numberOfBinsArray[2])
        aVelocityBin = np.linspace(self.lowerBoundsArray[3], self.upperBoundsArray[3], self.numberOfBinsArray[3])

        positionIndex = np.digitize(position, postionBin - 1)
        velocityIndex = np.digitize(velocity, velocityBin - 1)
        angleIndex = np.digitize(angle, angleBin - 1)
        aVelocityIndex = np.digitize(angularVelocity, aVelocityBin - 1)

        # Clips the indices to ensure they are within the valid range
        positionIndex = np.clip(positionIndex, 0, self.numberOfBinsArray[0] - 1)
        velocityIndex = np.clip(velocityIndex, 0, self.numberOfBinsArray[1] - 1)
        angleIndex = np.clip(angleIndex, 0, self.numberOfBinsArray[2] - 1)
        aVelocityIndex = np.clip(aVelocityIndex, 0, self.numberOfBinsArray[3] - 1)

        return tuple([positionIndex, velocityIndex, angleIndex, aVelocityIndex])

    def chooseAction(self, state, episodeIndex):

        # Random sampling for the first 500 episodes
        if episodeIndex < 500:
            return self.environment.action_space.sample()
        
        p = np.random.uniform(0, 1)

        # Decay the epsilon value for extreme episodes indices
        if episodeIndex > 5000:
            self.eps *= 0.999        

        if p < self.eps:
            return self.environment.action_space.sample()
        else:
            # Chooses the action with the highest Q value - the essence of the greedy policy of the Q-Learning algorithm
            return np.random.choice(np.where(self.qTable[self.discretizedIndex(state)] == self.qTable[self.discretizedIndex(state)].max())[0])
        
    def simulateLearning(self, selfLog = False):
        for episodeIndex in range(self.maxEpisodes):
            stateTuple = self.environment.reset()
            stateBefore = stateTuple[0] 
            terminate = False
            episodeRewards = []

            while not terminate:
                stateBeforeIndex = self.discretizedIndex(stateBefore)
                action = self.chooseAction(stateBefore, episodeIndex)
                (stateAfter, reward, terminate, _, _) = self.environment.step(action)
                episodeRewards.append(reward)

                stateAfter = list(stateAfter)

                stateAfterIndex = self.discretizedIndex(stateAfter)

                QAfter = np.max(self.qTable[stateAfterIndex])

                if not terminate:
                    error = reward + self.g * QAfter - self.qTable[stateBeforeIndex + (action, )]
                    self.qTable[stateBeforeIndex + (action, )] += self.a * error
         
                else:
                    error = reward - self.qTable[stateBeforeIndex + (action, )]
                    self.qTable[stateBeforeIndex + (action, )] += self.a * error

                stateBefore = stateAfter
            if selfLog and episodeIndex % 100 == 0:
                print(f'E: {episodeIndex}, R: {sum(episodeRewards)}')
            self.sumOfRewards.append(sum(episodeRewards))


    def plotRewards(self,init_eps):
        plt.figure(figsize=(12, 6))
        plt.plot(self.sumOfRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Episode Overview')

        plt.annotate(f'a: {self.a}\ninit eps {init_eps}\neps: {self.eps}\ng: {self.g}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top')
        
        plt.show()
        
    def simulateOptimalLearnedStrategy(self, selfRender = False, selfLog = False):
        import gym
        import time

        if selfRender:
            firstEnvironment = gym.make('CartPole-v1', render_mode='human')
        else:
            firstEnvironment = gym.make('CartPole-v1')

        (currentState,_)= firstEnvironment.reset()

        if selfRender:
            firstEnvironment.render()

        timeSlots = 1000

        collectedRewards = []

        for timeIndex in range(timeSlots):
            if selfLog:
                print(f'Time Index: {timeIndex}')
            actionInStateBefore = np.random.choice(np.where(self.qTable[self.discretizedIndex(currentState)] == self.qTable[self.discretizedIndex(currentState)].max())[0])
            currentState, reward, terminate, _, _ = firstEnvironment.step(actionInStateBefore)
            collectedRewards.append(reward)
            time.sleep(0.1)
            if(terminate):
                time.sleep(2)
                break
        return sum(collectedRewards),collectedRewards, firstEnvironment

    def simulateRandomControlStrategy(self, selfRender = False, selfLog = False):
        import gym
        import time

        if selfRender:
            secondEnvironment = gym.make('CartPole-v1', render_mode='human')
        else:
            secondEnvironment = gym.make('CartPole-v1')
        (currentState,_)= secondEnvironment.reset()

        if selfRender:
            secondEnvironment.render()
        timeSlots = 1000

        episodeCap = 100

        collectedRewards = []

        for episodeIdex in range(episodeCap):
            rewardsPerEpisode = []
            stateZero = secondEnvironment.reset()
            if selfLog:
                print(f'Episode Index: {episodeIdex}')

            for timeIndex in range(timeSlots):
                if selfLog:
                    print(f'Time Index: {timeIndex}')

                actionInStateBefore = secondEnvironment.action_space.sample()
                surveyState, reward, terminate, _, _ = secondEnvironment.step(actionInStateBefore)
                rewardsPerEpisode.append(reward)
                if(terminate):
                    break
            collectedRewards.append(sum(rewardsPerEpisode))


        return sum(collectedRewards)/len(collectedRewards),collectedRewards, secondEnvironment


import gym

if __name__ == "__main__":
    # Creates a CartPole environment
    environment = gym.make('CartPole-v1')#, render_mode='human')

    # Defines parameters for Q-Learning
    a = 0.5 # learning rate
    eps = 0.2  # epsilon for epsilon-greedy policy
    g = 0.99  # discount factor
    maxEpisodes = 12000  # number of episodes

    upperBounds = environment.observation_space.high
    lowerBounds = environment.observation_space.low

    numberOfBinsArray = [30, 30, 30, 30]  # Discretization bins for each state dimension
    upperBounds[1] = 3 # maximum cart velocity
    upperBounds[3] = 10 # maximum pole angle velocity
    lowerBounds[1] = -3 # minimum cart velocity
    lowerBounds[3] = -10 # minimum pole angle velocity

    # Instantiates QLearner and simulate learning
    qLearner = QLearner(environment, a, eps, g, maxEpisodes, numberOfBinsArray, lowerBounds, upperBounds)

    qLearner.simulateLearning(selfLog=True)
    qLearner.plotRewards(eps)

    # Simulates the optimal learned strategy
    totalCollectedRewards, collectedRewards, env = qLearner.simulateOptimalLearnedStrategy()

    # Simulates random control strategy
    avgCollectedRewards, collectedRewardsRandom, envRandom = qLearner.simulateRandomControlStrategy()
    print(f'Average rewards collected with optimal strategy: {totalCollectedRewards}')
    print(f'Average rewards collected with random control: {avgCollectedRewards}')



                

        