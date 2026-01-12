import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:

    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0    # Randomness
        self.gamma = 0.9      # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]

        pointLeft = Point(head.x - 20, head.y)
        pointRight = Point(head.x + 20, head.y)
        pointUp = Point(head.x, head.y - 20)
        pointDown = Point(head.x, head.y + 20)

        directionLeft = game.direction == Direction.LEFT
        directionRight = game.direction == Direction.RIGHT
        directionUp = game.direction == Direction.UP
        directionDown = game.direction == Direction.DOWN

        state = [

            # Danger Straight
            (directionLeft and game.is_collision(pointLeft)) or
            (directionRight and game.is_collision(pointRight)) or
            (directionUp and game.is_collision(pointUp)) or
            (directionDown and game.is_collision(pointDown)),

            # Danger Right
            (directionUp and game.is_collision(pointRight)) or
            (directionDown and game.is_collision(pointLeft)) or
            (directionLeft and game.is_collision(pointUp)) or
            (directionRight and game.is_collision(pointDown)),

            # Danger Left
            (directionDown and game.is_collision(pointRight)) or
            (directionUp and game.is_collision(pointLeft)) or
            (directionRight and game.is_collision(pointUp)) or
            (directionLeft and game.is_collision(pointDown)),

            # Move Directions
            directionLeft,
            directionRight,
            directionUp,
            directionDown,

            # Food Location
            game.food.x < game.head.x, # Food Left
            game.food.x > game.head.x, # Food Right
            game.food.y < game.head.y, # Food Up
            game.food.y > game.head.y, # Food Down
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory
        
        states, actions, rewards, nextStates, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, nextStates, dones)
            

    def train_short_memory(self, state, action, reward, nextState, done):
        self.trainer.train_step(state, action, reward, nextState, done)


    def get_action(self, state):
        self.epsilon = 80 - self.numberOfGames
        finalMove = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[int(move)] = 1

        return finalMove


        

def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        stateOld = agent.get_state(game)
        finalMove = agent.get_action(stateOld)
        reward, done, score = game.play_step(finalMove)
        stateNew = agent.get_state(game)
        agent.train_short_memory(stateOld, finalMove, reward, stateNew, done)
        agent.remember(stateOld, finalMove, reward, stateNew, done)
        if done:
            game.reset()
            agent.numberOfGames += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
                
            print('Game', agent.numberOfGames, 'Score', score, 'Record', record)
            plotScores.append(score)
            totalScore += score
            meanScore = totalScore / agent.numberOfGames
            plotMeanScores.append(meanScore)
            plot(plotScores, plotMeanScores)





if __name__ == "__main__":
    train()


    