
import torch
import sys
sys.path.append('..')
import numpy as np
import copy
import math
import os

# SUM OF INTEGERS FROM 1 TO N
def sumIntegers(n):
        return int((n * ((n + 1)/2.)))

# FROM AN INTEGER ACTION CODE, CALCULATE THE PILE SIZE AND REMOVAL NUMBER
# CODING: FOR A SIZE=1 PILE THERE IS EXACTLY ONE POSSIBLE MOVE.
#         FOR A SIZE=2 PILE THERE ARE EXACTLY TWO POSSIBLE MOVES, ETC.
def actionDecode(action):
    action = action + 1
    quadraticSolution = (-1 + math.sqrt(1 + (8 * action)))/2.
    pileSize = math.ceil(quadraticSolution)
    removeSize = int(action - ((pileSize) * ((pileSize-1)/2.)))

    return pileSize, removeSize

# CALCULATE THE NIMBER ASSOCIATED WITH A STATE
def stateToNimber(pileList, shortNotation=True):
    if shortNotation:
        pileList = [i % 2 for i in pileList]
        longList = []
        for index, count in enumerate(pileList):
            longList.extend([index + 1 for i in range(count)])
        return stateToNimber(longList, False)
    else:  
        if len(pileList) == 0:
            return 0
        elif len(pileList) == 1:
            return pileList[0]
        elif len(pileList) == 2:
            a = format(pileList[0], 'b')
            b = format(pileList[1], 'b')
            
            if len(a) > len(b):
                b = b.zfill(len(a))
            else:
                a = a.zfill(len(b))
                
            c = ''
            for i in range(len(a)):
                c = c + str((int(b[i]) + int(a[i])) % 2)
            return int(c,2)
                
        else:
            return stateToNimber([pileList[0], stateToNimber(pileList[1:], False)], False)

class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        self.maxPileSize = 10
        self.maxNumPile = 3
        self.initialState = None
        self.numAction = sumIntegers(self.maxPileSize)

        ### Game
        self.observation_shape = self.maxPileSize  # Dimensions of the game observation
        self.action_space = [i for i in range(self.numAction)]  # Fixed list of all possible actions
        self.players = [-1,1]  # List of players


        ### Self-Play
        self.num_actors = 10  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 80  # Number of futur moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.self_play_delay = 0 # Number of seconds to wait after each played game to adjust the self play / training ratio to avoid over/underfitting

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


        ### Network
        self.encoding_size = 32
        self.hidden_size = 64


        ### Training
        self.results_path = "./pretrained"  # Path to store the model weights
        self.training_steps = 5000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.window_size = 1000  # Number of self-play games to keep in the replay buffer
        self.td_steps = 30  # Number of steps in the futur to take into account for calculating the target value
        self.training_delay = 0 # Number of seconds to wait after each training to adjust the self play / training ratio to avoid over/underfitting
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.008  # Initial learning rate
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000


        ### Test
        self.test_episodes = 2  # Number of game played to evaluate the network


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game:
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.config = MuZeroConfig()
        self.maxPileSize = self.config.maxPileSize
        self.maxNumPile = self.config.maxNumPile
        self.initialState = self.config.initialState
        self.numAction = self.config.numAction
        self.state = self.config.initialState
        self.currentPlayer = 1
        self.board = self.reset()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        observation = self.getNextState(self.board, action)
        reward, done = self.getGameEnded(observation, self.currentPlayer)

        self.board = observation
        self.currentPlayer = -self.currentPlayer
        #observation, reward, done, _ = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.currentPlayer

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        #return self.env.reset()
        initialBoard = self.getInitBoard()
        self.board = initialBoard
        return initialBoard

    def close(self):
        """
        Properly close the game.
        """
        #self.env.close()
        pass

    def render(self):
        """
        Display the game observation.
        """
        #self.env.render()
        #input("Press enter to take a step ")
        return str(list(board))


    #####  BORROWED AND MODIFIED FROM ALPHAZERO NIM  #####
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        if self.initialState is not None:
            return self.initialState
        else:
            return np.random.randint(low=0, high=self.maxNumPile, size=(self.maxPileSize,))

    def getNextState(self, board, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pileSize, removeSize = actionDecode(action)

        if board[pileSize - 1] == 0:
            return np.array([-1])
        else:
            nextBoard = copy.copy(board)
            nextBoard[pileSize - 1] = nextBoard[pileSize - 1] - 1
            if removeSize < pileSize:
                newPileSize = pileSize - removeSize
                nextBoard[newPileSize - 1] = nextBoard[newPileSize - 1] + 1
            
            return nextBoard

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        
        if board[0]==-1:
            # we made an illegal move
            return -1, True
        else:
            if sum(board) > 0:
                return 0, False
            else:
                return 1, True





class nimGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, config):
        self.maxPileSize = config['maxPileSize']
        self.maxNumPile = config['maxNumPile']
        self.initialState = config['initialState']
        #self.randomInitial = config['randomInitial']
        self.numAction = sumIntegers(self.maxPileSize)
        self.state = copy.deepcopy(self.initialState)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        if self.initialState is not None:
            return self.initialState
        else:
            return np.random.randint(low=0, high=self.maxNumPile, size=(self.maxPileSize,))

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (1, self.maxPileSize)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.numAction

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pileSize, removeSize = actionDecode(action)

        # if we don't use getValidMoves
        if board[pileSize - 1] == 0:
            return np.array([-1]), -player
        else:
            nextBoard = copy.copy(board)
            nextBoard[pileSize - 1] = nextBoard[pileSize - 1] - 1
            if removeSize < pileSize:
                newPileSize = pileSize - removeSize
                nextBoard[newPileSize - 1] = nextBoard[newPileSize - 1] + 1
            
            return nextBoard, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """


        '''
        validMoves = np.zeros(self.getActionSize())
        for action in range(self.getActionSize()):
            pileSize, _ = actionDecode(action)
            if board[pileSize - 1] > 0:
                validMoves[action] = 1
        '''

        # all moves are apriori valid.  let the algorithm find the invalid moves.
        validMoves = np.ones(self.getActionSize())
        return validMoves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        
        if board[0]==-1:
            # we made an illegal move
            return 1
        else:
            if sum(board) > 0:
                return 0
            else:
                return -1

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board,pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(list(board))


    @staticmethod
    def display(board):
        print("Board: " + str(board))
        print("Nimber: " + str(stateToNimber(board)))
        print(" ")

    @staticmethod
    def postAction(action):
        print("Action: " + str(actionDecode(action)))


