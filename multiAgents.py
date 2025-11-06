# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with base game score
        score = successorGameState.getScore()
        
        # Find closest dangerous ghost to determine danger level
        closest_dangerous_ghost = float('inf') # we will always replace this
        
        # Process all ghosts
        for ghostIndex, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)
            
            if newScaredTimes[ghostIndex] > 0:
                # Ghost is scared - reward being close (can eat it for points)
                if distanceToGhost <= 1:
                    score += 1000  # Can eat the ghost
                elif distanceToGhost <= 2:
                    score += 500  # Very close, good opportunity
            else:
                # track closest dangerous ghost
                if distanceToGhost < closest_dangerous_ghost:
                    closest_dangerous_ghost = distanceToGhost
                
                # penalize based on distance to ghost
                if distanceToGhost <= 1:
                    score -= 10000  # EXTREMELY dangerous run away NOW!
                elif distanceToGhost <= 2:
                    score -= 5000   # Very dangerous zone
  
        # Penalize STOP action
        if action == Directions.STOP:
            score -= 5000  # penalize stopping
        
        currentDirection = currentGameState.getPacmanState().getDirection()
        if action == Directions.REVERSE[currentDirection]:
            score -= 2000  # discourage going backwards

        return score
        
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState: GameState, agentIndex, depth):
        
        if depth == 0 or gameState.isWin() or gameState.isLose(): # terminal state
            return self.evaluationFunction(gameState)
        
        maximizingPlayerIndex = 0 # max player is index 0
        legalActions = gameState.getLegalActions(agentIndex)

        # navigate all agents
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0 # it's pacman's turn
            depth -= 1

        if agentIndex == maximizingPlayerIndex:
            maxEval = float('-inf')
            for action in legalActions: 
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.minimax(successorState, nextAgent, depth)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            minEval = float('inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.minimax(successorState, nextAgent, depth)
                minEval = min(minEval, eval)
            return minEval


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        pacmanIndex = 0
        legalActions = gameState.getLegalActions()

        bestAction = None
        bestValue = float("-inf")

        for action in legalActions: 
            successor = gameState.generateSuccessor(pacmanIndex, action)
            value = self.minimax(successor, 1, self.depth)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction if bestAction else legalActions[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, gameState: GameState, agentIndex, depth, alpha, beta):
        
        if depth == 0 or gameState.isWin() or gameState.isLose(): # terminal state
            return self.evaluationFunction(gameState)
        
        maximizingPlayerIndex = 0 # max player is index 0
        legalActions = gameState.getLegalActions(agentIndex)
        
        nextDepth = depth
        # navigate all agents
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0 # it's pacman's turn
            nextDepth = depth - 1

        if agentIndex == maximizingPlayerIndex:
            v = float('-inf')
            for action in legalActions: 
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.alphaBeta(successorState, nextAgent, nextDepth, alpha, beta)
                
                if eval > beta: 
                    return eval 
                
                v = max(v, eval)
                alpha = max(alpha, v)

            return v
        
        else:
            v = float('inf')
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.alphaBeta(successorState, nextAgent, nextDepth, alpha, beta)

                if eval < alpha: 
                    return eval 
                
                v = min(v, eval)
                beta = min(beta, v)

            return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        pacmanIndex = 0
        legalActions = gameState.getLegalActions()

        bestAction = None
        bestValue = float("-inf")

        alpha = float('-inf')
        beta = float('inf')   
        

        for action in legalActions: 
            successor = gameState.generateSuccessor(pacmanIndex, action)
            value = self.alphaBeta(successor, 1, self.depth, alpha, beta)

            if value > bestValue:
                bestValue = value
                bestAction = action
                alpha = max(alpha, value)

        return bestAction if bestAction else legalActions[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState: GameState, agentIndex, depth):
        
        if depth == 0 or gameState.isWin() or gameState.isLose(): # terminal state
            return self.evaluationFunction(gameState)
        
        maximizingPlayerIndex = 0 # max player is index 0
        legalActions = gameState.getLegalActions(agentIndex)

        # navigate all agents
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0 # it's pacman's turn
            depth -= 1

        if agentIndex == maximizingPlayerIndex:
            maxEval = float('-inf')
            for action in legalActions: 
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.expectimax(successorState, nextAgent, depth)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            # Chance node - calculate expected value (average) of all legal actions
            expectedValue = 0
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.expectimax(successorState, nextAgent, depth)
                expectedValue += eval
            return expectedValue / len(legalActions) if len(legalActions) > 0 else 0

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        pacmanIndex = 0
        legalActions = gameState.getLegalActions()

        bestAction = None
        bestValue = float("-inf")

        for action in legalActions: 
            successor = gameState.generateSuccessor(pacmanIndex, action)
            value = self.expectimax(successor, 1, self.depth)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction if bestAction else legalActions[0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    return currentGameState.getScore()



# Abbreviation
better = betterEvaluationFunction
