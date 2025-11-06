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
                # Ghost is scared, reward being close (can eat it for points)
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
                    score -= 10000  # Very dangerous
                elif distanceToGhost <= 2:
                    score -= 5000   # dangerous zone
  
        # Penalize stop action
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

        # manage all agents
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
        # manage all agents
        nextAgent = agentIndex + 1
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0 # it's pacman's turn
            nextDepth = depth - 1

        if agentIndex == maximizingPlayerIndex:
            v = float('-inf')
            for action in legalActions: 
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval = self.alphaBeta(successorState, nextAgent, nextDepth, alpha, beta)
                
                if eval > beta: # prune based on > for the auto grader
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
            # Chance node calculate expected value (average) of all legal actions
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

    DESCRIPTION: 

DESCRIPTION: 

    1. I get the manhattan distance to each ghost, then get the absolute difference between pacman and the closest ghost. 

    2. If pacman is farther than a distance of 3 from the closest ghost and the ghost is dangerous (not scared), the distance gets added to the state's score. This incentivises pacman to be farther away from dangerous ghosts.

    3. If pacman is closer than distance 2 to a dangerous ghost, I heavily penalize the score to avoid death

    4. Next we check if the closest ghost is scared. If it is, and its distance from pacman is less than 5, we multiple the distance by 3 and add it to the state's score. This is to incentivise pacman to eat scared ghosts that are close by.

    5. Then, we penalize the state's score by the distance pacman is to the closest piece of food. This gives pacman the incentive to move towards the food.

    6. We also penalize the distance to the closest capsule (with less weight than food) to encourage capsule collection

    7. Finally, we increase the score for pacman eating food and we punish the score for any remaining power capsules

    """

    currentGameScore = currentGameState.getScore()
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate Manhattan distance to each ghost with their indices
    ghostDistances = [(manhattanDistance(pacmanPosition, ghostPos), idx) for idx, ghostPos in enumerate(ghostPositions)]

    # Get the closest ghost (distance and index)
    if ghostDistances:
        closestGhostDistance, closestGhostIndex = min(ghostDistances, key=lambda x: x[0]) # only compare the first value in the tuple, that is the Manhattan distance to the ghost. 
    else:
        closestGhostDistance = None
        closestGhostIndex = None

    stateScore = currentGameScore
    if (closestGhostDistance is not None and closestGhostDistance > 3 and 
        closestGhostIndex is not None and scaredTimers[closestGhostIndex] == 0):  # ghost is not scared
        stateScore += closestGhostDistance
    
    
    if closestGhostDistance is not None and closestGhostIndex is not None and closestGhostDistance < 2:
        if scaredTimers[closestGhostIndex] == 0:  # Ghost is NOT scared (dangerous)
            stateScore -= currentGameScore 

    # Check if the closest ghost is scared and within distance < 5
    if closestGhostDistance is not None and closestGhostIndex is not None:
        if scaredTimers[closestGhostIndex] > 0 and closestGhostDistance < 5:
            stateScore += closestGhostDistance * 3

    # Find distance to closest food and subtract it from stateScore
    foodList = currentGameState.getFood().asList()
    if foodList:
        closestFoodDistance = min([manhattanDistance(pacmanPosition, food) for food in foodList])
        stateScore -= closestFoodDistance
    
    # Get the distance to the closest capsule and pacman, and subtract it from the state's score
    capsuleList = currentGameState.getCapsules()
    if capsuleList:
        closestCapsuleDistance = min([manhattanDistance(pacmanPosition, cap) for cap in capsuleList])
        stateScore -= closestCapsuleDistance / 2

    
    # reward eating food
    remainingFood = len(foodList)
    stateScore += (100 - remainingFood) 

    # punish remaining capsules 
    remainingCapsules = len(capsuleList)
    stateScore -= (100 + remainingCapsules * 2) if remainingCapsules > 0 else 0

    return stateScore

# Abbreviation
better = betterEvaluationFunction
