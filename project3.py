import numpy as np
import random
import math

s0 = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]

def createBoard(state):
    oneDBoard = []
    for i in range(1, 10):
        oneDBoard.append(state[i])
    oneDBoard = np.array(oneDBoard)
    return oneDBoard.reshape(3, 3)

def flattenBoard(turn, board):
    twoDBoard = np.array(board)
    oneDBoard = list(twoDBoard.flatten())
    oneDBoard.insert(0, turn)
    return oneDBoard

def toMove(state):
    return state[0]

def actions(state):
    turn = toMove(state)
    board = createBoard(state)
    turnCoordinates = []
    nullCoordinates = []
    opponentCoordinates = []
    actions = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == turn:
                turnCoordinates.append((j, i))
            elif board[i][j] == 0:
                nullCoordinates.append((j, i))
            else:
                opponentCoordinates.append((j, i))
    for coordinate in turnCoordinates:
        if turn == -1 and (coordinate[1] == 0 or coordinate[1] == 1) and (coordinate[0], coordinate[1]+1) in nullCoordinates and "down" not in actions:
            actions.append("down")
        if turn == 1 and (coordinate[1] == 1 or coordinate[1] == len(board[0]) - 1) and (coordinate[0], coordinate[1]-1) in nullCoordinates and "up" not in actions:
            actions.append("up")
        if turn == -1 and (coordinate[1] == 0 or coordinate[1] == 1) and (coordinate[0]+1, coordinate[1]+1) in opponentCoordinates and "down-right" not in actions:
            actions.append("down-right")
        if turn == -1 and (coordinate[1] == 0 or coordinate[1] == 1) and (coordinate[0]-1, coordinate[1]+1) in opponentCoordinates and "down-left" not in actions:
            actions.append("down-left")
        if turn == 1 and (coordinate[1] == 1 or coordinate[1] == len(board[0]) - 1) and (coordinate[0]+1, coordinate[1]-1) in opponentCoordinates and "up-right" not in actions:
            actions.append("up-right")
        if turn == 1 and (coordinate[1] == 1 or coordinate[1] == len(board[0]) - 1) and (coordinate[0]-1, coordinate[1]+1) in opponentCoordinates and "up-left" not in actions:
            actions.append("up-left")
    return actions

def result(state, action):
    board = createBoard(state)
    nextBoard = [r[:] for r in board]
    if action == "down":
        for i in len(nextBoard):
            for j in len(nextBoard[i]):
                if nextBoard[i][j] == -1 and nextBoard[i+1][j] == 0:
                    tmp = nextBoard[i][j]
                    nextBoard[i+1][j] = tmp
                    nextBoard[i][j] = 0
                    return flattenBoard(nextBoard[i+1][j], nextBoard)
    elif action == "up":
        for i in range(len(nextBoard)):
            for j in range(len(nextBoard[i])):
                if nextBoard[i][j] == 1 and nextBoard[i-1][j] == 0:
                    tmp = nextBoard[i][j]
                    nextBoard[i-1][j] = tmp
                    nextBoard[i][j] = 0
                    return flattenBoard(nextBoard[i-1][j], nextBoard)
    elif action == "down-right":
        for i in range(len(nextBoard)):
            for j in range(len(nextBoard[i])):
                if nextBoard[i][j] == -1 and nextBoard[i+1][j+1] == 1:
                    tmp = nextBoard[i][j]
                    nextBoard[i+1][j+1] = tmp
                    nextBoard[i][j] = 0
                    return flattenBoard(nextBoard[i+1][j+1], nextBoard)
    elif action == "down-left":
        for i in range(len(nextBoard)):
            for j in range(len(nextBoard[i])):
                if nextBoard[i][j] == -1 and nextBoard[i+1][j-1] == 1:
                    tmp = nextBoard[i][j]
                    nextBoard[i+1][j-1] = tmp
                    nextBoard[i][j] = 0
                    return flattenBoard(nextBoard[i+1][j-1], nextBoard)
    elif action == "up-right":
        for i in range(len(nextBoard)):
            for j in range(len(nextBoard[i])):
                if nextBoard[i][j] == 1 and nextBoard[i-1][j+1] == -1:
                    tmp = nextBoard[i][j]
                    nextBoard[i-1][j+1] = tmp
                    nextBoard[i][j] = 0
                    return flattenBoard(nextBoard[i-1][j+1], nextBoard)
    elif action == "up-left":
        for i in range(len(nextBoard)):
            for j in range(len(nextBoard[i])):
                if nextBoard[i][j] == 1 and nextBoard[i-1][j-1] == -1:
                    tmp = nextBoard[i][j]
                    nextBoard[i-1][j-1] = tmp
                    nextBoard[i][j] = 0
                    return flattenBoard(nextBoard[i-1][j-1], nextBoard)
                
def isTerminal(state):
    if state[1] == 1 or state[2] == 1 or state[3] == 1 or state[7] == -1 or state[8] == -1 or state[9] == -1 or actions(state) == []:
        return True
    return False

def utility(state):
    if isTerminal(state) and (state[1] == 1 or state[2] == 1 or state[3] == 1):
        return 1
    elif isTerminal(state) and (state[7] == -1 or state[8] == -1 or state[9] == -1):
        return 0
    elif isTerminal(state):
        return .5
    return 0

def minValue(state):
    if isTerminal(state):
        return (utility(state), None)
    pair = (float('inf'), None)
    for a in actions(state):
        pair2 = maxValue(result(state, a))
        if pair2[0] < pair[0]:
            pair = (pair2[0], a)
    return pair

def maxValue(state):
    if isTerminal(state):
        return (utility(state), None)
    pair = (float('-inf'), None)
    for a in actions(state):
        pair2 = minValue(result(state, a))
        if pair2[0] > pair[0]:
            pair = (pair2[0], a)
    return pair

def miniMaxSearch(state):
    pair = maxValue(state)
    return pair[1]

class NetworkLayer:
    def __init__(self, neurons, inputs, a_fn, a_fn_derivative):
        self.a_fn = a_fn
        self.a_fn_derivative = a_fn_derivative
        self.biases = [random.random() for neuron in range(neurons)]
        self.weights = [[random.random() for input in range(inputs)] for neuron in range(neurons)]

def sigmoid(inputSum):
    return 1/(1 + math.exp(-1 * inputSum))

def reLU(inputSum):
    return max(0, inputSum)

#networkLayers is a list of the class NetworkLayer
def classify(networkLayers, *inputs):
    outputs = np.array(inputs)
    allOutputs = []
    for networkLayer in networkLayers:
        inputs = outputs
        outputs = []
        for neuronIndex in len(range(networkLayer.neurons)):
            inputSum = 0
            for inputIndex in len(range(inputs)):
                inputSum += inputs[inputIndex] * networkLayer.weights[neuronIndex][inputIndex]
            inputSum += networkLayer.biases[neuronIndex]
            output = networkLayer.a_fn(inputSum)
            outputs.append(output)
        allOutputs.append(outputs)
    return (outputs, allOutputs)

#if __name__ == "__main__":
    #s = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]

    #network = Network([1, 2, 3], inputs, a_fn, a_fn_prime)
    #classify(graph, [[3, 5], [5, 1], [10, 2]])
