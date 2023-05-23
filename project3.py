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
    def __init__(self, numNeurons, inputs, activationFunction, activationFunctionDerivative):
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative
        self.numNeurons = numNeurons
        self.inputs = inputs
        self.biases = [random.randint(-1, 1) for n in range(numNeurons)]
        self.weights = [[random.randint(-1, 1) for input in range(len(inputs))] for n in range(numNeurons)]

def sigmoid(inputSum):
    return 1/(1 + math.exp(-1 * inputSum))

def sigmoidDerivative(inputSum):
    return sigmoid(inputSum) * (1 - sigmoid(inputSum))

def reLU(inputSum):
    return max(0, inputSum)

def reLUDerivative(inputSum):
    if inputSum < 0:
        return 0
    else:
        return 1

def classify(networkLayers, allInputs):
    allOutputs = []
    for inputs in allInputs:
        latestOutputs = inputs
        for networkLayer in networkLayers:
            inputs = latestOutputs
            networkLayer.inputs = inputs
            latestOutputs = []
            for neuronIndex in range(len(networkLayer.inputs) - networkLayer.numNeurons, networkLayer.numNeurons):
                inputSum = 0
                for inputIndex in range(len(networkLayer.inputs)):
                    inputSum += inputs[inputIndex] * networkLayer.weights[neuronIndex][inputIndex]
                inputSum += networkLayer.biases[neuronIndex]
                if networkLayer.activationFunction == "sigmoid":
                    output = sigmoid(inputSum)
                elif networkLayer.activationFunction == "reLU":
                    output = reLU(inputSum)
                latestOutputs.append(output)
            allOutputs.append(latestOutputs)
    return allOutputs

def updateWeights(networkLayers, expectedOutputs):
    calculatedOutputs = classify(networkLayers, networkLayers[0].inputs)
    errorDeltas = [None] * (len(networkLayers) * (len(networkLayers[0].inputs)))
    for outputsIndex in range(len(expectedOutputs)):
        for layerIndex in reversed(range(len(networkLayers))):
            for inputIndex in range(len(networkLayers[layerIndex].inputs)):
                if layerIndex == len(networkLayers) - 1:
                    if networkLayers[layerIndex].activationFunctionDerivative == "sigmoidDerivative":
                        networkLayerDeltaError = 2 * (calculatedOutputs[outputsIndex][inputIndex] - expectedOutputs[outputsIndex][inputIndex]) * sigmoidDerivative(networkLayers[layerIndex].inputs[inputIndex])
                    elif networkLayers[layerIndex].activationFunctionDerivative == "reLUDerivative":
                        networkLayerDeltaError = 2 * (calculatedOutputs[outputsIndex][inputIndex] - expectedOutputs[outputsIndex][inputIndex]) * reLUDerivative(networkLayers[layerIndex].inputs[inputIndex])
                else:
                    networkLayerDeltaError = 0
                    for neuronIndex in range(networkLayers[layerIndex].numNeurons):
                        if networkLayers[layerIndex].activationFunctionDerivative == "sigmoidDerivative":
                            networkLayerDeltaError += errorDeltas[(layerIndex + 1) * len(networkLayers[layerIndex].inputs) + neuronIndex] * networkLayers[layerIndex].weights[neuronIndex][inputIndex] * sigmoidDerivative(networkLayers[layerIndex].inputs[inputIndex])
                        elif networkLayers[layerIndex].activationFunctionDerivative == "reLUDerivative":
                            networkLayerDeltaError += errorDeltas[(layerIndex + 1) * len(networkLayers[layerIndex].inputs) + neuronIndex] * networkLayers[layerIndex].weights[neuronIndex][inputIndex] * reLUDerivative(networkLayers[layerIndex].inputs[inputIndex])
                errorDeltas.insert(layerIndex * len(networkLayers[layerIndex].inputs) + inputIndex, networkLayerDeltaError)
        for networkLayer in networkLayers:
            for inputIndex in range(1, len(networkLayer.inputs)):
                for neuronIndex in range(networkLayer.numNeurons):
                    networkLayer.weights[inputIndex][neuronIndex] += errorDeltas[layerIndex * len(networkLayers[0].inputs) + inputIndex]
    return None

def hexaspawnNetwork():
    networkLayer1 = NetworkLayer(9, [None, None, None, None, None, None, None, None, None, None], "reLU", "reLUDerivative")
    networkLayer2 = NetworkLayer(9, [None, None, None, None, None, None, None, None, None, None], "sigmoid", "sigmoidDerivative")
    print(classify([networkLayer1, networkLayer2], [s0]))
    networkLayer1.inputs = [s0]
    updateWeights([networkLayer1, networkLayer2], [[-1, -1, -1, 1, 0, 0, 0, 1, 1]])
    

if __name__ == "__main__":
    #hexaspawnNetwork()

    #Tests
    sigmoidTest = NetworkLayer(2, [[0, 0], [0, 1], [1, 0], [1, 1]], "sigmoid", "sigmoidDerivative")
    print(updateWeights([sigmoidTest], [[0, 0], [0, 1], [0, 1], [1, 0]]))
    reLUTest = NetworkLayer(2, [[0, 0], [0, 1], [1, 0], [1, 1]], "reLU", "reLUDerivative")
    print(updateWeights([reLUTest], [[0, 0], [0, 1], [0, 1], [1, 0]]))
