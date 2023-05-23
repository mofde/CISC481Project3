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

def classify(networkLayers, inputs):
    outputs = inputs
    #allOutputs = []
    for networkLayer in networkLayers:
        inputs = outputs
        networkLayer.inputs = inputs
        outputs = []
        for neuronIndex in range(networkLayer.numNeurons):
            inputSum = 0
            for inputIndex in range(len(inputs)):
                inputSum += inputs[inputIndex] * networkLayer.weights[neuronIndex][inputIndex]
            inputSum += networkLayer.biases[neuronIndex]
            if networkLayer.activationFunction == "sigmoid":
                output = sigmoid(inputSum)
            elif networkLayer.activationFunction == "reLU":
                output = reLU(inputSum)
            outputs.append(output)
        #allOutputs.append(outputs)
    return outputs
    #return (outputs, allOutputs)

def updateWeights(networkLayers, expectedOutputs):
    calculatedOutputs = classify(networkLayers, networkLayers[0].inputs)
    errorDeltas = [None] * (len(networkLayers) * len(networkLayers[0].inputs))
    for layerIndex in reversed(range(len(networkLayers))):
        for inputIndex in range(len(networkLayers[layerIndex].inputs)):
            if layerIndex == len(networkLayers) - 1:
                if networkLayers[layerIndex].activationFunctionDerivative == "sigmoidDerivative":
                    networkLayerDeltaError = 2 * (calculatedOutputs[inputIndex] - expectedOutputs[inputIndex]) * sigmoidDerivative(networkLayers[layerIndex].inputs[inputIndex])
                elif networkLayers[layerIndex].activationFunctionDerviative == "reLUDerivative":
                    networkLayerDeltaError = 2 * (calculatedOutputs[inputIndex] - expectedOutputs[inputIndex]) * reLUDerivative(networkLayers[layerIndex].inputs[inputIndex])
            else:
                networkLayerDeltaError = 0
                for neuronIndex in len(range(networkLayers[layerIndex].numNeurons)):
                    if networkLayers[layerIndex].activationFunctionDerivative == "sigmoidDerivative":
                        networkLayerDeltaError += errorDeltas[(layerIndex + 1) * len(networkLayers[0].inputs) + neuronIndex] * networkLayers[layerIndex].weights[neuronIndex][inputIndex] * sigmoidDerivative(networkLayers[layerIndex].inputs[inputIndex])
                    elif networkLayers[layerIndex].activationFunctionDerivative == "reLUDerivative":
                        networkLayerDeltaError += errorDeltas[(layerIndex + 1) * len(networkLayers[0].inputs) + neuronIndex] * networkLayers[layerIndex].weights[neuronIndex][inputIndex] * reLUDerivative(networkLayers[layerIndex].inputs[inputIndex])
            errorDeltas.insert(layerIndex * len(networkLayers[0].inputs) + inputIndex, networkLayerDeltaError)
    return errorDeltas

def hexaspaunNetwork():
    networkLayer = NetworkLayer(10, [None, None, None, None, None, None, None, None, None, None], "reLU", "reLUDerivative")
    print(networkLayer.weights)
    print(classify([networkLayer], s0))

if __name__ == "__main__":
    hexaspaunNetwork()