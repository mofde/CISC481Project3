import numpy as np

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

if __name__ == "__main__":
    s = [1, -1, 0, 0, 1, 0, -1, 0, 0, 1]
    print(isTerminal(s))
