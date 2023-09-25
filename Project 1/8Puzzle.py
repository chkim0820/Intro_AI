# Created by Chaehyeon Kim (cxk445) for CSDS 391 Project 1
# Algorithms for solving an 8-puzzle
import sys
import random
import copy
from queue import PriorityQueue

# Stores the current puzzle
puzzleState = None

# Set the puzzle state
def setState(state):
    global puzzleState
    puzzleState = list(state)

# Print the current puzzle state or the input state
def printState(state="default"):
    if state == "default":
        sys.stdout.write("Printing the current puzzle state:\n")
        state = puzzleState
    if state == None:
        sys.stdout.write("printState: no puzzle to print\n") #FIX? throw an error consistently or is it okay to just print?
        return
    for i in range(0, len(state), 4):
        sys.stdout.write(str(state[i:i+3]) + "\n")    


# Move the blank tile to the input direction (up, down, left, right)
def move(puzzle="default", direction="up"): # okay to have diff. parameters
    if puzzle == "default":
        puzzle =  copy.deepcopy(puzzleState)
    else:
        puzzle = copy.deepcopy(puzzle)
    pos = puzzle.index("0") # index of 0 (empty space)
    
    if (direction == "up"):
        newPos = pos - 4
    elif (direction == "down"):
        newPos = pos + 4
    elif (direction == "left"):
        newPos = pos - 1
    elif (direction == "right"):
        newPos = pos + 1
    # Check the validity of the movement
    if (newPos < 0 or newPos > 10 or newPos == 3 or newPos == 7): 
        return None # input not a valid movement
    puzzle[pos] = puzzle[newPos]
    puzzle[newPos] = '0'

    return puzzle # movement successful

# Make n random moves from the goal state; for ensuring solvable puzzle
def randomizeState(n):
    moves = ["up", "down", "left", "right"]
    setState("012 345 678")
    it = 0 # iteration
    while (it < n):
        state = move(direction=random.choice(moves))
        if (None != state):
            it += 1
            setState(state)

# Solve the puzzle from its current state using A-star search
def solveAStar(heuristic="h1"):
    sys.stdout.write("\nSolving the puzzle using A* search...\n")
    g = 0 # g(n) = depth of the state from the initial state; initially 0
    if (heuristic == "h1"):
        h = numMisplacedTiles(puzzleState) # h(n) = heuristic; number of misplaced tiles in the current state
    else:
        h = manhattanDistance(puzzleState) # h(n) = Manhattan distance
    
    open = PriorityQueue() # priority queue containing possible states
    closed = list() # list containing already-visited states

    # Insert each state in as a tuple (f-score, puzzle state, parent, direction)
    open.put((g+h, puzzleState, None, None)) # initial state

    while (not open.empty()):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        g += 1

        if numMisplacedTiles(state) == 0: # if goal state reached
            sys.stdout.write("goal reached!\n")
            traverseBackMoves(fromQueue)
            return state

        # Get all valid future states from the current state
        moves = ["up", "down", "left", "right"]
        for mv in moves: # try moves in all directions
            nextState = move(state, mv)
            if nextState != None and listSearch(closed, nextState) < 0: # is a valid movement and the same state is not in the closed list
                h = numMisplacedTiles(nextState) if heuristic == "h1" else manhattanDistance(nextState)
                open.put((g+h, nextState, fromQueue, mv))
        
        closed.append(fromQueue)
    sys.stdout.write("Unsolvable 8 puzzle\n")
    return None

# Returns the correct number of misplaced tiles
def numMisplacedTiles(puzzle):
    correctTiles = 0
    index = 1
    for n in range(len(puzzle)): # compare all tiles (1 to 8)
        curTile = puzzle[n]
        if (curTile != '0' and curTile != ' '): # only if a valid tile exists
            if (index == int(curTile)):
                correctTiles += 1
            index += 1
    return 8 - correctTiles

# A-star search using h2 (= sum of the distances of the tiles from their goal positions)
def manhattanDistance(puzzle):
    sum = 0
    for n in range(len(puzzle)): # compare all tiles (1 to 8)
        curTile = puzzle[n]
        if (curTile != '0' and curTile != ' '): # only if a valid tile exists
            #calculate distance here
            print()
    return 0

# Simple list search method for a list containing tuples; search for the input state
# Return 1 if the key exists, -1 if it doesn't in the list
def listSearch(list, key):
    for i in list:
        if i[1] == key:
            return 1
    return -1

def traverseBackMoves(state):
    moves = list() # moves/directions made from beginning to end
    states = list() # states visited from beginning to end
    parent = state
    
    while (parent != None):
        states.insert(0, parent[1])
        moves.insert(0, parent[3])
        parent = parent[2]

    sys.stdout.write("\nInitial state:\n")
    printState(states[0])
    for i in range(1, len(moves)):
        sys.stdout.write("\n" + str(i) +") Move " + str(moves[i]) + ":\n")
        printState(states[i])


# Solve the puzzle from its current state by adapting local beam search with k states
def solveBeam(k):
    print("solveBeam called with input:", k)

# Specify the maximum number of nodes to be considered during a search
def maxNodes(n):
    print("maxNodes called with input:", n)

# Main method
if __name__ == '__main__':

    # Reading the input file from the terminal
    if len(sys.argv) < 2:
        sys.stdout.write("Include one text file: python/python3 8Puzzle.py <text file containing text commands>")
        exit()
    with open(sys.argv[1], 'r') as f:
        textFile = f.read()

    # Translating the lines from the file to arguments in this program
    sys.stdout.write("Executing commands from the input text file...\n")
    for line in textFile.splitlines():
        try:
            methodName = line.split()[0]
        except:
            continue # no valid function call in this line

        if (methodName == "setState"):
            input = line.split()[1] + " " + line.split()[2] + " " +line.split()[3]
            setState(input)
        elif (methodName == "printState"):
            printState()
        elif (methodName == "move"):
            setState(''.join(move(direction=line.split()[1])))
        elif (methodName == "randomizeState"):
            randomizeState(int(line.split()[1]))
        elif (methodName == "solve"):
            if (line.split()[1] == "A-star"):
                if (len(line) < 3):
                    solveAStar()
                else:
                    solveAStar(line.split()[2])
            elif (line.split()[1] == "beam"):
                solveBeam(int(line.split()[2]))
        elif (methodName == "maxNodes"):
            maxNodes(int(line.split()[1]))
        
    # Printing out the output to the terminal using stdout within each method