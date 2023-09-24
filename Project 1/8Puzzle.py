# Created by Chaehyeon Kim (cxk445) for CSDS 391 Project 1
# Algorithms for solving an 8-puzzle
import sys
import random
import copy
from queue import PriorityQueue

puzzleState = None

# Set the puzzle state
def setState(state):
    global puzzleState
    puzzleState = list(state)

# Print the current puzzle state
def printState():
    if puzzleState == None:
        print("printState: no puzzle to print") #FIX? throw an error consistently or is it okay to just print?
        return
    print("\n8 Puzzle:")
    print(puzzleState[:3])
    print(puzzleState[4:7])
    print(puzzleState[8:11])

# Move the blank tile to the input direction (up, down, left, right)
def move(puzzle, direction): # okay to have diff. parameters
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
    it = 0 # for iteration
    while (it < n):
        state = move("default", random.choice(moves))
        if (None != state):
            it += 1
            setState(state)

# Solve the puzzle from its current state using A-star search
def solveAStar(heuristic):
    g = 0 # g(n) = depth of the state from the initial state; initially 0
    if (heuristic == "h1"):
        h = numMisplacedTiles(puzzleState) # h(n) = heuristic; number of misplaced tiles in the current state
    else:
        h = manhattanDistance(puzzleState) # h(n) = Manhattan distance
    
    open = PriorityQueue() # priority queue containing possible states
    closed = list() # list containing already-visited states

    # Insert each state in as a tuple (f-score, puzzle state, parent, direction)
    open.put((g+h, puzzleState, None, None)) # Initial state

    while (not open.empty()):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        successorList = list()
        depth += 1

        # Get all possible future states from the current state
        right = move(state, "right")
        left = move(state, "left")
        up = move(state, "up")
        down = move(state, "down")
        successorList.add((numMisplacedTiles(right)+depth, right, state, "right"))
        successorList.add((numMisplacedTiles(left)+depth, left, state, "left"))
        successorList.add((numMisplacedTiles(up)+depth, up, state, "up"))
        successorList.add((numMisplacedTiles(down)+depth, down, state, "down"))
        # copy for the other three
        
        # Check whether to add the four successors to the open queue
        if (): # if the same state as this current node is in the open or closed queue with a lower f-score, don't add. Else, do add.
            print()
        closed.add(fromQueue)

# A-star search using H1 (= # misplaced tiles)
# Keep both open and closed queues for memory
def AStarH1():
    g = 0 # g(n) = depth of the state from the initial state; initially 0
    h = numMisplacedTiles(puzzleState) # h(n) = heuristic; number of misplaced tiles in the current state
    open = PriorityQueue() # priority queue containing possible states
    closed = list() # priority queue containing visited states

    # Insert each state in as a tuple; insert current/initial state (f-score, puzzle state, parent, direction)
    open.put((g+h, puzzleState, None, None))

    while (not open.empty()):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        successorList = list()
        depth += 1

        # Get all possible future states from the current state
        right = move(state, "right")
        left = move(state, "left")
        up = move(state, "up")
        down = move(state, "down")
        successorList.add((numMisplacedTiles(right)+depth, right, state, "right"))
        successorList.add((numMisplacedTiles(left)+depth, left, state, "left"))
        successorList.add((numMisplacedTiles(up)+depth, up, state, "up"))
        successorList.add((numMisplacedTiles(down)+depth, down, state, "down"))
        # copy for the other three
        
        # Check whether to add the four successors to the open queue
        if (): # if the same state as this current node is in the open or closed queue with a lower f-score, don't add. Else, do add.
            print()
        closed.add(fromQueue)



        exit()


        
        tempR = copy.deepcopy(puzzleState)
        tempL = copy.deepcopy(puzzleState)
        tempU = copy.deepcopy(puzzleState)
        tempD = copy.deepcopy(puzzleState)
        if (1 == move(tempR, "right")):
            pq.put((numMisplacedTiles(tempR) + depth, tempR, "right")) # maybe add another for direction
        if (1 == move(tempL, "left")):
            pq.put((numMisplacedTiles(tempL)+ depth, tempL, "left"))
        if (1 == move(tempU, "up")):
            pq.put((numMisplacedTiles(tempU)+ depth, tempU, "up"))
        if (1 == move(tempD, "down")):
            pq.put((numMisplacedTiles(tempD)+ depth, tempD, "down"))



     
    while (h1 > 0): # does it only look one step ahead? if so, just choose a state with the lowest heuristics
              

        # use priority queue to choose the lowest h1
        pq = PriorityQueue()
        

        newState = pq.get()
        setState(newState[1])
        h1 = newState[0]

        depth += 1
        print(newState[2])
        printState()
    exit()
    return depth


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
def manhattanDistance():
    print("a star h2")

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
        print("Include one text file: python/python3 8Puzzle.py <text file containing text commands>")
        exit()
    with open(sys.argv[1], 'r') as f:
        textFile = f.read()

    # Translating the lines from the file to arguments in this program
    print("Executing commands...")
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
            setState(''.join(move(puzzle="default", direction=line.split()[1])))
        elif (methodName == "randomizeState"):
            randomizeState(int(line.split()[1]))
        elif (methodName == "solve"):
            if (line.split()[1] == "A-star"):
                solveAStar(line.split()[2])
            elif (line.split()[1] == "beam"):
                solveBeam(int(line.split()[2]))
        elif (methodName == "maxNodes"):
            maxNodes(int(line.split()[1]))
        
    # Printing out the output to the terminal using stdout within each method