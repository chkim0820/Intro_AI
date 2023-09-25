# Created by Chaehyeon Kim (cxk445) for CSDS 391 Project 1
# Algorithms for solving an 8-puzzle
import sys
import random
import copy
from queue import PriorityQueue

# Stores the current puzzle
puzzleState = None
# Stores the max number of nodes to be considered during search
maxNodesLimit = -1 # -1 if no limit

# Set the puzzle state
def setState(state):
    global puzzleState
    puzzleState = list(state)

# Print the current puzzle state or the input state
def printState(state="default"):
    if state == "default": # set state to the current state
        sys.stdout.write("Printing the current puzzle state:\n")
        state = puzzleState
    if state == None: # null state
        sys.stdout.write("printState: no puzzle to print\n") #FIX? throw an error consistently or is it okay to just print?
        return
    for i in range(0, len(state), 4):
        sys.stdout.write(str(state[i:i+3]) + "\n\n")    

# Move the blank tile to the input direction (up, down, left, right)
def move(puzzle="default", direction="up"): # okay to have diff. parameters
    if puzzle == "default":
        copyPuzzle =  copy.deepcopy(puzzleState)
    else:
        copyPuzzle = copy.deepcopy(puzzle)
    pos = copyPuzzle.index("0") # index of 0 (empty space)
    
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
        return False, puzzle if puzzle != "default" else puzzleState # input not a valid movement; return original
    copyPuzzle[pos] = copyPuzzle[newPos]
    copyPuzzle[newPos] = '0'

    return True, copyPuzzle # movement successful

# Make n random moves from the goal state; for ensuring solvable puzzle
def randomizeState(n):
    moves = ["up", "down", "left", "right"] # possible moves
    setState("012 345 678")
    it = 0 # iteration
    while (it < n):
        valid, state = move(direction=random.choice(moves))
        if (valid):
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

    i = 0
    while (not open.empty() and (True if (maxNodesLimit == -1 or i < maxNodesLimit) else False)):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        g += 1
        i += 1

        if numMisplacedTiles(state) == 0: # if goal state reached
            sys.stdout.write("goal reached!\n")
            traverseBackMoves(fromQueue)
            return state

        # Get all valid future states from the current state
        moves = ["up", "down", "left", "right"]
        for mv in moves: # try moves in all directions
            valid, nextState = move(state, mv)
            if valid: # is a valid movement
                h = numMisplacedTiles(nextState) if heuristic == "h1" else manhattanDistance(nextState)
                if listSearch(closed, nextState, g+h) < 0: # the same state is not in the closed list w/ smaller f-score
                    open.put((g+h, nextState, fromQueue, mv))
        
        # Append the current iteration's state to the closed list
        closed.append(fromQueue)
    if (maxNodesLimit != -1): # maxNodesLimit reached
        sys.stdout.write("Limit reached for maximum number of nodes considered\n")
    else: # solution does not exist
        sys.stdout.write("Unsolvable 8 puzzle\n")
    return None

# Returns the correct number of misplaced tiles
def numMisplacedTiles(puzzle):
    correctTiles = 0 # number of correct tiles
    index = 0 # correct index
    for n in range(len(puzzle)): # compare all tiles (1 to 8)
        curTile = puzzle[n]
        if (n != 3 and n != 7): # if not ' ' (blank index in list)
            if (index == int(curTile)):
                correctTiles += 1
            index += 1
    return 9 - correctTiles

# A-star search using h2 (= sum of the distances of the tiles from their goal positions)
def manhattanDistance(puzzle):
    sum = 0 # sum of Manhattan distances
    copyPuzzle = copy.deepcopy(puzzle) # to avoid altering original object
    del copyPuzzle[3]
    del copyPuzzle[6]
    for i in range(len(copyPuzzle)): # compare all tiles (1 to 8)
        curTile = int(copyPuzzle[i]) # value of ith tile
        sum += round(abs(curTile-i)/3) + abs(curTile-i)%3
    return sum

# Simple list search method for a list containing tuples; search for the input state
# Return 1 if the key exists, -1 if it doesn't in the list
def listSearch(list, key, f):
    for i in list:
        if i[1] == key and i[0] < f:
            return 1
    return -1

# Print out the steps from initial state to goal state
def traverseBackMoves(state):
    moves = list() # moves/directions made from beginning to end
    states = list() # states visited from beginning to end
    parent = state
    
    while (parent != None):
        states.insert(0, parent[1]) # insert parent's state to the front
        moves.insert(0, parent[3]) # insert parent's move to the front
        parent = parent[2]

    sys.stdout.write("\nTotal number of moves: " + str(len(moves)-1) + "\n")
    sys.stdout.write("\nInitial state:\n")
    printState(states[0])
    sys.stdout.write("Moves: " + str(moves[1:]) + "\n\n")
    # Commented under prints out all moves and states to get to the goal
    # for i in range(1, len(moves)):
        # sys.stdout.write("\n" + str(i) +") Move " + moves[i] + ":\n")
        # printState(states[i])
        

# Solve the puzzle from its current state by adapting local beam search with k states
def solveBeam(k):
    sys.stdout.write("\nSolving the puzzle using beam search with " + str(k) + " states...\n")
    g = 0 # g(n) = depth of the state from the initial state; initially 0
    h = numMisplacedTiles(puzzleState) # h(n) = heuristic
    
    open = PriorityQueue() # priority queue containing possible states
    closed = list() # list containing already-visited states

    # Insert each state in as a tuple (f-score, puzzle state, parent, direction)
    open.put((g+h, puzzleState, None, None)) # initial state

    i = 0
    while (not open.empty() and (True if (maxNodesLimit == -1 or i < maxNodesLimit) else False)):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        g += 1
        i += 1

        if numMisplacedTiles(state) == 0: # if goal state reached
            sys.stdout.write("goal reached!\n")
            traverseBackMoves(fromQueue)
            return state

        # Get all valid future states from the current state
        moves = ["up", "down", "left", "right"]
        for mv in moves: # try moves in all directions
            valid, nextState = move(state, mv)
            if valid: # is a valid movement
                h = numMisplacedTiles(nextState)
                if listSearch(closed, nextState, g+h) < 0: # the same state is not in the closed list w/ smaller f-score
                    open.put((g+h, nextState, fromQueue, mv))

        # Cut the nodes/states with the most f-scores; leave k queues
        open = cutQueue(k, open)
        # Append the current iteration's state to the closed list
        closed.append(fromQueue)
    if (maxNodesLimit != -1): # maxNodesLimit reached
        sys.stdout.write("Limit reached for maximum number of nodes considered\n")
    else: # beam search couldn't find solution either because no solution exists or because it is an incomplete search
        sys.stdout.write("Beam search could not find a solution...\n")
    return None

# Cutting down the size of input queue to k
def cutQueue(k, queue):
    newQueue = PriorityQueue()
    i = 0
    while not queue.empty() and i < k:
    # while not (queue.empty() or i > k):
        newQueue.put(queue.get())
        i += 1
    return newQueue

# Specify the maximum number of nodes to be considered during a search
def maxNodes(n=-1):
    global maxNodesLimit
    maxNodesLimit = n

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
            valid, state = move(direction=line.split()[1])
            setState(''.join(state))
        elif (methodName == "randomizeState"):
            randomizeState(int(line.split()[1]))
        elif (methodName == "solve"):
            if (line.split()[1] == "A-star"):
                if (len(line.split()) < 3): # if no specification for heuristic
                    solveAStar()
                else:
                    solveAStar(line.split()[2])
            elif (line.split()[1] == "beam"):
                solveBeam(int(line.split()[2]))
        elif (methodName == "maxNodes"):
            maxNodes(int(line.split()[1]))
        
    # Printing out the output to the terminal using stdout within each method