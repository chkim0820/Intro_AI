# Created by Chaehyeon Kim (cxk445) for CSDS 391 project 1 extra credit
# Algorithms for solving a 15-puzzle
import sys
import random
import copy
import math
import time
from queue import PriorityQueue

# Stores the current puzzle
puzzleState = None
# Stores the max number of nodes to be considered during search
maxNodesLimit = -1 # -1 if no limit
defaultState = ['0', '1', '2', '3', 
                '4', '5', '6', '7', 
                '8', '9', '10', '11', 
                '12', '13', '14', '15']

# Set the puzzle state
def setState(state):
    global puzzleState
    stateLength = len(state)
    if (stateLength == 16):
        for i in [4, 9, 14]:
            state.insert(i, ' ')
    elif (stateLength != 19):
        sys.stdout.write("Wrong input for setState")
        exit()
    puzzleState = state

# Print the current puzzle state or the input state
def printState(state="default"):
    if state == "default": # set state to the current state
        sys.stdout.write("Printing the current puzzle state:\n")
        state = puzzleState
    if state == None: # null state
        sys.stdout.write("printState: no puzzle to print\n")
        return
    for i in range(0, len(state), 5):
        sys.stdout.write(str(state[i:i+4]) + "\n\n")

# Move the blank tile to the input direction (up, down, left, right)
def move(puzzle="default", direction="up"): # okay to have diff. parameters
    if puzzle == "default":
        copyPuzzle =  copy.deepcopy(puzzleState)
    else:
        copyPuzzle = copy.deepcopy(puzzle)
    pos = copyPuzzle.index("0") # index of 0 (empty space)
    
    if (direction == "up"):
        newPos = pos - 5
    elif (direction == "down"):
        newPos = pos + 5
    elif (direction == "left"):
        newPos = pos - 1
    elif (direction == "right"):
        newPos = pos + 1
    # Check the validity of the movement
    if ((newPos < 0 or newPos > 18) or (newPos == 4 or newPos == 9 or newPos == 14)): # Falls outside of bound or invalid indices
        return False, puzzle if puzzle != "default" else puzzleState # input not a valid movement; return original
    copyPuzzle[pos] = copyPuzzle[newPos]
    copyPuzzle[newPos] = '0'

    return True, copyPuzzle # movement successful

# Make n random moves from the goal state; for ensuring solvable puzzle
def randomizeState(n):
    if n > 0:
        moves = ["up", "down", "left", "right"] # possible moves
        setState(defaultState) # Goal state
        it = 0 # iteration
        while (it < n):
            valid, state = move(direction=random.choice(moves))
            if (valid):
                it += 1
                setState(state)
        return copy.deepcopy(state)
    else:
        return None

# Solve the puzzle from its current state using A-star search
def solveAStar(heuristic="h1", puzzle="default"):
    if puzzle == "default":
        puzzle = puzzleState
    sys.stdout.write("\nSolving the puzzle using A* search...\n")
    g = 0 # g(n) = depth of the state from the initial state; initially 0
    if (heuristic == "h1"):
        h = numMisplacedTiles(puzzle) # h(n) = heuristic; number of misplaced tiles in the current state
    else:
        h = manhattanDistance(puzzle) # h(n) = Manhattan distance
    
    open = PriorityQueue() # priority queue containing possible states
    closed = list() # list containing already-visited states

    # Insert each state in as a tuple (f-score, puzzle state, parent, direction)
    open.put((g+h, puzzle, None, None)) # initial state

    i = 0
    while (not open.empty() and (True if (maxNodesLimit == -1 or i < maxNodesLimit) else False)):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        i += 1

        if numMisplacedTiles(state) == 0: # if goal state reached
            sys.stdout.write("goal reached!\n")
            return traverseBackMoves(fromQueue, 0)

        # Get all valid future states from the current state
        moves = ["up", "down", "left", "right"]
        for mv in moves: # try moves in all directions
            valid, nextState = move(state, mv)
            if valid: # is a valid movement
                g = traverseBackMoves(fromQueue, 1) + 1
                h = numMisplacedTiles(nextState) if heuristic == "h1" else manhattanDistance(nextState)
                if listSearch(closed, nextState, g+h) < 0: # the same state is not in the closed list w/ smaller f-score
                    open.put((g+h, nextState, fromQueue, mv))
        
        # Append the current iteration's state to the closed list
        closed.append(fromQueue)
    if (maxNodesLimit != -1): # maxNodesLimit reached
        sys.stdout.write("Limit reached for maximum number of nodes considered\n")
    else: # solution does not exist
        sys.stdout.write("Unsolvable 8 puzzle\n")
    return -1 # returned if no moves to return

# Returns the correct number of misplaced tiles
def numMisplacedTiles(puzzle): # FIX
    correctTiles = 0 # number of correct tiles
    index = 0 # correct index
    for n in range(len(puzzle)): # compare all tiles (0/empty to 15)
        curTile = puzzle[n]
        if (n != 4 and n != 9 and n != 14): # if not ' ' (blank index in list)
            if (index == int(curTile)):
                correctTiles += 1
            index += 1
    return 16 - correctTiles

# A-star search using h2 (= sum of the distances of the tiles from their goal positions)
def manhattanDistance(puzzle):
    sum = 0 # sum of Manhattan distances
    copyPuzzle = copy.deepcopy(puzzle) # to avoid altering original object
    del copyPuzzle[4]
    del copyPuzzle[8]
    del copyPuzzle[12]
    for i in range(len(copyPuzzle)): # compare all tiles (1 to 8)
        curTile = int(copyPuzzle[i]) # value of ith tile
        sum += abs(curTile//4 - i//4) + abs(curTile%4 - i%4) # % calculates x/column & // calculates y/row
    return sum

# Simple list search method for a list containing tuples; search for the input state
# Return 1 if the key exists, -1 if it doesn't in the list
def listSearch(list, key, f): # closed, nextState, f-score
    for i in list:
        if i[1] == key and i[0] <= f: # same as visited state with lower or same f-score
            return 1
    return -1

# Print out the steps from initial state to goal state
def traverseBackMoves(state, printOut=1):
    moves = list() # moves/directions made from beginning to end
    states = list() # states visited from beginning to end
    parent = state
    
    while (parent != None):
        states.insert(0, parent[1]) # insert parent's state to the front
        moves.insert(0, parent[3]) # insert parent's move to the front
        parent = parent[2]

    if printOut==0:
        sys.stdout.write("\nTotal number of moves: " + str(len(moves)-1) + "\n")
        sys.stdout.write("\nInitial state:\n")
        printState(states[0])
        sys.stdout.write("Moves: " + str(moves[1:]) + "\n\n")

    return len(moves) - 1
        
# Solve the puzzle from its current state by adapting local beam search with k states
def solveBeam(k, puzzle="default"):
    if puzzle == "default":
        puzzle = puzzleState
    sys.stdout.write("\nSolving the puzzle using beam search with " + str(k) + " states...\n")
    g = 0 # g(n) = depth of the state from the initial state; initially 0
    h = manhattanDistance(puzzle) # h(n) = heuristic
    
    open = PriorityQueue() # priority queue containing possible states
    closed = list() # list containing already-visited states

    # Insert each state in as a tuple (f-score, puzzle state, parent, direction)
    open.put((g+h, puzzle, None, None)) # initial state

    i = 0
    while (not open.empty() and (True if (maxNodesLimit == -1 or i < maxNodesLimit) else False)):
        fromQueue = open.get() # get the front node from open queue
        state = fromQueue[1] # puzzle state from the above node
        i += 1

        if numMisplacedTiles(state) == 0: # if goal state reached
            sys.stdout.write("goal reached!\n")
            return traverseBackMoves(fromQueue, 0)

        # Get all valid future states from the current state
        moves = ["up", "down", "left", "right"]
        for mv in moves: # try moves in all directions
            valid, nextState = move(state, mv)
            if valid: # is a valid movement
                g = traverseBackMoves(fromQueue, 1) + 1
                h = manhattanDistance(nextState)
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
    return -1 # returned if no moves to return

# Cutting down the size of input queue to k
def cutQueue(k, queue):
    newQueue = PriorityQueue()
    i = 0
    while not queue.empty() and i < k:
        newQueue.put(queue.get())
        i += 1
    return newQueue

# Specify the maximum number of nodes to be considered during a search
def maxNodes(n=-1):
    global maxNodesLimit
    maxNodesLimit = n


######################### Experiment functions below #########################


# Experiment results are printed out to the terminal
def allExperiments():
    states = generateRandomSamples() # generate the states for experiments

    # Test different maxNodes limits
    experimentOne(states)
    maxNodes(-1)

    solA1, solA2, solB = searchResults(states, -1) # Saved so the search algorithms run only once
    experimentTwo(solA1, solA2)
    experimentThree(solA1, solA2, solB)
    experimentFour(solA1, solA2, solB)

# For experiment, generate 50 samples of puzzles with 1-50 random moves from goal state
def generateRandomSamples():
    states = list()
    i = 1
    while i <= 50: 
        puzzle = randomizeState(i)
        if puzzle not in states:
            states.append(puzzle)
            i += 1
    return states

# Solve the given 50 random states with the given algorithm and other specifications
def testSearch(algorithm, states, heuristic=0, limit=-1):
    maxNodes(limit) # set maxNodes limit
    valid = list() # list of solved puzzles
    for state in states: # for all 50 generated puzzles
        numMoves = 0 # number of moves to goal state
        start = time.time()
        if (algorithm==0):
            numMoves = solveAStar(puzzle=state, heuristic="h1") if heuristic==0 else solveAStar(puzzle=state, heuristic="h2")
        else:
            numMoves = solveBeam(k=10, puzzle=state)     
        end = time.time()
        if numMoves!=-1: # search works
            valid.append((numMoves, state, end - start))
    maxNodes(-1) # set maxNodes limit back to -1 (no limit)
    return valid

# All three algorithms run on the 50 random puzzles with testSearch function
def searchResults(states, limit=-1):
    solA1 = testSearch(algorithm=0, states=states, heuristic=0, limit=limit)
    solA2 = testSearch(algorithm=0, states=states, heuristic=1, limit=limit)
    solB = testSearch(algorithm=1, states=states, limit=limit)
    return solA1, solA2, solB

# Compare number of solvable puzzles with different maxNodes limits
def experimentOne(states):
    sys.stdout.write("Experiment 1) Fraction of solvable puzzles with different maxNodes limits:\n")
    for i in [100, 1000, 10000]:
        sys.stdout.write("\nmaxNodes = " + str(i) + "\n")
        a1, a2, b = searchResults(states, limit=i)
        numValid = len(a1)
        sys.stdout.write("A* search with h1: " + str(numValid) + "/50 = " + str(numValid/50) + "\n")
        numValid = len(a2)
        sys.stdout.write("A* search with h2: " + str(numValid) + "/50 = " + str(numValid/50) + "\n")
        numValid = len(b)
        sys.stdout.write("Beam search: " + str(numValid) + "/50 = " + str(numValid/50) + "\n")

# Determine which heuristic is better; compare runtime since guaranteed optimality & completeness
def experimentTwo(solA1, solA2):
    sys.stdout.write("\nExperiment 2) Testing which A* search heuristic is better:\n")

    sys.stdout.write("h1 (number of misplaced tiles):\n")
    sum = 0
    for i in solA1:
        sum += i[2]
    sys.stdout.write("For A* search with h1, the average runtime over the sample 50 puzzles is " + str(sum/len(solA1)) + "\n") 

    sys.stdout.write("h2 (Manhattan distance):\n")
    sum = 0
    for i in solA2:
        sum += i[2]
    sys.stdout.write("For A* search with h2, the average runtime over the sample 50 puzzles is " + str(sum/len(solA2)) + "\n")  
    
# Compares the solution length (number of moves from beginning to end) across three algorithms
def experimentThree(solA1, solA2, solB):
    sys.stdout.write("\nExperiment 3) Variance in solutions' lengths (in number of moves);" +  
          "The average over solving all solvable puzzles (out of 50) are calculated.\n")
    sum = 0
    for i in solA1:
        sum += i[0]
    sys.stdout.write("For A* search with h1: " + str(sum/len(solA1)) + "\n")
    sum = 0
    for i in solA2:
        sum += i[0]
    sys.stdout.write("For A* search with h2: " + str(sum/len(solA2)) + "\n")  
    sum = 0
    for i in solB:
        sum += i[0]
    sys.stdout.write("For beam search: " + str(sum/len(solB)) + "\n")

def experimentFour(solA1, solA2, solB):
    sys.stdout.write("\nExperiment 4) The fraction of solvable puzzles:\n")
    sys.stdout.write("For A* search with h1: " + str(len(solA1)) + " solved puzzles -> " + str(len(solA1)/50) + "\n")
    sys.stdout.write("For A* search with h2: " + str(len(solA2)) + " solved puzzles -> " + str(len(solA2)/50) + "\n")
    sys.stdout.write("For beam search: " + str(len(solB)) + " solved puzzles -> " + str(len(solB)/50) + "\n")


# Main method
if __name__ == '__main__':

    allExperiments() # Calls the experiment functions
    exit()
    
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
            setState(line.split()[1:17])
        elif (methodName == "printState"):
            printState()
        elif (methodName == "move"):
            valid, state = move(direction=line.split()[1])
            setState(state)
        elif (methodName == "randomizeState"):
            randomizeState(int(line.split()[1]))
        elif (methodName == "solve"):
            if (line.split()[1] == "A-star"):
                if (len(line.split()) < 3): # if no specification for heuristic
                    solveAStar()
                else:
                    solveAStar(heuristic=line.split()[2])
            elif (line.split()[1] == "beam"):
                solveBeam(k=int(line.split()[2]))
        elif (methodName == "maxNodes"):
            maxNodes(int(line.split()[1]))
        
    # Printing out the output to the terminal using stdout within each method