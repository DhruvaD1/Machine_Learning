import queue 
def bfs(maze):
    start =None
    waypoint=None
    for i in range(maze.size.y):
        for j in range(maze.size.x):
            if maze[i,j]==maze.legend.start:
                start=(i,j)

            if maze[i,j]==maze.legend.waypoint:
                waypoint=(i,j)
    queue = []
    queue.append([start])
    visited = set(); 
    while queue:
        p = queue.pop(0)
        index = p[-1]
        if index not in visited:
            visited.add(index)
            if index == waypoint:
                return p
            for neighbor in maze.neighbors(index[0],index[1]):
                np = list(p)
                np.append(neighbor)
                queue.append(np)
    return []

def astar_single(maze):
    path = []
    queuee = queue.PriorityQueue()
    seen = set()
    dictt = {}
    points = maze.waypoints[0]
    dictt[maze.start] = None
    seen.add(maze.start)
    queuee.put((abs(maze.start[0] -points[0]) + abs(maze.start[1] - points[1]), maze.start, 0) )
    while not queuee.empty():
        val = queuee.get()
        if val[1] != points:
            for near in maze.neighbors(val[1][0], val[1][1]):
                if near not in seen:
                    seen.add(near)
                    dictt[near] = val[1]
                    queuee.put(((abs(near[0] - points[0])+abs(near[1] -points[1]))+val[2]+1, near, val[2]+1))
        else:
            break
    while points != None:
        path.append(points)
        points = dictt[points]
    path.reverse()
    return path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []


import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    ml = []
    t = dict()
    if depth==False:
        return (evaluate(board), ml, t)
    else:
        if side==True:
          vall=99999999999999999999999
        else:
          vall=-99999999999999999999999
        for move in generateMoves(side, board, flags):  
            newside, newboard, newflags = makeMove(side,board, move[0], move[1], flags, move[2])
            score, moveList, moveTree = minimax(newside, newboard,newflags, depth-1)
            t[encode(*move)] =moveTree
            if side == False:
                if score > vall:
                    vall = score
                    ml =[move]+ moveList
            else:
                if score <vall:
                    vall = score
                    ml= [move]+ moveList
        return (vall, ml, t)

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    if depth == 0:
        return chess.lib.heuristics.evaluate(board),[], dict()
    else:
      mt =  dict()
      mov, mmov = None, None
      min_movelist=[]
      max_movelist = []
      min_score, max_score = 999999999999999999, -999999999999999999
      for i in generateMoves(side,board,flags):
          newside, newboard, newflags = chess.lib.makeMove(side, board, i[0], i[1], flags, i[2])
          s, next_movelist, next_movetree =alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
          if s > max_score:
              max_score = s
              max_movelist = next_movelist
              mmov = i
          if s < min_score:
              min_score = s
              min_movelist = next_movelist
              mov = i
          if side:
              beta = min(beta, min_score)
          else:
              alpha = max(alpha, max_score)
          mt[encode(*i)] = next_movetree
          if alpha >= beta:
              break
      if not side:
          return max_score, [mmov] + max_movelist, mt
      else:
          return min_score, [mov] + min_movelist, mt
      
import copy
import queue
import string
import random
import collections


def standardize_variables(nonstandard_rules):
    sets = set()
    def _replace_something(proposition, unique_name):
        if "something" in proposition:
            sets.add(unique_name)
        return [word if word != "something" else unique_name for word in proposition]

    def _standardize(rule, rule_name):
        rules = copy.deepcopy(rule)
        unique_name = str(rule_name) + "".join(random.choices(string.ascii_lowercase + string.digits, k=8) )

        rules["antecedents"] = [_replace_something(antecedent, unique_name)for antecedent in rules["antecedents"]]
        rules["consequent"] = _replace_something(rules["consequent"], unique_name)
        return rules

    rules = {
        rule_name: _standardize(rule, rule_name)
        for rule_name, rule in nonstandard_rules.items()
}
    return rules, sets


def find(i,arr):
    if i not in arr:
        return i
    return find(arr[i], arr)

def unify(query, datum, variables):
    if query[-1] == True and datum[-1] == False:
        return None, None
    if len(query) != len(datum):
        return None, None

    arr = []
    sarr = dict()

    for i in range(len(query)):
        if query[i] in variables and datum[i] in variables:
            sarr[find(query[i],sarr)] =datum[i]
        elif datum[i] in variables:
            sarr[find(datum[i], sarr)] =query[i]
        elif query[i] in variables:
            sarr[find(query[i], sarr)] =datum[i]
        elif query[i] != datum[i]:
            return None, None
    for i in query:
        arr.append(find(i, sarr))

    return arr, sarr


def apply(rule, goals, variables):
    applications = []
    def helper(ss, proposition, variables):
      np = []
      for word in proposition:
          if word not in ss:
              np.append(word)
          else:
              np.append(ss[word])
      return np

    goalsets = []
    for i, mgoal in enumerate(goals):
        ss = unify(mgoal, rule["consequent"], variables)[1]
        if unify(mgoal, rule["consequent"], variables)[0]:
          app = copy.deepcopy(rule)
          app["antecedents"] =[helper(ss, antecedent,variables) for antecedent in app["antecedents"]]
          app["consequent"] = helper(ss, app["consequent"], variables)
          applications.append(app)
          newgoals = copy.deepcopy(goals)
          newgoals.pop(i)
          newgoals.extend(app["antecedents"])
          goalsets.append(newgoals)
    return applications, goalsets


def backward_chain(query, rules, variables):
    state = collections.namedtuple("state", ["applist", "goalset"])
    frontiers = [state([], [query])]
    while frontiers:
        p = frontiers.pop()
        if p.goalset == []:
            return p.applist
        for nothin, i in rules.items():
            applications, new_goalsets = apply(i, p.goalset, variables)
            for application, new_goalset in zip(applications, new_goalsets):
                frontiers.append(state(p.applist +[application], new_goalset))
    return None