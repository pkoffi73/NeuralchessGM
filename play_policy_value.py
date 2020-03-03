# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 01:02:48 2019

@author: phili
"""

import tensorflow as tf
import numpy as np
import chess, chess.svg
import traceback
import math
import random
import copy
import time
import pickle

from state import State
from flask import Flask, request

''' 
CONSTANTS:
- TIME (seconds): Unit of time for each computer move
        Can be divided by 2 for easy moves
        Can be multiplied by 2 for complex moves
- DELTA_MIN : delta threshold between move with highest numbers of visits
              and move with second highest numbers of visits - used to
              identify easy moves
- CPUCT_INIT & CPUCT_BASE are Monte-Carlo Tree Search algorithm parameters
'''
TIME = 120
DELTA_MIN = 500
CPUCT_INIT = 2.5
CPUCT_BASE = 19652

#Name of Neural Network model file
name_model = 'model/model_value_policy.h5'

#Dictionnary of links between vector component (output of Neural Network)
# and actual chess moves
name_dict = 'dict_policy.pkl'
with open(name_dict, 'rb') as f:
    dict_policy = pickle.load(f)

#Two headed (Policy and Value) Neural Network
class Value_Policy(object):
    def __init__(self):
        self.model = tf.keras.models.load_model(name_model)
        self.memo = {}
        
    def __call__(self, s):
        key = s.key()
        if key not in self.memo:
            self.memo[key] = self.value(s)
        return self.memo[key]
    
    def value(self, s):
        brd = s.bitmap()[None]
        output = self.model.predict(np.array(brd, np.float32), steps=1)
        #output[0][0][0] = board value; output[1][0] = policy vector
        return output[0][0][0], output[1][0]
    
    
class Node():
    def __init__(self, s, vp, prior=0, parent=None):
        self.state = copy.deepcopy(s)
        self.valuator_policy = vp
        #prior is estimated by the Policy head of the NN
        self.prior = prior
        self.visite = 0
        self.children_map = {}
        self.children = []
        self.parent = parent
        #colour = colour of the parent node selecting a child node
        self.colour = not self.state.board.turn
        self.expanded = False
        #value is estimated by the Value head of the NN
        self.value = 0
        #valueN is the accumulated value of the Node (taking into accound its children's values)
        self.valueN = 0
    
    #ubc = Q + U in a Monte Carlo Tree Search algorithm
    def ubc(self):
        cpuct = CPUCT_INIT + \
                math.log((self.parent.visite + 1 + CPUCT_BASE) / CPUCT_BASE)
        return self.valueN / (1 + self.visite) +  \
               cpuct * self.prior * math.sqrt(self.parent.visite) / (1 + self.visite)
    
    def expand(self):
        self.visite += 1
        self.value, probs = self.valuator_policy(self.state)
        if not self.colour:
            self.value = -self.value
        for e in self.state.board.legal_moves:
            self.state.board.push(e)
            try:
                child = Node(self.state, self.valuator_policy,\
                             probs[dict_policy.get(e, 0.)], self)
                self.children_map[child] = e
                self.children.append(child)
            except:
                pass
            self.state.board.pop()
        self.expanded = True

#backtrack node value to the parent nodes            
def backtrack(S, value):
    i = 0
    while S.parent:
        S.valueN += (-1)**i * value
        S = S.parent
        i += 1

'''
Monte-Carlo Tree Search Alogrithm:
    - Initialization
    - Rollout until picking a node which is not expanded or until picking a node
        which is an end state (game over)
    - if game over, backtrack of the actual value of the end state (1 or 0 (draw))
    - if not expanded, node expansion
    - Backtrack of the newly expanded node value
'''
def mtcs(s, vp):
    global q_last
    S0 = Node(s, vp)
    i = 0
    delta_visite = 0
    game_over = False
    start = time.time()
    #delta_Q_MAX : difference between Q value of the current best node and
    #the best current Q value
    #delta_Q_LAST : difference between Q vlaue of the current best node and
    #Q value of the previous move best node
    delta_Q_MAX, delta_Q_LAST = 0, 0
    while True :
        i += 1
        if i % 500 == 0:
            print("iteration #%s time %s delta %s delta_Q_LAST %s delta_Q_MAX %s" \
                  % (i, round(time.time() - start, 1), delta_visite, delta_Q_LAST, delta_Q_MAX))
        S = S0
        while S.expanded:
            S.visite += 1
            if S.children:
                S = max([child for child in S.children], key=lambda x:(x.ubc(), random.random()))
            elif S.state.board.is_game_over():
                if S.state.board.is_checkmate():
                    backtrack(S, 1)
                else:
                    backtrack(S, 0)
                game_over = True
                break
            else:
                break
        if game_over:
            game_over = False
            continue
        S.expand()
        backtrack(S, S.value)
        moves = [(S0.children_map[child], round(child.valueN / (1 + child.visite), 3),\
                  child.visite, round(child.prior, 3), round(child.value, 3)) for child in S0.children]
        if len(moves) > 1:
            #moves sorting to get the highest Q between moves
            moves.sort(key=lambda x: x[1], reverse=True)
            q_max = moves[0][1]
            #moves sorting based on visit number
            moves.sort(key= lambda x:(x[2], x[1]), reverse=True)
            delta_visite = moves[0][2] - moves[1][2]
        else:
            #if only one legal move early loop exit
            q = moves[0][1]
            break
        iteration_time = time.time() - start
        q = moves[0][1]
        prior = moves[0][3]
        delta_Q_LAST = round(q - q_last, 2)
        delta_Q_MAX = round(q - q_max, 2)
        #Exit loop test:
        #   - Easy moves if delta_visite > 500 or if prior > 0.9
        #   - Normal moves if delta_Q_LAST is not less than -0.05
        #   - Complex moves
        if iteration_time > TIME / 2 and (delta_visite > DELTA_MIN or prior > 0.9) \
            and delta_Q_MAX >= -0.01  and delta_Q_LAST >= -0.05:
            break
        elif iteration_time > TIME and \
            (delta_Q_LAST >= -0.05 or delta_visite > DELTA_MIN * 2):
            break
        elif iteration_time > TIME * 2:
            break
    q_last = q
    # return the top five moves (to be displayed)
    return moves[:min(5, len(moves))]    
                
                
def computer_move(s, vp):
    global q_last, move_number
    q0 = q_last
    print('move #%s' % move_number)
    move_number += 1
    move = mtcs(s, vp)
    if len(move) == 0:
        return
    for i, m in enumerate(move[:]):
        print(" move: %s Q: %s  N: %s P: %s V %s Q_LAST %s" % \
              (m[0], m[1], m[2], m[3], m[4], round(q0, 3)))
    print("computer moves", move[0][0])
    s.board.push(move[0][0])

app = Flask(__name__)

@app.route("/")
def index():
    ret = open("index.html").read()
    return ret.replace('start', s.board.fen())

@app.route("/move_coordinates")
def move_coordinates():
    if not s.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False
        move = s.board.san(chess.Move(source, target, \
                            promotion=chess.QUEEN if promotion else None))   
        if move is not None and move != "":
          print("human moves", move)
          try:
            s.board.push_san(move)
            computer_move(s, vp)
          except Exception:
            traceback.print_exc()
        response = app.response_class(
          response=s.board.fen(),
          status=200)
        if s.board.is_game_over():
            print("GAME IS OVER")
            if s.board.result() == '1-0':
                print("WHITE WINS")
            elif s.board.result() == '0-1':
                print("BLACK WINS")
            else:
                print("IT'S A DRAW")            
        return response
    return response

@app.route("/newgame")
def newgame():
    global move_number
    s.board.reset()
    move_number = 1
    response = app.response_class(
            response=s.board.fen(),
            status=200)
    return response

@app.route("/white", methods=['POST'])
def white():
    computer_move(s, vp)
    ret = open("index.html").read()
    return ret.replace('start', s.board.fen())

@app.route("/remove_moves", methods=['POST'])
def remove():
    global move_number
    s.board.pop()
    s.board.pop()
    move_number -= 1
    print('last computer and human moves removed')
    ret = open("index.html").read()
    return ret.replace('start', s.board.fen())

if __name__ == "__main__":
    s = State()
    vp = Value_Policy()
    q_last = -1
    move_number = 1
    app.run(debug=False)
