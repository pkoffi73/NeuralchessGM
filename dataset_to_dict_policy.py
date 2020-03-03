# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:37:46 2019

@author: phili
"""

import os
import chess.pgn
from state import State
import pickle

''' 
name_directory: name of directory where the raw chess dataset is stored
name_dict: name of dictionary to link actual chess moves with vector components
'''
name_directory = 'dataGM'
name_dict = 'dict_policy.pkl'

def get_memo():
    memo =  {}
    gn = 0
    for fn in os.listdir(name_directory):
        pgn = open(os.path.join(name_directory, fn))
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if game == None:
                    break
                if len(list(game.mainline_moves())) == 0:
                    continue
            except Exception:
                break
            print("parsing game %d . got %d examples" % (gn, len(memo)))
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                if move not in memo:
                    idx = len(memo)
                    memo[move] = idx + 1
                board.push(move)
                s = State(board)
            gn += 1
    return memo



if __name__ == "__main__":
    dict_chess = get_memo()
    with open(name_dict, 'wb') as f:
        pickle.dump(dict_chess, f)
    del dict_chess
