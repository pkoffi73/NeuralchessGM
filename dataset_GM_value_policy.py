# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:37:46 2019

@author: phili
"""

import os
import chess.pgn
from state import State
import numpy as np
import pickle

'''
CONSTANTS:
- RESUME_GAME: Game number to resume data processing from. if 0, we start from scratch
- INDEX_INIT: File index to start saving processed data (when resuming processing). if 0, we start from scratch
'''
RESUME_GAME = 0
INDEX_INIT = 0


def get_dataset(num_samples=None):
    X, Y1, Y2 = [], [], []
    gn, index = 0, INDEX_INIT
    last_game = 0
    with open(name_dict, 'rb') as f:
        memo = pickle.load(f)
    values = {"1/2-1/2":0, "0-1":-1, "1-0":1}
    for fn in os.listdir("dataGM"):
        pgn = open(os.path.join("dataGM", fn))
        while 1:
            try:
                game = chess.pgn.read_game(pgn)
                if game == None:
                    break
                if len(list(game.mainline_moves())) == 0:
                    continue
            except Exception:
                break
            gn += 1
            if gn <= RESUME_GAME:
                if gn % 1000 == 0:
                    print(gn)
                continue
            print("parsing game %d . got %d examples . last game saved %d" \
                  % (gn, len(X), last_game))
            res = game.headers['Result']
            if res not in values:
                continue
            else:
                value = values[res]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                try:
                    s = State(board)
                    ser = s.bitmap()
                    move_one_hot = [0] * 1917
                    move_one_hot[memo[move]] = 1
                    board.push(move)
                    Y1.append(value)
                    Y2.append(move_one_hot)
                    X.append(ser)
                except:
                    continue
            if num_samples is not None and len(X) > num_samples:
                name_file = 'processedGM6/dataset_3_classes_bitmap' + str(index) +'.npz'
                np.savez(name_file, np.array(X), np.array(Y1), np.array(Y2))
                index += 1
                last_game = gn
                X, Y1, Y2 = [], [], []
            


if __name__ == "__main__":
    SIZE = 1000000
    name_dict = 'dict_policy.pkl'
    get_dataset(SIZE)
