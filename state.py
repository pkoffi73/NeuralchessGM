# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 01:19:37 2019

@author: phili
"""

import chess
import numpy as np

FEATURES = 22

class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
            
    def key(self):
        return (self.board.board_fen(), self.board.turn, \
                self.board.castling_rights, self.board.ep_square)
    

    def bitmap(self):
        assert self.board.is_valid()        
        bstate = np.zeros((64, FEATURES), np.uint8)
        pieces = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                  "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12}
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                for j in range(12):
                    pp = self.board.piece_at(i)
                    if pieces[pp.symbol()] == j + 1:
                        bstate[i, j] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            for i in range(64):
                if i % 8 < 4:
                    bstate[i, 13] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            for i in range(64):
                if i % 8 < 4:
                    bstate[i, 14] = 1
        if self.board.has_kingside_castling_rights(chess.WHITE):
            for i in range(64):
                if i % 8 >= 4:
                    bstate[i, 13] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            for i in range(64):
                if i % 8 >= 4:
                    bstate[i, 14] = 1
        central_board = [26, 27, 28, 34, 35, 36]
        countWHITE = 0
        countBLACK = 0
        for i in central_board:
            if self.board.color_at(i):
                countWHITE += 1
            elif not self.board.color_at(i) and self.board.piece_at(i):
                countBLACK += 1
        if countWHITE > countBLACK:
            for i in central_board:
                bstate[i, 15] = 1
        elif countBLACK > countWHITE:
            for i in central_board:
                bstate[i, 16] = 1
        for i in range(64):
            squaresetWHITE = self.board.pin(True, i)
            if len(squaresetWHITE) < 64:
                for j in squaresetWHITE:
                    bstate[j, 18] = 1
                break
        for i in range(64):
            squaresetBLACK = self.board.pin(False, i)
            if len(squaresetBLACK) < 64:
                for j in squaresetBLACK:
                    bstate[j, 19] = 1
                break
        for i in range(64):
            if self.board.is_attacked_by(chess.WHITE, i):
                bstate[i, 20] = 1
            if self.board.is_attacked_by(chess.BLACK, i):
                bstate[i, 21] = 1
        bstate1 = np.zeros((8, 8, FEATURES), np.uint8)
        for i in range(FEATURES):
            bstate1[:,:,i]=bstate[:,i].reshape(8, 8)
        bstate1[:, :, 12] = (self.board.turn*1.0)
        if self.board.is_check():
            bstate1[:, :, 17] = 1
        return bstate1         
    
if __name__ == '__main__':
    s = State()
