import numpy as np

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0

class QLearner:


    GAME_NUM = 100000

    def __init__(self, alpha=.7, gamma=.9, initial_value=0.5, side=None):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value
        # self.state = ?

    def set_side(self, side):
        self.side = side

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((3, 3))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def _select_best_move(self, board):
        state = board.encode_state()
        q_values = self.Q(state)
        row, col = 0, 0
        curr_max = -np.inf
        while True:
            i, j = self._find_max(q_values)
            if board.is_valid_move(i, j):
                return i, j
            else:
                q_values[i][j] = -1.0

    def _find_max(self, q_values):
        curr_max = -np.inf
        row, col = 0, 0
        for i in range(0, 3):
            for j in range(0, 3):
                if q_values[i][j] > curr_max:
                    curr_max = q_values[i][j]
                    row, col = i, j
        return row, col


    def move(self, board):
        """ make a move
        """
        if board.game_over():
            return
        row, col = self._select_best_move(board)
        self.history_states.append((board.encode_state(), (row, col)))
        return board.move(row, col, self.side)

    def learn(self, board):
        """ when games ended, this method will be called to update the qvalues
        """
        if board.game_result == 0:
            reward = DRAW_REWARD
        elif board.game_result == self.side:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        for hist in self.history_states:
            state, move = hist
            q = self.Q(state)
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            max_q_value = np.max(q)
        self.history_states = []


import sys
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))

from Board import Board
from RandomPlayer import RandomPlayer
from QLearner import QLearner
from PerfectPlayer import PerfectPlayer
from SmartPlayer import SmartPlayer


PLAYER_X = 1
PLAYER_O = 2

def play(board, player1, player2, learn):
    """ Player 1 -> X, X goes first
        player 2 -> O
    """
    player1.set_side(PLAYER_X)
    player2.set_side(PLAYER_O)

    while not board.game_over():
        player1.move(board)
        player2.move(board)

    if learn == True:
        player1.learn(board)
        player2.learn(board)

    return board.game_result



def battle(board, player1, player2, iter, learn=False, show_result=True):
    p1_stats = [0, 0, 0] # draw, win, lose
    for i in range(0, iter):
        result = play(board, player1, player2, learn)
        p1_stats[result] += 1
        board.reset()

    p1_stats = [round(x / iter * 100.0, 1) for x in p1_stats]
    if show_result:
        print('_' * 60)
        print('{:>15}(X) | Wins:{}% Draws:{}% Losses:{}%'.format(player1.__class__.__name__, p1_stats[1], p1_stats[0], p1_stats[2]).center(50))
        print('{:>15}(O) | Wins:{}% Draws:{}% Losses:{}%'.format(player2.__class__.__name__, p1_stats[2], p1_stats[0], p1_stats[1]).center(50))
        print('_' * 60)
        print()

    return p1_stats


if __name__ == "__main__":

    # Example Usage
    # battle(Board(show_board=True, show_result=True), RandomPlayer(), RandomPlayer(), 1, learn=False, show_result=True)
    # battle(Board(), RandomPlayer(), RandomPlayer(), 100, learn=False, show_result=True)
    # battle(Board(), RandomPlayer(), SmartPlayer(), 100, learn=False, show_result=True)
    # battle(Board(), RandomPlayer(), PerfectPlayer(), 100, learn=False, show_result=True)
    # battle(Board(), SmartPlayer(), PerfectPlayer(), 100, learn=False, show_result=True)

    qlearner = QLearner()
    NUM = qlearner.GAME_NUM

    # train: play NUM games against players who only make random moves
    print('Training QLearner against RandomPlayer for {} times......'.format(NUM))
    board = Board()
    battle(board, RandomPlayer(), qlearner, NUM, learn=True, show_result=False)
    battle(board, qlearner, RandomPlayer(), NUM, learn=True, show_result=False)

    # test: play 1000 games against each opponent
    print('Playing QLearner against RandomPlayer for 1000 times......')
    q_rand = battle(board, qlearner, RandomPlayer(), 500)
    rand_q = battle(board, RandomPlayer(), qlearner, 500)
    print('Playing QLearner against SmartPlayer for 1000 times......')
    q_smart = battle(board, qlearner, SmartPlayer(), 500)
    smart_q = battle(board, SmartPlayer(), qlearner, 500)
    print('Playing QLearner against PerfectPlayer for 1000 times......')
    q_perfect = battle(board, qlearner, PerfectPlayer(), 500)
    perfect_q = battle(board, PerfectPlayer(), qlearner, 500)

    # summarize game results
    winning_rate_w_random_player  = round(100 -  (q_rand[2] + rand_q[1]) / 2, 2)
    winning_rate_w_smart_player   = round(100 - (q_smart[2] + smart_q[1]) / 2, 2)
    winning_rate_w_perfect_player = round(100 - (q_perfect[2] + perfect_q[1]) / 2, 2)

    print("Summary:")
    print("_" * 60)
    print("QLearner VS  RandomPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_random_player))
    print("QLearner VS   SmartPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_smart_player))
    print("QLearner VS PerfectPlayer |  Win/Draw Rate = {}%".format(winning_rate_w_perfect_player))
    print("_" * 60)

    grade = 0
    if winning_rate_w_random_player >= 85:
        grade += 25 if winning_rate_w_random_player >= 95 else winning_rate_w_random_player * 0.15
    if winning_rate_w_smart_player >= 85:
        grade += 25 if winning_rate_w_smart_player >= 95 else winning_rate_w_smart_player * 0.15
    if winning_rate_w_perfect_player >= 85:
        grade += 20 if winning_rate_w_perfect_player >= 95 else winning_rate_w_perfect_player * 0.10
    grade = round(grade, 1)

    print("\nTask 2 Grade: {} / 70 \n".format(grade))

#   output_file = sys.argv[1]
#    with open(output_file, 'w') as f:
#        f.write(str(grade) + '\n')
