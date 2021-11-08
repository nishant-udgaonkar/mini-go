import math
from collections import deque
from functools import lru_cache

import numpy
import os
import random
import time
import signal
import json

BLANK = '0'
P1 = '1'
P2 = '2'

point_idx_mapping = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3,
    (0, 4): 4,
    (1, 0): 5,
    (1, 1): 6,
    (1, 2): 7,
    (1, 3): 8,
    (1, 4): 9,
    (2, 0): 10,
    (2, 1): 11,
    (2, 2): 12,
    (2, 3): 13,
    (2, 4): 14,
    (3, 0): 15,
    (3, 1): 16,
    (3, 2): 17,
    (3, 3): 18,
    (3, 4): 19,
    (4, 0): 20,
    (4, 1): 21,
    (4, 2): 22,
    (4, 3): 23,
    (4, 4): 24,
}

idx_point_mapping = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (0, 3),
    4: (0, 4),
    5: (1, 0),
    6: (1, 1),
    7: (1, 2),
    8: (1, 3),
    9: (1, 4),
    10: (2, 0),
    11: (2, 1),
    12: (2, 2),
    13: (2, 3),
    14: (2, 4),
    15: (3, 0),
    16: (3, 1),
    17: (3, 2),
    18: (3, 3),
    19: (3, 4),
    20: (4, 0),
    21: (4, 1),
    22: (4, 2),
    23: (4, 3),
    24: (4, 4),
}


@lru_cache(256)
def get_other_piece(piece):
    if piece == P2:
        return P1

    return P2


def point_to_array_index(x, y):
    """
    Convert an (x, y) point to the index in our 1-D board representation.
    """
    return point_idx_mapping[(x, y)]


def array_index_to_point(idx):
    """
    Convert an index in our 1-D board to an (x, y) point.
    """
    if idx == -1:
        return -1, -1
    return idx_point_mapping[idx]


@lru_cache(256)
def valid_point(pt):
    """
    Check if ``pt`` is a valid point.
    """
    i, j = pt
    return 0 <= i < 5 and 0 <= j < 5


@lru_cache(256)
def get_neighbors(idx):
    """
    Find all the neighbors of a given point. Input and output from this
    function is in terms of 1-D board indices.
    """
    x, y = array_index_to_point(idx)

    potential_neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    neighbors = [point_to_array_index(*neighbor)
                 for neighbor in potential_neighbors
                 if valid_point(neighbor)]

    return neighbors


@lru_cache(3)
def get_corner_locations():
    corner_points = [(0, 0), (0, 4), (4, 0), (4, 4)]
    return [point_to_array_index(*pt) for pt in corner_points]


@lru_cache(3)
def get_side_locations():
    side_points = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 4),
        (2, 0), (2, 4),
        (3, 0), (3, 4),
        (4, 1), (4, 2), (4, 3)
    ]
    return [point_to_array_index(*pt) for pt in side_points]


@lru_cache(256)
def get_surroundings(idx):
    x, y = array_index_to_point(idx)

    candidates = [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
        (x + 1, y + 1),
        (x + 1, y - 1),
        (x - 1, y + 1),
        (x - 1, y - 1)
    ]

    candidates = [pt for pt in candidates if valid_point(pt)]
    candidates = [point_to_array_index(*pt) for pt in candidates]

    return candidates


def get_group_and_reachable(board, idx, piece=None):
    """
    1. Return all indices that are connected to ``idx`` by stones of the same
       color.
    2. Return all indices that are "reachable" from ``idx`` using the
       definition from the Tromp-Taylor rules at
       http://tromp.github.io/go.html.
    """
    if piece is None:
        piece = board[idx]

    group = []
    reachable = []

    frontier = deque([idx])
    already_seen = set()

    while frontier:
        i = frontier.popleft()

        group.append(i)
        already_seen.add(i)

        neighbors = get_neighbors(i)
        for neighbor in neighbors:
            if neighbor in already_seen:
                continue
            if board[neighbor] == piece:
                frontier.append(neighbor)
            else:
                reachable.append(neighbor)

    return group, reachable


def is_blank(board, idx):
    """
    Check if a particular location on the board is BLANK.
    """
    return board[idx] == BLANK


def get_num_captures(board, idx, piece):
    """
    Check if putting ``piece`` at ``idx`` enables a capture.
    """
    other_piece = get_other_piece(piece)

    neighbors = get_neighbors(idx)
    to_examine = []

    for neighbor in neighbors:
        if board[neighbor] == other_piece:
            to_examine.append(neighbor)

    num_captures = 0

    for neighbor in to_examine:
        # Find out what can be reached from the neighbors that are the
        # opponent's pieces
        group, reachable = get_group_and_reachable(board, neighbor)
        if idx in reachable:
            reachable.remove(idx)

        # If they do not have any liberties, the entire group can be captured
        if not any(board[reach] == BLANK for reach in reachable):
            num_captures += len(group)

    return num_captures


def is_suicide(board, idx, piece):
    """
    Check if a move is suicide.
    """
    neighbors = get_neighbors(idx)
    other_piece = get_other_piece(piece)

    # Check if surrounded by enemy pieces that cannot be captured
    if all(board[neighbor] == other_piece for neighbor in neighbors):
        if get_num_captures(board, idx, piece) == 0:
            return True

    # Check if capturing own pieces
    temp_board = place_piece(board, idx, piece)
    temp_board = perform_captures(temp_board, idx, piece)
    if not get_liberties_at_location(temp_board, idx):
        return True

    return False


@lru_cache(256)
def place_piece(board, idx, piece):
    """
    Return a new board with ``piece`` played on ``idx``.
    """
    # -1 is a PASS move
    if idx == -1:
        return board

    return board[:idx] + piece + board[idx + 1:]


def check_valid_move(board, old_board, idx, piece):
    """
    Check if a move is valid.
    """
    # -1 denotes a PASS move; we do not need to check anything else
    if idx == -1:
        return True

    # Make sure the space is BLANK
    if not is_blank(board, idx):
        return False

    # Check if suicide
    if is_suicide(board, idx, piece):
        return False

    # Check for KO rule
    new_board = place_piece(board, idx, piece)
    new_board = perform_captures(new_board, idx, piece)
    if new_board == old_board and idx != -1:
        return False

    return True


def get_liberties_at_location(board, idx):
    """
    Find all the liberties of a piece placed at ``idx``.
    """
    _, reachable = get_group_and_reachable(board, idx)
    liberties = [reach for reach in reachable if board[reach] == BLANK]

    return liberties


def get_opponent_liberties_at_location(board, idx, piece):
    """
    Find all the liberties of a piece placed at ``idx``.
    """
    piece = get_other_piece(piece)
    _, reachable = get_group_and_reachable(board, idx, piece)
    liberties = [reach for reach in reachable if board[reach] == BLANK]

    return liberties


def perform_captures(board, idx, piece):
    """
    Perform all captures when placing ``piece`` at ``idx`` and return the
    subsequent board.
    """
    other_piece = get_other_piece(piece)

    neighbors = get_neighbors(idx)
    to_examine = []

    for neighbor in neighbors:
        if board[neighbor] == other_piece:
            to_examine.append(neighbor)

    next_board = list(board)

    for neighbor in to_examine:
        # Find out what can be reached from the neighbors that are the
        # opponent's pieces
        group, reachable = get_group_and_reachable(board, neighbor)
        if idx in reachable:
            reachable.remove(idx)

        # If they do not have any liberties, the entire group can be captured
        if not any(board[reach] == BLANK for reach in reachable):
            for idx in group:
                next_board[idx] = BLANK

    return ''.join(next_board)


def get_valid_moves(board, old_board, piece):
    """
    Find all valid locations to put ``piece``.
    """
    moves = []

    for idx, item in enumerate(board):
        if item != BLANK:
            continue

        if not check_valid_move(board, old_board, idx, piece):
            continue

        moves.append(idx)

    return moves


def get_valid_moves_with_pass(board, previous_board, piece):
    moves = get_valid_moves(board, previous_board, piece)

    if board != previous_board:
        moves.append(-1)

    return moves


def score(board):
    """
    Return the score of the position.
    """
    return board.count(P1), board.count(P2) + 2.5


def score_diff(board, piece):
    """
    Return the difference of scores.
    """
    scores = score(board)
    other_piece = get_other_piece(piece)

    return scores[int(piece) - 1] - scores[int(other_piece) - 1]


def get_all_liberties(board, piece):
    """
    Return the liberties for a player.
    """
    liberties = set()

    for idx, item in enumerate(board):
        if item == piece:
            _, reachable = get_group_and_reachable(board, idx)
            reachable = [reach for reach in reachable if board[reach] == BLANK]
            liberties.update(reachable)

    return liberties


def get_locations(board, piece):
    locations = []
    for idx, item in enumerate(board):
        if item == piece:
            locations.append(idx)
    return locations


def display_board(board):
    """
    Print the board.
    """
    for idx, item in enumerate(board):
        if (idx + 1) % 5 == 0:
            end = '\n'
        else:
            end = ' '

        if item == '1':
            item = 'B'
        elif item == '2':
            item = 'W'
        else:
            item = '-'

        print(item, end=end)


def play_move(board, old_board, idx, piece):
    """
    Play a move, including performing any captures.
    """
    # Check if valid move
    if not check_valid_move(board, old_board, idx, piece):
        return False

    # -1 is a PASS move; the board does not change
    if idx == -1:
        return board

    # Place piece
    board = place_piece(board, idx, piece)

    # Perform captures
    board = perform_captures(board, idx, piece)

    # Return the new board after playing the move
    return board


def read_board_from_file():
    """
    Read the input from input.txt.
    """
    with open('input.txt', 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    piece = lines[0]
    previous_state = ''.join(lines[1:6])
    current_state = ''.join(lines[6:])

    return piece, previous_state, current_state


def write_move_to_file(idx):
    """
    Write the move to the output file.
    """
    if idx == -1:
        output = 'PASS'
    else:
        x, y = array_index_to_point(idx)
        output = '{},{}'.format(x, y)

    with open('output.txt', 'w') as f:
        f.write(output)


def read_move_ctr_from_file():
    if not os.path.exists('move_ctr.txt'):
        return 0

    with open('move_ctr.txt', 'r') as f:
        lines = f.readlines()

    return int(lines[0].strip())


def write_move_ctr_to_file(ctr):
    with open('move_ctr.txt', 'w') as f:
        f.write('{}'.format(ctr))


@lru_cache(256)
def flip_piece_colors(board):
    board_flipped = list(board)
    for i in range(len(board)):
        if board_flipped[i] == '1':
            board_flipped[1] = '2'
        elif board_flipped[i] == '2':
            board_flipped[i] = '1'

    return ''.join(board_flipped)


def get_similar_boards(board, piece):
    other_piece = get_other_piece(piece)
    result = [(board, piece)]

    b1 = flip_piece_colors(board)
    result.append((b1, other_piece))

    b2d = [list(board[:5]), list(board[5:10]), list(board[10:15]),
           list(board[15:20]), list(board[20:])]

    b2d_90 = numpy.rot90(b2d)

    b2 = ''.join([''.join(b) for b in b2d_90])
    result.append((b2, piece))

    b3 = flip_piece_colors(b2)
    result.append((b3, other_piece))

    b2d_180 = numpy.rot90(b2d_90)

    b4 = ''.join([''.join(b) for b in b2d_180])
    result.append((b4, piece))

    b5 = flip_piece_colors(b4)
    result.append((b5, other_piece))

    b2d_270 = numpy.rot90(b2d_180)

    b6 = ''.join([''.join(b) for b in b2d_270])
    result.append((b6, piece))

    b7 = flip_piece_colors(b6)
    result.append((b7, other_piece))

    return result


def similar_boards(board1, board2):
    """
    Check if the 2 input boards are the same, just with colors exchanged or
    with the board reflected.
    """
    # Same board
    if board1 == board2:
        return True

    # Check if flipping colors gives the same board
    board2_flipped = flip_piece_colors(board2)

    if board2_flipped == board1:
        return True

    # Flip on X axis
    board2_fx = (board2[20:] + board2[15:20] + board2[10:15] + board2[10:15] +
                 board2[:5])

    if board2_fx == board1:
        return True

    if flip_piece_colors(board2_fx) == board1:
        return True

    # Flip on Y axis
    board2_fy = [array_index_to_point(i) for i in board2]
    for i in range(len(board2_fy)):
        x, y = board2_fy[i]
        board2_fy[i] = (x, 4 - y)
    board2_fy = [point_to_array_index(x, y) for x, y in board2_fy]

    if board2_fy == board1:
        return True

    if flip_piece_colors(board2_fy) == board1:
        return True

    # Flip on both axes
    board2_fxy = [array_index_to_point(i) for i in board2]
    for i in range(len(board2_fxy)):
        x, y = board2_fxy[i]
        board2_fxy[i] = (4 - x, 4 - y)
    board2_fxy = [point_to_array_index(x, y) for x, y in board2_fxy]

    if board2_fxy == board1:
        return True

    if flip_piece_colors(board2_fxy) == board1:
        return True

    # Flip board 90 degrees
    board2_2d = [board2[:5], board2[5:10], board2[10:15], board2[15:20],
                 board2[20:]]

    board2_90_2d = numpy.rot90(board2_2d)
    board2_90 = (board2_90_2d[0] + board2_90_2d[1] + board2_90_2d[2] +
                 board2_90_2d[3] + board2_90_2d[4])

    if board2_90 == board1:
        return True

    if flip_piece_colors(board2_90) == board1:
        return True

    # Flip the board 180 degrees
    board2_180_2d = numpy.rot90(board2_90_2d)
    board2_180 = (board2_180_2d[0] + board2_180_2d[1] + board2_180_2d[2] +
                  board2_180_2d[3] + board2_180_2d[4])

    if board2_180 == board1:
        return True

    if flip_piece_colors(board2_180) == board1:
        return True

    # Flip the board 270 degrees
    board2_270_2d = numpy.rot90(board2_180_2d)
    board2_270 = (board2_270_2d[0] + board2_270_2d[1] + board2_270_2d[2] +
                  board2_270_2d[3] + board2_270_2d[4])

    if board2_270 == board1:
        return True

    if flip_piece_colors(board2_270) == board1:
        return True

    return False


def idx_distance(idx1, idx2):
    p1 = array_index_to_point(idx1)
    p2 = array_index_to_point(idx2)

    x_diff = (p1[0] - p2[0]) ** 2
    y_diff = (p1[1] - p2[1]) ** 2

    return math.sqrt(x_diff + y_diff)


class Player(object):
    def __init__(self, piece, old_board, board, bdata=None, wdata=None):
        self.piece = piece
        self.old_board = old_board
        self.board = board
        self.best_moves = [-1]
        self.bdata = bdata
        self.wdata = wdata

        signal.signal(signal.SIGALRM, self.alarm_handler)
        signal.alarm(9)

    def select_move(self, move_number):
        raise NotImplemented

    def play_move(self, move_number, verbose=False):
        move = self.select_move(move_number)

        new_board = play_move(self.board, self.old_board, move, self.piece)

        self.old_board = self.board
        self.board = new_board

        if verbose:
            print('{} {}, move {} - {}'.format(self.__class__.__name__,
                                             self.piece, move_number,
                                               array_index_to_point(move)))

        signal.alarm(0)
        return move

    def set_next_board(self, board):
        self.old_board = self.board
        self.board = board

    def alarm_handler(self, signum, frame):
        print('alarm from {}'.format(self.__class__.__name__))
        return self.choose_preferred_move(self.best_moves)

    def choose_preferred_move(self, moves):
        ordered_moves = [
            12, 6, 8, 7, 18,
            13, 16, 17, 11, 1,
            3, 5, 9, 15, 19,
            21, 23, 2, 10, 14,
            22, 0, 4, 20, 24, -1
        ]

        for move in ordered_moves:
            if move in moves:
                return move


class RandomPlayer(Player):
    def select_move(self, move_number):
        moves = get_valid_moves(self.board, self.old_board, self.piece)
        moves.append(-1)

        return random.choice(moves)


# class MinLibertyPlayer(Player):
#     def select_move(self, move_number):
#         moves = get_valid_moves(self.board, self.piece)
#
#         if self.board != self.old_board:
#             moves.append(-1)
#
#         min_liberty = 100
#         min_liberty_move = -1
#         for move in moves:
#             board = place_piece(self.board, move, self.piece)
#             liberty = len(get_all_liberties(board, self.piece))
#             if liberty < min_liberty:
#                 min_liberty = liberty
#                 min_liberty_move = move
#
#         return min_liberty_move


# class MaxLibertyPlayer(Player):
#     def select_move(self, move_number):
#         moves = get_valid_moves(self.board, self.piece)
#
#         if self.board != self.old_board:
#             moves.append(-1)
#
#         max_liberty = -1
#         max_liberty_move = -1
#         for move in moves:
#             board = place_piece(self.board, move, self.piece)
#             liberty = len(get_all_liberties(board, self.piece))
#             if liberty > max_liberty:
#                 max_liberty = liberty
#                 max_liberty_move = move
#
#         return max_liberty_move


# class CentralPlayer(Player):
#     def select_move(self, move_number):
#         moves = get_valid_moves(self.board, self.piece)
#
#         if not moves:
#             return -1
#
#         middle = 12
#         diffs = [abs(move - middle) for move in moves]
#
#         return moves[numpy.argmin(diffs)]


class ManualPlayer(Player):
    def select_move(self, move_number):
        move = input('Play move: ')

        if move == 'exit':
            exit()

        move = list(map(int, move.split()))

        if move[0] == -1:
            return -1

        return point_to_array_index(*move)


# class CaptureOrCentralPlayer(Player):
#     def select_move(self, move_number):
#         moves = get_valid_moves(self.board, self.piece)
#
#         if not moves:
#             return -1
#
#         for move in moves:
#             if get_num_captures(self.board, move, self.piece) > 0:
#                 return move
#
#         middle = 12
#         diffs = [abs(move - middle) for move in moves]
#
#         return moves[numpy.argmin(diffs)]


# class BiggestCaptureOrCentralPlayer(Player):
#     def select_move(self, move_number):
#         moves = get_valid_moves(self.board, self.piece)
#
#         if not moves:
#             return -1
#
#         best_capture_score = 0
#         best_capture_move = -2
#
#         for move in moves:
#             num_captures = get_num_captures(self.board, move, self.piece)
#             if num_captures > best_capture_score:
#                 best_capture_score = num_captures
#                 best_capture_move = move
#
#         if best_capture_move != -2:
#             return best_capture_move
#
#         middle = 12
#         diffs = [abs(move - middle) for move in moves]
#
#         return moves[numpy.argmin(diffs)]


# class MinimaxPlayer(Player):
#     def select_move(self, move_number):
#         t = time.time()
#
#         moves = get_valid_moves(self.board, self.piece)
#
#         if not moves:
#             return -1
#
#         boards = [play_move(self.board, self.old_board, move, self.piece) for
#                   move in moves]
#         scores = [self.minimizer(board, 2, move_number, t) for board in boards]
#
#         return moves[numpy.argmax(scores)]
#
#     def maximizer(self, board, depth, move_number, start_time):
#         moves = get_valid_moves(board, self.piece)
#
#         if not moves:
#             return -100
#
#         boards = [play_move(board, old_boardmove, self.piece) for move in moves]
#
#         if depth == 0 or time.time() - start_time > 8.7 or move_number == 24:
#             scores = [score_diff(board, self.piece) for board in boards]
#         else:
#             scores = [self.minimizer(board, depth - 1, move_number + 1,
#                                      start_time)
#                       for board in boards]
#
#         return max(scores)
#
#     def minimizer(self, board, depth, move_number, start_time):
#         other_piece = get_other_piece(self.piece)
#         moves = get_valid_moves(board, other_piece)
#
#         if not moves:
#             return 100
#
#         boards = [play_move(board, move, other_piece) for move in moves]
#
#         if depth == 0 or time.time() - start_time > 8.7 or move_number == 24:
#             scores = [score_diff(board, other_piece) for board in boards]
#         else:
#             scores = [self.maximizer(board, depth - 1, move_number + 1,
#                                      start_time)
#                       for board in boards]
#
#         return max(scores)


# class CaptureOrCentralThenMinimaxPlayer(Player):
#     def maximizer(self, board, depth, move_number, start_time):
#         moves = get_valid_moves(board, self.piece)
#
#         if not moves:
#             return -100
#
#         boards = [play_move(board, move, self.piece) for move in moves]
#
#         if depth == 0 or time.time() - start_time > 8.7 or move_number == 24:
#             scores = [score_diff(board, self.piece) for board in boards]
#         else:
#             scores = [self.minimizer(board, depth - 1, move_number + 1,
#                                      start_time)
#                       for board in boards]
#
#         return max(scores)
#
#     def minimizer(self, board, depth, move_number, start_time):
#         other_piece = get_other_piece(self.piece)
#         moves = get_valid_moves(board, other_piece)
#
#         if not moves:
#             return 100
#
#         boards = [play_move(board, move, other_piece) for move in moves]
#
#         if depth == 0 or time.time() - start_time > 8.7 or move_number == 24:
#             scores = [score_diff(board, other_piece) for board in boards]
#         else:
#             scores = [self.maximizer(board, depth - 1, move_number + 1,
#                                      start_time)
#                       for board in boards]
#
#         return max(scores)
#
#     def capture_or_central_move(self):
#         moves = get_valid_moves(self.board, self.piece)
#
#         if not moves:
#             return -1
#
#         for move in moves:
#             if get_num_captures(self.board, move, self.piece) > 0:
#                 return move
#
#         points = [array_index_to_point(move) for move in moves]
#         x_diffs = [(pt[0] - 2) ** 2 for pt in points]
#         y_diffs = [(pt[1] - 2) ** 2 for pt in points]
#         diffs = [math.sqrt(x_diff + y_diff)
#                  for x_diff, y_diff in zip(x_diffs, y_diffs)]
#
#         return moves[numpy.argmin(diffs)]
#
#     def select_move(self, move_number):
#         if move_number < 10:
#             return self.capture_or_central_move()
#         else:
#             t = time.time()
#
#             moves = get_valid_moves(self.board, self.piece)
#
#             if not moves:
#                 return -1
#
#             boards = [play_move(self.board, move, self.piece) for move in moves]
#             scores = [self.minimizer(board, 20, move_number, t) for board in
#                       boards]
#
#             return moves[numpy.argmax(scores)]


class AlphaBetaPlayer(Player):
    def select_move(self, move_number):
        t = time.time()

        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in
                  moves]
        scores = [self.minimizer(board, self.board, 4, move_number, t, -100,
                                 100)
                  for board in boards]

        max_score = max(scores)
        if scores.count(max_score) > 1:
            corners = get_corner_locations()
            max_moves = [move
                         for move, score in zip(moves, scores)
                         if score == max_score and move not in corners]
            if max_moves:
                return random.choice(max_moves)

        return moves[numpy.argmax(scores)]

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 24:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 24:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class CaptureOrCentralThenAlphaBetaPlayer(AlphaBetaPlayer):
    def capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        for move in moves:
            if get_num_captures(self.board, move, self.piece) > 0:
                return move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        if move_number < 6:
            return self.capture_or_central_move()
        else:
            t = time.time()

            moves = self.get_moves_to_check(self.board, self.old_board,
                                            self.piece)

            if not moves:
                return -1

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in
                      moves]
            scores = [self.minimizer(board, self.board, 3, move_number, t,
                                     -100, 100)
                      for board in boards]

            return moves[numpy.argmax(scores)]


class CentralThenAlphaBetaPlayer(AlphaBetaPlayer):
    def central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        if move_number < 7:
            return self.central_move()
        else:
            t = time.time()

            moves = get_valid_moves(self.board, self.old_board, self.piece)

            if not moves:
                return -1

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in
                      moves]
            scores = [self.minimizer(board, self.board, 20, move_number, t,
                                     -100, 100)
                      for board in boards]

            return moves[numpy.argmax(scores)]


class BiggestCaptureOrCentralThenAlphaBetaPlayer(AlphaBetaPlayer):
    def biggest_capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        best_capture_score = 0
        best_capture_move = -2

        for move in moves:
            num_captures = get_num_captures(self.board, move, self.piece)
            if num_captures > best_capture_score:
                best_capture_score = num_captures
                best_capture_move = move

        if best_capture_move != -2:
            return best_capture_move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        if move_number < 7:
            return self.biggest_capture_or_central_move()
        else:
            t = time.time()

            moves = get_valid_moves(self.board, self.old_board, self.piece)

            if not moves:
                return -1

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in
                      moves]
            scores = [self.minimizer(board, self.board, 20, move_number, t,
                                     -100, 100)
                      for board in boards]

            return moves[numpy.argmax(scores)]


class Idea1AlphaBetaPlayer(AlphaBetaPlayer):
    def remove_side_moves(self, moves):
        side_locations = get_side_locations()
        other_moves = [move for move in moves if move not in side_locations]

        if other_moves:
            return other_moves
        else:
            return moves

    def choose_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)
        moves = self.remove_side_moves(moves)

        return random.choice(moves)

    def select_move(self, move_number):
        if move_number < 7:
            return self.choose_move()
        else:
            t = time.time()

            moves = get_valid_moves(self.board, self.old_board, self.piece)

            if not moves:
                return -1

            moves = self.remove_side_moves(moves)

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in
                      moves]
            scores = [self.minimizer(board, self.board, 20, move_number, t,
                                     -100, 100)
                      for board in boards]

            return moves[numpy.argmax(scores)]


class ConnectedPlayer(CaptureOrCentralThenAlphaBetaPlayer):
    def get_moves_to_check(self, board, old_board, piece):
        if piece == '1':
            moves = get_all_liberties(board, piece)
            moves = [move for move in moves if check_valid_move(board,
                                                                old_board, move,
                                                                piece)]
        else:
            moves = get_valid_moves_with_pass(board, old_board, piece)

        new_moves = []
        multi_capture_moves = []
        for move in moves:
            if get_num_captures(board, move, piece) > 1:
                multi_capture_moves.append(move)
                continue

            surroundings = get_surroundings(move)
            if any(board[idx] == piece for idx in surroundings):
                new_moves.append(move)

        if multi_capture_moves:
            return multi_capture_moves

        return new_moves


class Idea2Player(AlphaBetaPlayer):
    def select_move(self, move_number):
        if self.piece == P1 and move_number < 2:
            return 12

        if move_number < 8:
            other_piece = get_other_piece(self.piece)
            group, reachable = get_group_and_reachable(self.board, 12)
            moves = get_valid_moves(self.board, self.old_board, self.piece)

            for r in reachable:
                if r not in moves:
                    continue
                neighbors = get_neighbors(r)
                for r in reachable:
                    if get_num_captures(self.board, r, self.piece) > 0:
                        return r

            m = random.choice(reachable)
            while m not in moves:
                m = random.choice(reachable)
            return m
        else:
            t = time.time()

            moves = get_valid_moves(self.board, self.old_board, self.piece)

            if not moves:
                return -1

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in moves]
            scores = [self.minimizer(board, self.board, 20, move_number, t,
                                     -100, 100)
                      for board in boards]

            return moves[numpy.argmax(scores)]


class Idea3Player(AlphaBetaPlayer):
    def select_move(self, move_number):
        t = time.time()

        # Play in the middle if possible
        if move_number < 2 and self.board[12] == BLANK:
            return 12

        moves = get_valid_moves(self.board, self.old_board, self.piece)

        # Play a capture if needed and if possible
        if score_diff(self.board, self.piece) < 3:
            for move in moves:
                if get_num_captures(self.board, move, self.piece) > 0:
                    return move

        # Play strategic opening moves
        if len(moves) > 20:
            opening_moves = [12, 6, 16, 8, 18]
            for m in opening_moves:
                if m in moves:
                    return m

        # Reduce opponents liberty
        cur_liberties = len(get_all_liberties(self.board,
                                              get_other_piece(self.piece)))
        min_liberty = cur_liberties
        min_move = -2
        for move in moves:
            liberties = get_opponent_liberties_at_location(self.board, move,
                                                           self.piece)
            if len(liberties) < min_liberty:
                min_liberty = len(liberties)
                min_move = move

        if cur_liberties - min_liberty > 2:
            return min_move

        # Minimax
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        scores = [self.minimizer(board, self.board, 20, move_number, t,
                                 -100, 100)
                  for board in boards]

        return moves[numpy.argmax(scores)]


class BlackPlayer(Player):
    def get_moves_to_check(self, board, old_board, piece, move_number):
        if (move_number > 2 and board == old_board and
                score_diff(board, piece) > 0):
            return [-1]

        reachable = get_all_liberties(board, piece)
        to_examine = []

        for move in reachable:
            surroundings = get_surroundings(move)
            if any(board[surrounding] == piece for surrounding in surroundings):
                to_examine.append(move)

        to_examine = [move for move in to_examine
                      if check_valid_move(board, old_board, move, piece)]

        multi_capture_moves = []
        for move in to_examine:
            if get_num_captures(board, move, piece) > 1:
                multi_capture_moves.append(move)

        if multi_capture_moves:
            return multi_capture_moves

        return to_examine

    def select_move(self, move_number):
        if move_number == 0:
            return 12

        other_piece = get_other_piece(self.piece)

        if move_number == 2:
            if other_piece in (self.board[7], self.board[13]):
                return 8
            if other_piece in (self.board[17], self.board[11]):
                return 16

        t = time.time()

        moves = self.get_moves_to_check(self.board, self.old_board,
                                        self.piece, move_number)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        scores = [self.minimizer(board, self.board, 6, move_number, t, -100,
                                 100)
                  for board in boards]

        max_score = max(scores)

        if scores.count(max_score) > 1:
            corners = get_corner_locations()
            max_moves = [move
                         for move, score in zip(moves, scores)
                         if score == max_score and move not in corners]
            if max_moves:
                if 12 in max_moves:
                    return 12
                return random.choice(max_moves)

        return moves[numpy.argmax(scores)]

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece,
                                        move_number)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece,
                                        move_number)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer2(Player):
    def get_moves_to_check_min(self, board, old_board, piece, move_number):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves

    def get_moves_to_check_max(self, board, old_board, piece, move_number):
        if (move_number > 2 and board == old_board and
                score_diff(board, piece) > 0):
            return [-1]

        moves = get_valid_moves(board, old_board, piece)

        max_captures = 0
        max_capture_moves = []
        for move in moves:
            captures = get_num_captures(board, move, piece)
            if captures > max_captures:
                max_captures = captures
            if captures > 0 and captures == max_captures:
                max_capture_moves.append(move)
        if max_capture_moves:
            return max_capture_moves

        other_piece = get_other_piece(piece)
        if move_number < 6:
            diffs = []

            other_locations = get_locations(board, other_piece)
            central_moves = [6, 7, 8, 11, 13, 16, 17, 18]
            central_moves = [cm for cm in central_moves if cm in moves]

            for move in central_moves:
                diffs.append(min(idx_distance(move, idx)
                                 for idx in other_locations))

            closest = min(diffs)
            return [move for move, diff in zip(central_moves, diffs)
                    if diff == closest]

        surrounding_moves = []
        for move in moves:
            surroundings = get_surroundings(move)
            if any(board[surrounding] == piece
                   for surrounding in surroundings):
                surrounding_moves.append(move)
        if surrounding_moves:
            return surrounding_moves

        return moves

    def select_move(self, move_number):
        if move_number == 0:
            return 12

        t = time.time()

        moves = self.get_moves_to_check_max(self.board, self.old_board,
                                            self.piece, move_number)

        if not moves:
            return -1

        depth = 4
        if move_number > 18:
            depth = 10
        elif move_number > 16:
            depth = 7
        elif move_number > 10:
            depth = 6

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]

        max_score = -100
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number, t,
                                   -100, 100)
            if score > max_score:
                max_score = score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        if len(self.best_moves) == 1:
            return self.best_moves[0]
        else:
            corners = get_corner_locations()
            not_corners = [move for move in self.best_moves
                           if move not in corners]
            if not_corners:
                diffs = [idx_distance(12, move) for move in not_corners]
            else:
                diffs = [idx_distance(12, move) for move in self.best_moves]

            return self.best_moves[numpy.argmin(diffs)]

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check_max(board, old_board, self.piece,
                                            move_number)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, self.old_board, move, self.piece) for move
                  in moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check_min(board, old_board, other_piece,
                                            move_number)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer3(Player):
    def get_moves_to_check_min(self, board, old_board, piece, move_number):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves

    def get_moves_to_check_max(self, board, old_board, piece, move_number):
        if (move_number > 2 and board == old_board and
                score_diff(board, piece) > 0):
            return [-1]

        moves = get_valid_moves(board, old_board, piece)

        max_captures = 0
        max_capture_moves = []
        for move in moves:
            captures = get_num_captures(board, move, piece)
            if captures > max_captures:
                max_captures = captures
            if captures > 0 and captures == max_captures:
                max_capture_moves.append(move)
        if max_capture_moves:
            return max_capture_moves

        other_piece = get_other_piece(piece)
        if move_number < 6:
            diffs = []

            other_locations = get_locations(board, other_piece)
            if other_locations:
                central_moves = [6, 7, 8, 11, 13, 16, 17, 18]
                central_moves = [cm for cm in central_moves if cm in moves]

                for move in central_moves:
                    diffs.append(min(idx_distance(move, idx)
                                     for idx in other_locations))

                closest = min(diffs)
                return [move for move, diff in zip(central_moves, diffs)
                        if diff == closest]

        surrounding_moves = []
        for move in moves:
            surroundings = get_surroundings(move)
            if any(board[surrounding] == piece
                   for surrounding in surroundings):
                surrounding_moves.append(move)
        if surrounding_moves:
            return surrounding_moves

        return moves

    def select_move(self, move_number):
        if move_number == 0:
            return 12

        t = time.time()

        moves = self.get_moves_to_check_max(self.board, self.old_board,
                                            self.piece, move_number)

        if not moves:
            return -1

        depth = 4
        if move_number == 8:
            depth = 2
        elif move_number == 14:
            depth = 5
        elif move_number > 12:
            if len(moves) < 8:
                depth = 6
            if len(moves) < 12:
                depth = 7

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]

        max_score = -100
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number, t,
                                   -100, 100)
            if score > max_score:
                max_score = score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        if len(self.best_moves) == 1:
            return self.best_moves[0]
        else:
            corners = get_corner_locations()
            not_corners = [move for move in self.best_moves
                           if move not in corners]
            if not_corners:
                diffs = [idx_distance(12, move) for move in not_corners]
            else:
                diffs = [idx_distance(12, move) for move in self.best_moves]

            return self.best_moves[numpy.argmin(diffs)]

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check_max(board, old_board, self.piece,
                                            move_number)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check_min(board, old_board, other_piece,
                                            move_number)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer4(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        scores = [self.minimizer(board, self.board, 5, move_number, t, -100,
                                 100)
                  for board in boards]

        max_score = max(scores)
        max_moves = [move for move, score in zip(moves, scores)
                     if score == max_score]
        return random.choice(max_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)

        moves.sort(key=lambda move: get_num_captures(board, move, piece),
                   reverse=True)

        return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 9 or move_number >= 24:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 9 or move_number >= 24:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer5(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        # if 6 < move_number < 14:
        #     with open('b_5.json', 'r') as f:
        #         data = json.load(f)
        #     if self.board in data:
        #         alpha = data[self.board][0]
        #         alpha_move = data[self.board][1]

        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        max_captures = max(num_captures)
        if max_captures > 0:
            for idx in range(len(num_captures)):
                if num_captures[idx] == max_captures:
                    return moves[idx]

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        max_score = -100
        depth = 4
        if move_number > 15:
            depth = 6
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        # if max_score == alpha:
        #     return alpha_move

        return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)

        moves.sort(key=lambda move: get_num_captures(board, move, piece),
                   reverse=True)

        return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        max_captures = max(num_captures)
        if max_captures > 0:
            for idx in range(len(num_captures)):
                if num_captures[idx] == max_captures:
                    return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer6(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        # if 6 < move_number < 14:
        #     with open('b_5.json', 'r') as f:
        #         data = json.load(f)
        #     if self.board in data:
        #         alpha = data[self.board][0]
        #         alpha_move = data[self.board][1]

        moves = self.get_moves_to_check(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece)
                  for move in moves]
        max_score = -100
        depth = 4
        if move_number > 15:
            depth = 6
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        # if max_score == alpha:
        #     return alpha_move

        return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)

        max_captures = 0
        max_capture_moves = []
        for move in moves:
            num_captures = get_num_captures(board, move, piece)
            if num_captures  > max_captures:
                max_captures = num_captures
                max_capture_moves = []
            if num_captures == max_captures:
                max_capture_moves.append(move)

        if max_captures > 1:
            return max_capture_moves

        moves.sort(key=lambda move: get_num_captures(board, move, piece),
                   reverse=True)

        return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        max_captures = max(num_captures)
        if max_captures > 0:
            for idx in range(len(num_captures)):
                if num_captures[idx] == max_captures:
                    return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer7(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        # if 6 < move_number < 14:
        #     with open('b_5.json', 'r') as f:
        #         data = json.load(f)
        #     if self.board in data:
        #         alpha = data[self.board][0]
        #         alpha_move = data[self.board][1]
        #         if alpha > 0:
        #             return alpha_move

        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        max_captures = max(num_captures)
        if max_captures > 0:
            for idx in range(len(num_captures)):
                if num_captures[idx] == max_captures:
                    return moves[idx]

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        max_score = -100
        depth = 4
        if move_number > 15:
            depth = 6
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        # if max_score == alpha:
        #     return alpha_move

        return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)

        moves.sort(key=lambda move: get_num_captures(board, move, piece),
                   reverse=True)

        return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        max_captures = max(num_captures)
        if max_captures > 0:
            for idx in range(len(num_captures)):
                if num_captures[idx] == max_captures:
                    return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer8(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        # if 6 < move_number < 14:
        #     with open('b_5.json', 'r') as f:
        #         data = json.load(f)
        #     if self.board in data:
        #         alpha = data[self.board][0]
        #         alpha_move = data[self.board][1]
        #         if alpha > 0:
        #             return alpha_move

        moves = self.get_moves_to_check(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        max_score = -100
        depth = 4
        if move_number > 15:
            depth = 6
        if move_number > 17:
            depth = 9
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        # if max_score == alpha:
        #     return alpha_move

        return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)

        new_moves = []
        for move in moves:
            if get_num_captures(board, move, piece) > 0:
                new_moves.append(move)

        if new_moves:
            new_moves.sort(key=lambda move: get_num_captures(board, move, piece),
                           reverse=True)

            return new_moves
        else:
            moves.sort(key=lambda x: get_opponent_liberties_at_location(
                board, x, piece))

            return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        if num_captures:
            max_captures = max(num_captures)
            if max_captures > 0:
                for idx in range(len(num_captures)):
                    if num_captures[idx] == max_captures:
                        return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer9(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        if 6 < move_number < 14:
            with open('b_6.json', 'r') as f:
                data = json.load(f)
            if self.board in data:
                alpha = data[self.board][0]
                alpha_move = data[self.board][1]
                if alpha > 0:
                    return alpha_move

        moves = self.get_moves_to_check(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        max_score = -100
        depth = 4
        if move_number > 13:
            depth = 6
        if move_number > 18:
            depth = 7
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        return self.choose_preferred_move(self.best_moves)

    def choose_preferred_move(self, moves):
        board = self.board
        piece = self.piece
        if 12 in moves:
            return 12

        for move in [6, 8]:
            if move in moves:
                return move

        if board[6] == piece and board[8] == piece and 2 in moves:
            return 2

        for move in [6, 16]:
            if move in moves:
                return move

        if board[6] == piece and board[16] == piece and 10 in moves:
            return 10

        for move in [8, 18]:
            if move in moves:
                return move

        if board[8] == piece and board[18] == piece and 14 in moves:
            return 14

        for move in [16, 18]:
            if move in moves:
                return move

        if board[16] == piece and board[18] == piece and 22 in moves:
            return 22

        ordered_moves = [12, 6, 8, 2, 16, 10, 18, 22, 14]

        for move in ordered_moves:
            if move in moves:
                return move

        return moves[0]

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)

        new_moves = []
        for move in moves:
            if get_num_captures(board, move, piece) > 0:
                new_moves.append(move)

        if new_moves:
            new_moves.sort(key=lambda move: get_num_captures(board, move, piece),
                           reverse=True)

            return new_moves
        else:
            moves.sort(key=lambda x: get_opponent_liberties_at_location(
                board, x, piece))

            return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        if num_captures:
            max_captures = max(num_captures)
            if max_captures > 1:
                for idx in range(len(num_captures)):
                    if num_captures[idx] == max_captures:
                        return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer10(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        moves = self.get_moves_to_check(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        max_score = -100
        depth = 4
        if move_number > 13:
            depth = 6
        if move_number > 18:
            depth = 7
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        return self.choose_preferred_move(self.best_moves)

    def choose_preferred_move(self, moves):
        board = self.board
        piece = self.piece
        if 12 in moves:
            return 12

        for move in [6, 8]:
            if move in moves:
                return move

        if board[6] == piece and board[8] == piece and 2 in moves:
            return 2

        for move in [6, 16]:
            if move in moves:
                return move

        if board[6] == piece and board[16] == piece and 10 in moves:
            return 10

        for move in [8, 18]:
            if move in moves:
                return move

        if board[8] == piece and board[18] == piece and 14 in moves:
            return 14

        for move in [16, 18]:
            if move in moves:
                return move

        if board[16] == piece and board[18] == piece and 22 in moves:
            return 22

        ordered_moves = [12, 6, 8, 2, 16, 10, 18, 22, 14]

        for move in ordered_moves:
            if move in moves:
                return move

        return moves[0]

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)

        new_moves = []
        for move in moves:
            if get_num_captures(board, move, piece) > 0:
                new_moves.append(move)

        if new_moves:
            new_moves.sort(key=lambda move: get_num_captures(board, move, piece),
                           reverse=True)

            return new_moves
        else:
            moves.sort(key=lambda x: get_opponent_liberties_at_location(
                board, x, piece))

            return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        if num_captures:
            max_captures = max(num_captures)
            if max_captures > 1:
                for idx in range(len(num_captures)):
                    if num_captures[idx] == max_captures:
                        return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                if self.data and new_board in self.data:
                    value, _ = self.data[new_board]
                else:
                    value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class BlackPlayer11(Player):
    def select_move(self, move_number):
        t = time.time()

        if move_number == 0:
            return 12

        if (self.old_board == self.board and
                score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        moves = self.get_moves_to_check(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        boards = [play_move(self.board, self.old_board, move, self.piece) for
                  move in moves]
        max_score = -100
        depth = 4
        if move_number > 13:
            depth = 6
        if move_number > 18:
            depth = 7
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number + 1, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        return self.choose_preferred_move(self.best_moves)

    def choose_preferred_move(self, moves):
        board = self.board
        piece = self.piece
        if 12 in moves:
            return 12

        for move in [6, 8]:
            if move in moves:
                return move

        if board[6] == piece and board[8] == piece and 2 in moves:
            return 2

        for move in [6, 16]:
            if move in moves:
                return move

        if board[6] == piece and board[16] == piece and 10 in moves:
            return 10

        for move in [8, 18]:
            if move in moves:
                return move

        if board[8] == piece and board[18] == piece and 14 in moves:
            return 14

        for move in [16, 18]:
            if move in moves:
                return move

        if board[16] == piece and board[18] == piece and 22 in moves:
            return 22

        ordered_moves = [12, 6, 8, 2, 16, 10, 18, 22, 14]

        for move in ordered_moves:
            if move in moves:
                return move

        return moves[0]

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)

        new_moves = []
        for move in moves:
            if get_num_captures(board, move, piece) > 0:
                new_moves.append(move)

        if new_moves:
            new_moves.sort(key=lambda move: get_num_captures(board, move, piece),
                           reverse=True)

            return new_moves
        else:
            moves.sort(key=lambda x: get_opponent_liberties_at_location(
                board, x, piece))

            return moves

    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, self.piece)

        num_captures = [get_num_captures(self.board, move, self.piece)
                        for move in moves]

        if num_captures:
            max_captures = max(num_captures)
            if max_captures > 1:
                for idx in range(len(num_captures)):
                    if num_captures[idx] == max_captures:
                        return moves[idx]

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                if self.bdata and new_board in self.bdata:
                    value, _ = self.bdata[new_board]
                else:
                    value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            return -1

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                if new_board in self.wdata:
                    value, _ = self.wdata[new_board]
                else:
                    value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value


class WhitePlayer(Player):
    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value

    def capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        for move in moves:
            if get_num_captures(self.board, move, self.piece) > 0:
                return move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        if (move_number > 2 and self.old_board == self.board
                and score_diff(self.board, self.piece) > 0):
            return -1

        if move_number < 6:
            return self.capture_or_central_move()
        else:
            t = time.time()

            moves = self.get_moves_to_check(self.board, self.old_board,
                                            self.piece)

            if not moves:
                return -1

            depth = 3
            if move_number > 15:
                depth = 5

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in moves]
            max_score = -100
            self.best_moves = []
            for idx, board in enumerate(boards):
                score = self.minimizer(board, self.board, depth, move_number, t,
                                       -100, 100)
                if score > max_score:
                    max_score = score
                    self.best_moves = []
                if score == max_score:
                    self.best_moves.append(moves[idx])

            return random.choice(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves


class WhitePlayer2(Player):
    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, other_piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value

    def capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        for move in moves:
            if get_num_captures(self.board, move, self.piece) > 0:
                return move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        t = time.time()
        if (move_number > 2 and self.old_board == self.board
                and score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        if move_number < 6:
            return self.capture_or_central_move()
        else:
            if move_number < 14:
                with open('w_5.json', 'r') as f:
                    data = json.load(f)
                if self.board in data:
                    alpha = data[self.board][0]
                    alpha_move = data[self.board][1]

            moves = self.get_moves_to_check(self.board, self.old_board,
                                            self.piece)

            if not moves:
                return -1

            depth = 3
            if move_number > 15:
                depth = 5

            boards = [play_move(self.board, self.old_board, move, self.piece) for
                      move in
                      moves]
            max_score = -100
            self.best_moves = []
            for idx, board in enumerate(boards):
                score = self.minimizer(board, self.board, depth, move_number, t,
                                       alpha, 100)
                if score > max_score:
                    max_score = score
                    self.best_moves = []
                if score == max_score:
                    self.best_moves.append(moves[idx])

            if max_score == alpha:
                return alpha_move

            return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves


class WhitePlayer3(Player):
    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value

    def capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        for move in moves:
            if get_num_captures(self.board, move, self.piece) > 0:
                return move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        t = time.time()
        if (move_number > 2 and self.old_board == self.board
                and score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        if move_number < 6:
            return self.capture_or_central_move()
        else:
            # if move_number < 14:
            #     with open('w_5.json', 'r') as f:
            #         data = json.load(f)
            #     if self.board in data:
            #         alpha = data[self.board][0]
            #         alpha_move = data[self.board][1]

            moves = self.get_moves_to_check(self.board, self.old_board,
                                            self.piece)

            if not moves:
                return -1

            depth = 4
            if move_number > 15:
                depth = 5

            boards = [play_move(self.board, self.old_board, move, self.piece)
                      for move in moves]
            max_score = -100
            self.best_moves = []
            for idx, board in enumerate(boards):
                score = self.minimizer(board, self.board, depth, move_number, t,
                                       alpha, 100)
                if score > max_score:
                    max_score = score
                    self.best_moves = []
                if score == max_score:
                    self.best_moves.append(moves[idx])

            # if max_score == alpha:
            #     return alpha_move

            return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves


class WhitePlayer4(Player):
    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                if new_board in self.wdata:
                    value, _ = self.wdata[new_board]
                else:
                    value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value

    def capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        for move in moves:
            if get_num_captures(self.board, move, self.piece) > 0:
                return move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        t = time.time()
        if (move_number > 2 and self.old_board == self.board
                and score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        if move_number < 6:
            return self.capture_or_central_move()
        if move_number < 14:
            with open('w_5.json', 'r') as f:
                data = json.load(f)
            if self.board in data:
                alpha = data[self.board][0]
                alpha_move = data[self.board][1]
                if alpha > 0:
                    return alpha_move

        moves = self.get_moves_to_check(self.board, self.old_board,
                                        self.piece)

        if not moves:
            return -1

        depth = 4
        if move_number > 15:
            depth = 6

        boards = [play_move(self.board, self.old_board, move, self.piece)
                  for move in moves]
        max_score = -100
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        if max_score == alpha:
            return alpha_move

        return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves


class WhitePlayer5(Player):
    def maximizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        moves = self.get_moves_to_check(board, old_board, self.piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, self.piece)

        if not moves:
            return -100

        if (-1 in moves and old_board == board and move_number > 2 and
                score_diff(board, self.piece) > 0):
            return 100

        boards = [play_move(board, old_board, move, self.piece) for move in
                  moves]

        max_value = -100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                if new_board in self.wdata:
                    value, _ = self.wdata[new_board]
                else:
                    value = self.minimizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                    break

        return max_value

    def minimizer(self, board, old_board, depth, move_number, start_time,
                  alpha, beta):
        other_piece = get_other_piece(self.piece)
        moves = self.get_moves_to_check(board, old_board, other_piece)

        if not moves:
            moves = get_valid_moves_with_pass(board, old_board, other_piece)

        if not moves:
            return 100

        boards = [play_move(board, old_board, move, other_piece) for move in
                  moves]

        min_value = 100

        if depth == 0 or time.time() - start_time > 8.7 or move_number >= 23:
            return score_diff(board, self.piece)
        else:
            for new_board in boards:
                value = self.maximizer(new_board, board, depth - 1,
                                       move_number + 1, start_time,
                                       alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break

        return min_value

    def capture_or_central_move(self):
        moves = get_valid_moves(self.board, self.old_board, self.piece)

        if not moves:
            return -1

        for move in moves:
            if get_num_captures(self.board, move, self.piece) > 0:
                return move

        points = [array_index_to_point(move) for move in moves]
        x_diffs = [(pt[0] - 2) ** 2 for pt in points]
        y_diffs = [(pt[1] - 2) ** 2 for pt in points]
        diffs = [math.sqrt(x_diff + y_diff)
                 for x_diff, y_diff in zip(x_diffs, y_diffs)]

        return moves[numpy.argmin(diffs)]

    def select_move(self, move_number):
        t = time.time()
        if (move_number > 2 and self.old_board == self.board
                and score_diff(self.board, self.piece) > 0):
            return -1

        alpha = -100
        if move_number < 6:
            return self.capture_or_central_move()

        moves = self.get_moves_to_check(self.board, self.old_board,
                                        self.piece)

        if not moves:
            return -1

        depth = 4
        if move_number > 15:
            depth = 6

        boards = [play_move(self.board, self.old_board, move, self.piece)
                  for move in moves]
        max_score = -100
        self.best_moves = []
        for idx, board in enumerate(boards):
            score = self.minimizer(board, self.board, depth, move_number, t,
                                   alpha, 100)
            if score > max_score:
                max_score = score
                alpha = max_score
                self.best_moves = []
            if score == max_score:
                self.best_moves.append(moves[idx])

        return self.choose_preferred_move(self.best_moves)

    def get_moves_to_check(self, board, old_board, piece):
        moves = get_valid_moves(board, old_board, piece)
        moves.append(-1)
        return moves


if __name__ == '__main__':
    player, last_board, current_board = read_board_from_file()

    move_ctr_set = False
    if last_board == '0' * 25:
        move_ctr_set = True
        if current_board == last_board:
            move_ctr = 0
        else:
            move_ctr = 1

    if not move_ctr_set:
        move_ctr = read_move_ctr_from_file()

    with open('w_6.json', 'r') as f:
        wdata = json.load(f)

    with open('b_6.json', 'r') as f:
        bdata = json.load(f)

    if player.strip() == '1':
        Agent = BlackPlayer10
    else:
        Agent = WhitePlayer4

    agent = Agent(player, last_board, current_board, bdata, wdata)
    play = agent.play_move(move_ctr, True)

    move_ctr += 1

    write_move_to_file(play)
    write_move_ctr_to_file(move_ctr)
