from my_player3 import *
from argparse import ArgumentParser
import json


def maximizer(piece, board, old_board, depth, move_number, alpha, beta):
    if old_board == board and score_diff(board, piece) > 0:
        return -1

    p_moves = get_valid_moves(board, old_board, piece)

    moves = []
    for move in p_moves:
        surr = get_surroundings(move)
        if any(board[sur] == piece for sur in surr):
            moves.append(move)

    if not moves:
        moves = p_moves

    if board != old_board:
        moves.append(-1)

    if not moves:
        return -100

    if (-1 in moves and old_board == board and move_number > 2 and
            score_diff(board, piece) > 0):
        return 100

    boards = [play_move(board, old_board, move, piece) for move in moves]

    max_value = -100

    if depth == 0 or move_number >= 23:
        # import ipdb; ipdb.set_trace()
        return score_diff(board, piece)
    else:
        for new_board in boards:
            value = minimizer(piece, new_board, board, depth - 1,
                              move_number + 1,
                              alpha, beta)
            max_value = max(max_value, value)
            alpha = max(alpha, value)

            if beta <= alpha:
                break

    return max_value


def minimizer(piece, board, old_board, depth, move_number,
              alpha, beta):
    other_piece = get_other_piece(piece)
    moves = get_valid_moves(board, old_board, other_piece)

    if not moves:
        return 100

    boards = [play_move(board, old_board, move, other_piece) for move in moves]

    min_value = 100

    if depth == 0 or move_number >= 23:
        return score_diff(board, piece)
    else:
        for new_board in boards:
            value = maximizer(piece, new_board, board, depth - 1,
                              move_number + 1,
                              alpha, beta)
            min_value = min(min_value, value)
            beta = min(beta, value)

            if beta <= alpha:
                break

    return min_value


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--depth', type=int)
    args = ap.parse_args()
    files = ['boards10.txt', 'boards11.txt', 'boards12.txt', 'boards13.txt']
    exts = ['b10', 'b11', 'w10', 'w11']
    move_number = 0
    depth = args.depth

    res = {}

    # for ext, filename in zip(exts, files):
    #     with open(filename, 'r') as f:
    #         boards = f.readlines()
    #     boards = [board.strip() for board in boards]
    #     boards = [board for board in boards if board]
    #
    # # Depth 7: 450
    # # Depth 6: 115
    # # Depth 5: 35
    # # Depth 4: 10
    #
    #     for i, board in enumerate(boards):
    #         board =
    # board.strip()
    #         if not board:
    #             continue
    #         t = time.time()
    #         p_moves = get_valid_moves(board, piece)
    #         bb = [play_move(board, move, piece) for move in p_moves]
    #         scores = [minimizer(board, board, depth, move_number, -100, 100)
    #                   for board in bb]
    #         idx = numpy.argmin(scores)
    #         m = p_moves[idx]
    #         res[board] = m
    #         print('Board {}/{}: Depth: {}, Time: {:.3f}'.format(i,
    #                                                             len(
    #                                                                 boards)-1, depth,
    #                                                          time.time() - t))
    #
    #     with open('score_{}_{}.json'.format(ext, depth), 'w') as f:
    #         json.dump(res, f)
    for ext, filename in zip(exts, files):
        with open(filename, 'r') as f:
            boards = f.readlines()
        boards = [board.strip() for board in boards]
        boards = [board for board in boards if board]
        res = {}

        if ext.startswith('b'):
            piece = P1
        else:
            piece = P2

    # Depth 7: 450
    # Depth 6: 115
    # Depth 5: 35
    # Depth 4: 10

        for i, board in enumerate(boards):
            board = board.strip()
            if not board:
                continue
            t = time.time()
            p_moves = get_valid_moves(board, board, piece)
            bb = [play_move(board, board, move, piece) for move in p_moves]
            scores = [minimizer(piece, board, board, depth, move_number, -100,
                                100)
                      for board in bb]
            idx = numpy.argmax(scores)
            m = p_moves[idx]
            res[board] = (scores[idx], m)
            print('Board {}/{}: Depth: {}, Time: {:.3f}'.format(i,
                                                                len(boards)-1,
                                                                depth,
                                                             time.time() - t))

        if res:
            with open('score_{}_{}.json'.format(ext, depth), 'a') as f:
                json.dump(res, f)
            res = {}
            print('Wrote to file score_{}_{}.json'.format(ext, depth))
