import json
from my_player3 import (get_similar_boards, get_valid_moves, play_move,
                        score_diff)
import time
from collections import deque

score_tracker1 = {}
score_tracker2 = {}
score_tracker3 = {}
score_tracker4 = {}
score_tracker5 = {}


def do_q_learning():
    t = time.time()
    b = '0' * 25
    p = 1

    alpha1 = 0.5
    beta1 = 0.5

    alpha2 = 0.7
    beta2 = 0.3

    alpha3 = 0.9
    beta3 = 0.1

    alpha4 = 0.3
    beta4 = 0.7

    alpha5 = 0.1
    beta5 = 0.9

    boards = deque([(b, 0)])

    while boards:
        board, move_ctr = boards.popleft()

        p = (p + 1) % 2
        pp = str(p + 1)
        opp = str(((p + 1) % 2) + 1)

        if move_ctr == 5:
            if '1' + board in score_tracker1:
                continue
            if '2' + board in score_tracker1:
                continue
            if score_diff(board, '1') > 0:
                score_tracker1['1' + board] = 100
                score_tracker2['1' + board] = 100
                score_tracker3['1' + board] = 100
                score_tracker4['1' + board] = 100
                score_tracker5['1' + board] = 100

                score_tracker1['2' + board] = -100
                score_tracker2['2' + board] = -100
                score_tracker3['2' + board] = -100
                score_tracker4['2' + board] = -100
                score_tracker5['2' + board] = -100
            else:
                score_tracker1['1' + board] = -100
                score_tracker2['1' + board] = -100
                score_tracker3['1' + board] = -100
                score_tracker4['1' + board] = -100
                score_tracker5['1' + board] = -100

                score_tracker1['2' + board] = 100
                score_tracker2['2' + board] = 100
                score_tracker3['2' + board] = 100
                score_tracker4['2' + board] = 100
                score_tracker5['2' + board] = 100

            continue

        moves = get_valid_moves(board, piece=pp)
        bs = []
        for move in moves:
            b = play_move(board, move, pp)
            b_in_scores = False
            sb = get_similar_boards(b, opp)
            for sb_ in sb:
                if sb_ in score_tracker1:
                    b2 = sb_[1] + sb_[0]
                    b_in_scores = True
                    break
            if not b_in_scores:
                b2 = opp + b

            board_in_scores = False
            sb = get_similar_boards(board, pp)
            for sb_ in sb:
                if sb_ in score_tracker1:
                    board2 = sb_[1] + sb_[0]
                    board_in_scores = True
                    break
            if not board_in_scores:
                board2 = pp + board

            if b2 in score_tracker1:
                if board2 not in score_tracker1:
                    score_tracker1[board2] = 0
                    score_tracker2[board2] = 0
                    score_tracker3[board2] = 0
                    score_tracker4[board2] = 0
                    score_tracker5[board2] = 0

                score_tracker1[board2] = (alpha1 * score_tracker1[board2] +
                                          beta1 * score_tracker1[b2])
                score_tracker2[board2] = (alpha2 * score_tracker2[board2] +
                                          beta2 * score_tracker2[b2])
                score_tracker3[board2] = (alpha3 * score_tracker3[board2] +
                                          beta3 * score_tracker3[b2])
                score_tracker4[board2] = (alpha4 * score_tracker4[board2] +
                                          beta4 * score_tracker4[b2])
                score_tracker5[board2] = (alpha5 * score_tracker5[board2] +
                                          beta5 * score_tracker5[b2])

            bs.append((b2[1:], move_ctr + 1))
        boards.extend(bs)

    print(time.time() - t)


if __name__ == '__main__':
    for _ in range(6):
        do_q_learning()

    with open('score_tracker1.json', 'w') as f:
        json.dump(score_tracker1, f)
    with open('score_tracker2.json', 'w') as f:
        json.dump(score_tracker2, f)
    with open('score_tracker3.json', 'w') as f:
        json.dump(score_tracker3, f)
    with open('score_tracker4.json', 'w') as f:
        json.dump(score_tracker4, f)
    with open('score_tracker5.json', 'w') as f:
        json.dump(score_tracker5, f)
