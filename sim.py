from my_player3 import *
import time
import itertools


boards_to_write0 = set()
boards_to_write1 = set()
boards_to_write2 = set()
boards_to_write3 = set()

def play_game(p1, p2, verbose=False):
    moves = 0
    board = '0' * 25
    board2 = '0' * 25
    ended = False

    pass_ctr = 0

    while moves < 24:
        # if 5 < moves < 10:
        #     boards_to_write0.add(board)
        # if 9 < moves < 15:
        #     boards_to_write1.add(board)
        t = time.time()
        try:
            m1 = p1.play_move(moves, verbose)
        except IndexError:
            ended = True
            res = (0, 1)
            break
        if m1 == -1:
            pass_ctr += 1
        else:
            pass_ctr = 0
        if pass_ctr == 2:
            return score(board)

        board, board2 = play_move(board, board2, m1, p1.piece), board
        t2 = time.time()
        if t2 - t > 9.5:
            print(moves, t2-t)
        if verbose:
            print('Time: {:.2f}'.format(t2 - t))
        p2.set_next_board(board)
        if verbose:
            display_board(board)
        moves += 1

        # if 6 < moves < 10:
        #     boards_to_write2.add(board)
        # if 9 < moves < 14:
        #     boards_to_write3.add(board)
        t = time.time()
        try:
            m2 = p2.play_move(moves, verbose)
        except IndexError:
            ended = True
            res = (1, 0)
            break
        if m2 == -1:
            pass_ctr += 1
        else:
            pass_ctr = 0
        if pass_ctr == 2:
            return score(board)
        board, board2 = play_move(board, board2, m2, p2.piece), board
        t2 = time.time()
        if t2 - t > 9.5:
            print(moves, t2-t)
        if verbose:
            print('Time: {:.2f}'.format(t2 - t))
        p1.set_next_board(board)
        if verbose:
            display_board(board)
        moves += 1

    if not ended:
        res = score(board)

    if verbose:
        print('{} {} - {} {}'.format(p1.__class__.__name__, res[0], res[1],
                                     p2.__class__.__name__))

    return res


if __name__ == '__main__':

    game = '0' * 25

    out_ctr = 1
    in_ctr = 100

    black_total = 0
    white_total = 0

    verbose = True

    for x in range(out_ctr):
        p1 = BlackPlayer10
        p2 = RandomPlayer

        black = 0
        white = 0

        # play_game(p1(P1, game, game), p2(P2, game, game), verbose=True)
        # input()
        for i in range(in_ctr):
            t1 = time.time()
            res = play_game(p1(P1, game, game), p2(P2, game, game),
                            verbose=verbose)
            if res[0] > res[1]:
                winner = 'Black'
                black += 1
            else:
                winner = 'White'
                white += 1
                input()
            print('Game {}: Winner {}, Time {:.2f}'.format((in_ctr*x) + i,
                                                           winner,
                                                        time.time() - t1))

        black_total += black
        white_total += white
        print('Black ({}): {}, White ({}): {}, ({}:{})'.format(p1.__name__,
                                                              black,
                                                      p2.__name__, white,
                                                               black_total,
                                                               white_total))

    # with open('boards10.txt', 'w') as f:
    #     for board in boards_to_write0:
    #         f.write(board+'\n')
    #
    # with open('boards11.txt', 'w') as f:
    #     for board in boards_to_write1:
    #         f.write(board+'\n')

    # with open('boards12.txt', 'w') as f:
    #     for board in boards_to_write2:
    #         f.write(board+'\n')
    #
    # with open('boards13.txt', 'w') as f:
    #     for board in boards_to_write3:
    #         f.write(board+'\n')
