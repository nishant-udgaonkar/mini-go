from my_player3 import *
import time
import itertools


def play_game(p1, p2, verbose=False):
    moves = 0
    board = '0' * 25
    ended = False

    while moves < 24:
        t = time.time()
        try:
            m1 = p1.play_move(moves, verbose)
        except IndexError:
            ended = True
            res = (0, 1)
            break
        board = play_move(board, m1, p1.piece)
        t2 = time.time()
        if verbose:
            print('Time: {}'.format(t2 - t))
        p2.set_next_board(board)
        if verbose:
            display_board(board)
        moves += 1

        t = time.time()
        try:
            m2 = p2.play_move(moves, verbose)
        except IndexError:
            ended = True
            res = (1, 0)
            break
        board = play_move(board, m2, p2.piece)
        t2 = time.time()
        if verbose:
            print('Time: {}'.format(t2 - t))
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


def tournament(rounds, agents, verbose=False):
    matchups = list(itertools.permutations(players, 2))
    pop_idx = -1
    for i in range(len(matchups)):
        if matchups[i][0] == matchups[i][1]:
            pop_idx = i
            break
    matchups.pop(pop_idx)
    results = {}
    for ag in agents:
        results[ag] = {}
        for a in agents:
            if a != ag:
                results[ag][a] = {'W': 0, 'B': 0}

    b = '0' * 25

    for i in range(rounds):
        print('Round {}'.format(i))
        for idx, ps in enumerate(matchups):
            tt = time.time()
            p1, p2 = ps
            res = play_game(p1(P1, b, b), p2(P2, b, b), verbose=verbose)
            if res[0] > res[1]:
                results[p1][p2]['B'] += 1
            else:
                results[p2][p1]['W'] += 1

            print('{}/{} - {} {} - {} {} - {:.4f}'.format(
                idx, len(matchups), p1.__name__, res[0], res[1], p2.__name__,
                time.time() - tt))

    return results


if __name__ == '__main__':

    game = '0' * 25

    all_players = [
        RandomPlayer,
        MinLibertyPlayer,
        MaxLibertyPlayer,
        CentralPlayer,
        CaptureOrCentralPlayer,
        BiggestCaptureOrCentralPlayer,
        MinimaxPlayer,
        CaptureOrCentralThenMinimaxPlayer,
        AlphaBetaPlayer,
        CaptureOrCentralThenAlphaBetaPlayer,
        CentralThenAlphaBetaPlayer,
        ManualPlayer
    ]

    players = [Idea2Player, RandomPlayer]
    ties = 3

    r = tournament(ties, players, False)

    for k, v in r.items():
        out_d = dict()
        wins = 0
        for k2, v2 in v.items():
            out_d[k2.__name__] = v2
            for v3 in v2.values():
                wins += v3

        print(k.__name__, '({}/{})'.format(wins, ties * 2 * (len(players) - 1)))
        for x, y in out_d.items():
            print("      {} {}".format(x, y))
