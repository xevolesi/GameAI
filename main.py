from framework.Coach import Coach
from tic_tac_toe.TicTacToeGame import TicTacToeGame
from tic_tac_toe.tensorflow.NNet import NNetWrapper as nn
from framework.utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 20,
    'arenaCompare': 50,
    'cpuct': 1,

    'checkpoint': './temp/tensorflow_ttt/',
    'load_model': False,
    'load_folder_file': ('./temp/tensorflow_ttt/', 'best.pth.tar'),
})

if __name__ == "__main__":
    g = TicTacToeGame(3)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()
