import sys
from argparse import ArgumentParser, BooleanOptionalAction
from model import Game
from model import Pos_camera
from pyqtgraph.Qt import QtGui

# define arguments
parser = ArgumentParser()
parser.add_argument("-m", "--mapping",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Starts the mapping of the board")

parser.add_argument("-s", "--start",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Chess game starts")

parser.add_argument("-d", "--debugthiswill",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Debug it plz")

parser.add_argument("-c", "--camera",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Positionnement camera")

parser.add_argument("-i", "--image",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Working image test")


args = vars(parser.parse_args())

if __name__ == "__main__":
  # calibration mapping
  if args['mapping']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.mapping()

  # start a game
  if args['start']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.start()
    sys.exit(app.exec_())

  # Debug Test 
  if args['debugthiswill']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.debugthiswill()
    sys.exit(app.exec_())

  # Positionnement de la caméra
  if args['camera']:
    app = QtGui.QApplication(sys.argv)
    Camera_pos = Pos_camera()
    #Camera_pos.Pos_camera()
    sys.exit(app.exec_())

  # Test d'image sur laquelle la détection est possible 
  if args['image']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.Debug_using_image()
    #sys.exit(app.exec_())