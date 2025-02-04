from typing import Dict, Tuple
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
from model.agent import Agent
from model.camera import Camera
from model.gui import GUI
from dotenv import dotenv_values
from pyqtgraph.Qt import QtCore, QtGui
from utils import draw_bounding_box_on_image

import cv2
import numpy as np
import time
import imutils
import PIL.Image as Image

STANDARD_COLORS = [
  'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
  'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
  'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
  'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
  'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
  'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
  'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
  'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
  'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
  'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
  'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
  'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
  'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
  'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
  'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
  'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
  'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
  'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
  'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
  'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
  'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
  'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
  'WhiteSmoke', 'Yellow', 'YellowGreen'
]

# sorteio de cores para não pegar sempre as mesmas
COLORS_INDEX = np.random.randint(0, len(STANDARD_COLORS), 12)
COLORS = [STANDARD_COLORS[i] for i in COLORS_INDEX]

class Game(GUI):
  __cam_address: str
  __running_calibration: ChessboardCalibration
  __board: Board
  __config: Dict
  __agent: Agent
  __camera: Camera = None
  __fps: float
  __lastupdate: float
  __detections: list = None
  __scan_timer: QtCore.QTimer = None

  def __init__(self, **kwargs):
    super(Game, self).__init__(**kwargs)
    self.__config = dotenv_values()
    #self.__cam_address = self.__config.get('CAM_ADDRESS')
    self.__cam_address = 2
    self.__agent = Agent()
    self.__debug = bool(int(self.__config.get('DEBUG')))

    # frame rate metrics
    self.__fps = 0.
    self.__lastupdate = time.time()

  def mapping(self):
    """
    Start mapping chess board
    """
    #camera = Camera(self.__cam_address)
    #frame = camera.capture()
    frame = cv2.imread('New_chessboard/1_raw.jpg')

    # do calibration mapping
    chessboard_calibration = ChessboardCalibration(debug=self.__debug)
    chessboard_calibration.mapping(
      chessboard_img=frame,
      fix_rotate=False,
      rotate_val=0,
      add_padding=True
    )
    chessboard_calibration.saveMapping()
    
    # release camera
    #camera.destroy()
    print('Done!')

  def start(self):
    """
    Start game
    """
    self.__camera = Camera(self.__cam_address)

    self.__running_calibration = ChessboardCalibration()
    found, self.__board = self.__running_calibration.loadMapping()
    if not found:
      raise Exception('No mapping found. Run calibration mapping')

    self.__captureFrame()
    self.__runScan(only_prediction=True)
    self.show()

  def __captureFrame(self):
    frame = self.__camera.capture()
    self.__processed_image = self.__running_calibration.applyMapping(frame)

    result, hand_is_detected = self.__addHandBoundingBoxes(self.__processed_image)
    if hand_is_detected:
      self.__scheduleScan()
    else:
      result = self.__addPiecesBoundingBoxes(self.__processed_image)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    self.setImage(result)
    self.__updateFrameRate()

    QtCore.QTimer.singleShot(1, self.__captureFrame)

  def __scheduleScan(self):
    self.__detections = None
    if self.__scan_timer is not None:
      self.__scan_timer.stop()

    self.__scan_timer = QtCore.QTimer(self)
    self.__scan_timer.timeout.connect(self.__runScan)
    self.__scan_timer.setSingleShot(True)
    self.__scan_timer.start(800)

  def __addPiecesBoundingBoxes(self, image):
    if self.__detections is None:
      return image

    image_pil = Image.fromarray(np.uint8(image.copy())).convert('RGB')
    height, width = image.shape[:2]

    for (name, bbox, acc, cls_id) in self.__detections:

      # These lines containing np.random cause the bounding boxes and the precision
      # to fluctuate as if they were being detected in real time.
      #
      # The illusion is part of the show :)
      #
      x, y, w, h = bbox - np.random.randint(-1, 1, len(bbox))
      acc = acc - np.random.uniform(0.01, 0)

      xmin = x / width
      ymin = y / height
      xmax = w / width
      ymax = h / height

      draw_bounding_box_on_image(
        image=image_pil,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        color=COLORS[cls_id],
        display_str_list=['{}: {:.1f}%'.format(name, acc * 100)]
      )

    return np.array(image_pil)

  def __addHandBoundingBoxes(self, image):
    inverted = cv2.bitwise_not(image.copy())
    cv2.imwrite('Bounding_boxes/1_2_Inverted.jpg', inverted)
    #hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    first_step = hsv.copy()
    cv2.imwrite('Bounding_boxes/1_first_step_boxes.jpg', first_step)

    # ========
    # Eliminate the squares
    # ========
    '''
    # ---- red
    lower = np.array([135, 6, 91])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    mask = 255-mask
    green_squares_mask = mask.copy()
    cv2.imwrite('Bounding_boxes/2_green_squares_mask.jpg', green_squares_mask)
    '''
        # ---- green
    #lower = np.array([0, 60, 20])
    #upper = np.array([30, 255, 255])
    #lower = np.array([0, 40, 200])
    #upper = np.array([30, 255, 255])
    lower = np.array([50, 33, 0])
    upper = np.array([92, 255, 90])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    cv2.imwrite('Bounding_boxes/2_1_green_squares_mask.jpg', mask)
    mask = 255-mask
    green_squares_mask = mask.copy()
    cv2.imwrite('Bounding_boxes/2_2_green_squares_mask.jpg', green_squares_mask)

    # ---- white
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    cv2.imwrite('Bounding_boxes/3_0_white_squares_mask.jpg', hsv)
    lower = np.array([0, 0, 81])
    upper = np.array([255, 35, 255])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    mask = 255-mask
    white_squares_mask = mask.copy()
    cv2.imwrite('Bounding_boxes/3_white_squares_mask.jpg', white_squares_mask)

    # ---- Final result without the squares
    image_final = image.copy()
    image_final = cv2.bitwise_and(image_final, image_final, mask=green_squares_mask)
    image_final = cv2.bitwise_and(image_final, image_final, mask=white_squares_mask)
    cv2.imwrite('Bounding_boxes/4_image_final.jpg', image_final)

    # ========
    # Select the pieces
    # ========

    #hsv = cv2.imread('New_chessboard/a.png')


    inverted = cv2.bitwise_not(image_final.copy())
    hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)
    cv2.imwrite('Bounding_boxes/5_first_step_pieces.jpg', hsv)

    # ---- White pieces
    target = image.copy()
    lower = np.array([76, 87, 50])
    upper = np.array([255, 255, 255])
    white_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)
    cv2.imwrite('Bounding_boxes/6_white_pieces_mask.jpg', white_pieces_mask)

    # ---- Black pieces
    lower = np.array([0, 0, 159])
    upper = np.array([55, 255, 255])
    black_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)
    cv2.imwrite('Bounding_boxes/7_black_pieces_mask.jpg', black_pieces_mask)

    hand_is_detected, hand_contours = self.__hand_detected(image_final, white_pieces_mask, black_pieces_mask)
    if hand_is_detected:
      self.__drawHand(target, hand_contours)
      cv2.imwrite('Bounding_boxes/8_Hand_is_detected.jpg', target)

    return (target, hand_is_detected)

  def __drawHand(self, target, hand_contours):
    peri = cv2.arcLength(hand_contours, True)
    biggest_cnt = cv2.approxPolyDP(hand_contours, 0.015 * peri, True)
    x, y, w, h = cv2.boundingRect(biggest_cnt)
    cv2.rectangle(target, (x,y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(target, 'HUMAN HAND', (x, y - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

  def __hand_detected(self, no_houses_frame, white_pieces_mask, black_pieces_mask) -> Tuple[bool, list]:
    """
    return `True` or `False` if hand is detected
    """
    white_pieces_mask = 255-white_pieces_mask
    black_pieces_mask = 255-black_pieces_mask

    no_houses_frame = cv2.bitwise_and(no_houses_frame, no_houses_frame, mask=white_pieces_mask)
    no_houses_frame = cv2.bitwise_and(no_houses_frame, no_houses_frame, mask=black_pieces_mask)

    # convert image to gray scale
    gray = cv2.cvtColor(no_houses_frame, cv2.COLOR_BGR2GRAY)

    # This is the threshold level for every pixel.
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=8)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts is not None and len(cnts) > 0:
      # I am assuming that the largest thing in the image, apart from the houses and the pieces, is a hand, but that is not an absolute truth.
      cnt = max(cnts, key=cv2.contourArea)
      return (True, cnt)
    else:
      return (False, None)

  def __updateFrameRate(self):
    now = time.time()
    dt = (now - self.__lastupdate)
    if dt <= 0:
      dt = 0.000000000001

    fps2 = 1.0 / dt
    self.__lastupdate = now
    self.__fps = self.__fps * 0.9 + fps2 * 0.1
    self.print('Mean Frame Rate:  {:.2f} FPS'.format(self.__fps), index=0)

  def __runScan(self, only_prediction: bool = False):
    print('scanning...')
    squares, self.__detections = self.__board.scan(self.__processed_image)
    board_state = self.__board.toMatrix(squares)
    #print(squares)
    print(board_state)

    if not only_prediction:
      human_move = self.__agent.state2Move(board_state)
      if human_move is not None:
        self.print('HUMAN: {}'.format(human_move.uci()))
        self.__agent.makeMove(human_move)
        self.__agent.updateState(board_state)

      cpu_move = self.__agent.chooseMove()
      if cpu_move is not None:
        self.print('BOT: {}'.format(cpu_move.uci()))
        print('BOT: {}'.format(cpu_move.uci()))
        self.__agent.makeMove(cpu_move)
        self.__agent.updateState(self.__agent.board.state())





  """
  Tests by Will 
  """


  def debugthiswill(self):
    """
    You need to try this yourself to better understand it
    """

    print("It's working")
    self.__camera = Camera(self.__cam_address)

    self.__running_calibration = ChessboardCalibration()
    found, self.__board = self.__running_calibration.loadMapping()
    if not found:
      raise Exception('No mapping found. Run calibration mapping')

    self.__captureUndistortedFrame()
    self.__runScan(only_prediction=True)
    self.show()

  def __captureUndistortedFrame(self):
    undistorted_frame = self.__camera.capture_test_fisheye()
    self.__processed_image = self.__running_calibration.applyMapping(undistorted_frame)

    result, hand_is_detected = self.__addHandBoundingBoxes(self.__processed_image)
    if hand_is_detected:
      self.__scheduleScan()
    else:
      result = self.__addPiecesBoundingBoxes(self.__processed_image)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    self.setImage(result)
    self.__updateFrameRate()

    QtCore.QTimer.singleShot(1, self.__captureUndistortedFrame)


  def Debug_using_image(self):
    image_path = 'New_chessboard/1_raw.jpg'
    working_image = cv2.imread(image_path)
    if working_image is None:
            raise FileNotFoundError(f"Image non trouvée : {image_path}")

    
    self.__running_calibration = ChessboardCalibration()    
    found, self.__board = self.__running_calibration.loadMapping()
    if not found:
      raise Exception('No mapping found. Run calibration mapping')
    
    #self.__processed_image = self.__running_calibration.applyMapping(frame)
    self.__processed_image = self.__running_calibration.applyMapping(working_image)
    #result = self.__addPiecesBoundingBoxes(self.__processed_image)

    result, hand_is_detected = self.__addHandBoundingBoxes(self.__processed_image)
    if hand_is_detected:
      self.__scheduleScan()
    else:
      result = self.__addPiecesBoundingBoxes(self.__processed_image)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    

    save_result = cv2.imwrite('Image_saved/Check_result.jpg', result)
    if save_result:
        print(f"Image sauvegardée avec succès : {'Image_saved/Check_result.jpg'}")
    else:
        raise IOError(f"Échec de la sauvegarde de l'image : {'Image_saved/Check_result.jpg'}")
    

    self.__runScan(only_prediction=True)
    #self.show()




    # Sauvegarder l'image modifiée
    output_path = 'Image_saved/2_working.jpg'
    success = cv2.imwrite(output_path, self.__processed_image)
    cv2.imwrite('Bounding_boxes/1_1_raw.jpg', self.__processed_image)
    if success:
        print(f"Image sauvegardée avec succès : {output_path}")
    else:
        raise IOError(f"Échec de la sauvegarde de l'image : {output_path}")
