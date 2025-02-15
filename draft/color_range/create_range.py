import sys
sys.path.append('C:\Users\William\Desktop\Stage Galatae\Reconnaisance image new\chess_recognition-master\src')

import cv2
import imutils
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, uic, QtCore
from model import *
import json

app = QtWidgets.QApplication(sys.argv)
imageConfigScreen = uic.loadUi('draft/color_range/rangeUI.ui')
imageConfigScreen.show()

colorMapSelelect = imageConfigScreen.colorMapSelelect
colorMapSelelect.addItems([
  "AUTUMN",
  "BONE",
  "JET",
  "WINTER",
  "RAINBOW",
  "OCEAN",
  "SUMMER",
  "SPRING",
  "COOL",
  "HSV",
  "PINK",
  "HOT",
  "PARULA",
  "MAGMA",
  "INFERNO",
  "PLASMA",
  "VIRIDIS",
  "CIVIDIS",
  "TWILIGHT",
  "TWILIGHT_SHIFTED",
  "TURBO"
])

def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  return cv2.LUT(image, table)

def dictionary():
  gammaSlider = imageConfigScreen.gammaSlider
  invertImageCheckBox = imageConfigScreen.invertImageCheckBox
  hsvImageCheckBox = imageConfigScreen.hsvImageCheckBox
  rgbImageCheckBox = imageConfigScreen.rgbImageCheckBox
  colorMapCheckBox = imageConfigScreen.colorMapCheckBox

  lhSlider = imageConfigScreen.lhSlider
  lsSlider = imageConfigScreen.lsSlider
  lvSlider = imageConfigScreen.lvSlider
  lower = [lhSlider.value(), lsSlider.value(), lvSlider.value()]

  uhSlider = imageConfigScreen.uhSlider
  usSlider = imageConfigScreen.usSlider
  uvSlider = imageConfigScreen.uvSlider
  upper = [uhSlider.value(), usSlider.value(), uvSlider.value()]

  return {
    "gamma": gammaSlider.value(),
    "inverted": bool(invertImageCheckBox.checkState()),
    "hsv": bool(hsvImageCheckBox.checkState()),
    "rgb": bool(rgbImageCheckBox.checkState()),
    "colorMap": colorMapSelelect.currentIndex() if bool(colorMapCheckBox.checkState()) else 'null',
    "lower": lower,
    "upper": upper
  }

def saveConfig():
  with open('config.json', 'w') as f:
    f.write('%s' % json.dumps(dictionary()))
  print('saved!')
imageConfigScreen.saveButton.clicked.connect(saveConfig)

camera = Camera(cam_address=2)
calibration = ChessboardCalibration()
calibration.loadMapping()

original_frame = camera.capture()
original_frame = calibration.applyMapping(original_frame)

while True:
  frame = original_frame.copy()

  # --------
  # RGB image
  rgbImageCheckBox = imageConfigScreen.rgbImageCheckBox
  if bool(rgbImageCheckBox.checkState()):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # --------
  # apply gamma
  gammaSlider = imageConfigScreen.gammaSlider
  frame = adjust_gamma(frame, gammaSlider.value())

  # --------
  # invert image
  invertImageCheckBox = imageConfigScreen.invertImageCheckBox
  if bool(invertImageCheckBox.checkState()):
    frame = cv2.bitwise_not(frame)

  # --------
  # apply color map
  colorMapCheckBox = imageConfigScreen.colorMapCheckBox
  if bool(colorMapCheckBox.checkState()):
    frame = cv2.applyColorMap(frame, colorMapSelelect.currentIndex())

  # --------
  # HSV image
  hsvImageCheckBox = imageConfigScreen.hsvImageCheckBox
  if bool(hsvImageCheckBox.checkState()):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # --------
  # inRange
  lhSlider = imageConfigScreen.lhSlider
  lsSlider = imageConfigScreen.lsSlider
  lvSlider = imageConfigScreen.lvSlider
  lower = np.array([lhSlider.value(), lsSlider.value(), lvSlider.value()])

  uhSlider = imageConfigScreen.uhSlider
  usSlider = imageConfigScreen.usSlider
  uvSlider = imageConfigScreen.uvSlider
  upper = np.array([uhSlider.value(), usSlider.value(), uvSlider.value()])
  mask = cv2.inRange(frame, lower, upper)
  res = cv2.bitwise_and(frame, frame, mask=mask)
  cv2.imshow("res", res)
  
  if cv2.waitKey(1) == ord("q"):
    camera.destroy()
    cv2.destroyAllWindows()
    break

  cv2.imshow("frame", frame)

sys.exit(app.exec_())
