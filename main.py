import sys
sys.path.append('../galatae-api/')
from robot import Robot
import time

print('Salut Roger')

r=Robot('/dev/ttyACM0')
r.reset_pos()
r.calibrate_gripper()
#r.go_to_point([0,0,600,0,0])
#r.go_to_point([300,0,300,90,0])
r.go_to_point([300,0,300,180,0])
time.sleep(1)
r.go_to_point([250,0,30,180,0])
time.sleep(1)
r.close_gripper()
time.sleep(1)
r.go_to_point([300,0,300,180,0])
r.go_to_point([280,100,30,180,0])
r.open_gripper()
time.sleep(1)
r.go_to_foetus_pos()