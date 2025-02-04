import cv2
import numpy as np
import time


class Camera:
    cap: cv2.VideoCapture
    resize_percent: float
    fps: int
    resolution: tuple
    previousMillis: int = 0
    stop: bool
    DIM = (640, 480)
    K = np.array([[502.94337692838667, 0.0, 365.2868612244552], [0.0, 504.59040139986365, 248.46569471393298], [0.0, 0.0, 1.0]])
    D = np.array([[-0.07126905916306937], [-0.19631725148173187], [0.7427177452391733], [-0.9441904283709754]])

    def __init__(self, cam_address, resize_percent=0.35, fps=10):
        """
        Initialize a `Camera` object

        @param cam_address The path of the camera can number, video file name or ip address
        @param resize_percent The percent of resize frame
        @param fps The number of frames that can get in 1 second
        """
        self.resize_percent = resize_percent
        self.fps = fps
        cam_index = 2
        self.cap = cv2.VideoCapture(cam_index)
        #self.cap = cv2.VideoCapture(cam_address)
        if self.cap.isOpened():
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.resolution = (height, width)
        else:
            raise Exception('Could not open camera.')

    def startRunning(self, callback):
        """
        Running a loop getting the next frame of the video camera

        @param callback The return function where frame captured is sent

        **OBS**
        This method lock execution the nexts lines of the caller
        """
        self.stop = False
        while True:
            currentMillis = round(time.time() * 1000)

            # check to see if it's time to capture the frame; that is, if the difference
            # between the current time and last time we capture the frame is bigger than
            # the interval at which we want to capture the frame.
            # The interval is a 1s divide by number of frames
            if currentMillis - self.previousMillis >= (1000 / self.fps):
                self.previousMillis = currentMillis

                # capture the next frame
                frame = self.capture()
                callback(frame, self)

            # check to see if it's time to break
            if self.stop:
                break



    def undistort_frame(self, frame, balance=0.0, dim2=None, dim3=None):
        """
        Apply distortion correction to a frame
        """
        dim1 = frame.shape[:2][::-1]  # Dimensions de l'image (largeur, hauteur)
        assert dim1[0] / dim1[1] == self.DIM[0] / self.DIM[1], \
            "Le flux vidéo doit avoir le même ratio d'aspect que l'image utilisée pour la calibration."

        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1

        # Mise à l'échelle de la matrice K en fonction de la taille du flux vidéo
        scaled_K = self.K * dim1[0] / self.DIM[0]
        scaled_K[2][2] = 1.0

        # Calcul de la nouvelle matrice de correction
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

        # Correction de la distorsion
        undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_frame


    def capture(self):
        """
        Return a single next frame of the capture camera
        """
        try:
            retval, frame = self.cap.read()

            if not retval:
                raise Exception('Could not get the next video frame.')

            # Apply distortion correction
            frame = self.undistort_frame(frame)

            # Define new resolution
            #height, width = self.resolution
            #new_w = int(width - (self.resize_percent * width))
            #new_h = int(height - (self.resize_percent * height))

            frame = cv2.resize(frame, (640, 480))
            return frame
        except:
            self.stopRunning()
            raise Exception('Could not get the next video frame.')
        
    def capture_test_fisheye(self):
        """
        Return a single next frame of the capture camera
        """
        try:
            retval, frame = self.cap.read()

            if not retval:
                raise Exception('Could not get the next video frame.')

            # Apply distortion correction
            undistorted_frame = self.undistort_frame(frame)

            return undistorted_frame
        except:
            self.stopRunning()
            raise Exception('Could not get the next video frame.')

    def stopRunning(self):
        """
        Stop running and clean the session
        """
        self.stop = True
        if self.cap:
            self.destroy()

    def destroy(self):
        """
        Release the camera
        """
        self.cap.release()