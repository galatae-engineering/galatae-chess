Don't forget to activate the venv!
In terminal for windows : 
    venv\Scripts\Activate

First step is to map the board. The objective here is such that from a raw image (or camera frame) we'll isolate the chessboard and locate the position of every 64 boxes(/tiles) inside of it. The image transformations and coordinates are saved in a .json file (chessboard-mapping.json)  
To use the camera in input you have to uncomment the lines 75,76 and 90 of src/model/game.py , and comment the line 77. To change the image used, replace the image path in line 77 (frame = cv2.imread('New_chessboard/1_raw.jpg')).
Type in terminal :
    python src/main.py --mapping
So the mapping consists on multiple steps. You can check the results (images) of each one in the debug folder. Those methodes are located in src/model/chessboard_calibration.py . 
    - Finding the biggest countour (square)
    - Cropping the image to center the board and add a bit of padding to keep the infos of the borders. 
    - Find the corners of each boxes. I won't get into the details, but here's an idea of the steps used to obtain it (most of then are located in src/utils/__init__.py):
            - Gray scaling 
            - Gaussian blurr (to reduce noise)
            - canny_edge (algorithme used to define the countours)
            - hough_line (algorithme used to find straight lines in the image)
            - line_intersections (find the intersections of the lines found)
            - cluster_points


Now we are ready to start the detection of the pieces on the chessboard.

If you want to use the camera.
Type in terminal :
    python src/main.py --start

If you want to use an image. 
Type in terminal :
    python src/main.py --image
If you want to use another image you can change this line in src/model/game.py : image_path = 'New_chessboard/1_raw.jpg'





If you wonder, the images inside Image_saved and New_chessboard are just captures used to test the code.

















