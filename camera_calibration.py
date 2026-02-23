import cv2 
import numpy as np
import glob

# Number of inner corners 
board_dimension = (8, 6)  

# Size of a square on the chessboard in millimeters
square_size = 24  

# Criteria for refining corner locations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0)

points_3d = []  # Store 3D points in real world (object points)
points_2d = []  # Store 2D points in image plane (image points)

# Creates array to hold 3d coordinates for chessboard corners (start with all zeros)
obj_3d = np.zeros((board_dimension[0] * board_dimension[1], 3), np.float32)
# Fill the first two columns with the x and y coordinates (x,y,0)
obj_3d[:, :2] = np.mgrid[0:board_dimension[0], 0:board_dimension[1]].T.reshape(-1, 2)
# Multiply by real square size to get real world coordinates in mm
obj_3d = obj_3d * square_size

# Find all jpg images in folder
images = glob.glob('calibration_images/*.jpg')
print(f"Found {len(images)} images")

# If no images found, exit script
if not images:
    print("No images found. Check the path.")
    exit()

# Loop through each calibration image
for fname in images:
    img = cv2.imread(fname)  # Read image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Try to find the chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(gray, board_dimension, None)

    if ret:
        # If found, add object points (same for all images)
        points_3d.append(obj_3d)

        # Refine corner locations for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        points_2d.append(corners2)

# Only calibrate if at least one set of corners was found
if len(points_3d) > 0:
    # Get image size from last processed image
    img_height, img_width = gray.shape[:2]
    
    # Run camera calibration to get camera matrix, distortion coefficients, etc.
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        points_3d, points_2d, (img_width, img_height), None, None
    )

    print("\n=== Calibration Results ===")
    print(f"RMS re-projection error: {ret}")
    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs}")

    # Save calibration results
    np.savez('calibration_data.npz', 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)
else:
    print("No chessboard corners were detected in any image. Calibration failed.")