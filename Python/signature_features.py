import cv2  # Imports the OpenCV library for image processing

# Loads the signature image in grayscale mode
# Grayscale is ideal for feature extraction because it removes color noise
image = cv2.imread("signature1.png", cv2.IMREAD_GRAYSCALE)

# Checks if the image was loaded correctly
if image is None:
    raise FileNotFoundError("Could not load the image. Check the file path.")

# Creates an ORB detector object
# ORB (Oriented FAST and Rotated BRIEF) is fast, free, and works well for signatures
orb = cv2.ORB_create()

# Detects keypoints and computes descriptors
# Keypoints = important points in the image
# Descriptors = numerical vectors that describe each keypoint
keypoints, descriptors = orb.detectAndCompute(image, None) #none means no mask is used

# Prints how many keypoints were found
print("Number of keypoints detected:", len(keypoints))

# Prints the shape of the descriptor matrix
# Rows = number of keypoints
# Columns = descriptor length (usually 32 or 64)
print("Descriptor matrix shape:", descriptors.shape)

# Draws the keypoints on the image for visualization
image_with_keypoints = cv2.drawKeypoints(
    image, keypoints, None, color=(0, 255, 0), flags=0
) # None means no mask is used, flags=0 means default drawing

# Displays the image with detected keypoints
# This window will show where ORB found important features
cv2.imshow("Signature Keypoints", image_with_keypoints)
cv2.waitKey(0) # Waits indefinitely until a key is pressed
cv2.destroyAllWindows() # Closes all OpenCV windows
