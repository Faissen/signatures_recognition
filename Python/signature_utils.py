import cv2 # OpenCV library for image processing
import os # For file path operations

def extract_features(image_path):
    """
    Loads an image, extracts ORB keypoints and descriptors.
    Returns (keypoints, descriptors).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load image in grayscale

   # Check if image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Create ORB detector, ORB stands for Oriented FAST and Rotated BRIEF
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None) # None means no mask is used
 
    #keypoints are the points of interest, descriptors are the feature vectors
    return keypoints, descriptors 


def compare_descriptors(desc1, desc2):
    """
    Compares two descriptor sets using BFMatcher.
    Returns a similarity score between 0 and 1.
    """
    #Normal Hamming distance is best for ORB descriptors
    # CrossCheck ensures that matches are mutual
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Matches descriptors between both sets
    matches = bf.match(desc1, desc2)
    # Sorts matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)
    # Selects only the best matches (distance < 50 is a good threshold)
    good_matches = [m for m in matches if m.distance < 50]

    # Calculate similarity score
    if len(matches) == 0: # Avoid division by zero
        return 0
    # Similarity is ratio of good matches to total matches
    return len(good_matches) / len(matches)
