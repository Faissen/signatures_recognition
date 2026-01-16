import cv2  # OpenCV library for image processing

# Loads two signature images in grayscale
# Both images must be in the same folder as this script
img1 = cv2.imread("signature1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("signature2.png", cv2.IMREAD_GRAYSCALE)

# Checks if both images were loaded correctly
if img1 is None or img2 is None:
    raise FileNotFoundError("One or both signature images could not be loaded.")

# Creates an ORB detector to extract keypoints and descriptors
orb = cv2.ORB_create()

# Detects keypoints and computes descriptors for both signatures
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Creates a Brute Force Matcher using Hamming distance (best for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Matches descriptors between both signatures
matches = bf.match(descriptors1, descriptors2)

# Sorts matches by distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Selects only the best matches (distance < 50 is a good threshold)
good_matches = [m for m in matches if m.distance < 50]

# Calculates similarity score based on the number of good matches
similarity = len(good_matches) / len(matches) if len(matches) > 0 else 0

print("Total matches:", len(matches))
print("Good matches:", len(good_matches))
print("Similarity score:", similarity)

# Draws the matches for visualization
result = cv2.drawMatches(
    img1, keypoints1,
    img2, keypoints2,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Shows the comparison result
cv2.imshow("Signature Comparison", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
