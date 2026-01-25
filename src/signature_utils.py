import cv2
import numpy as np
import os
import json


# ---------------------------------------------------------
# 1. NORMALIZE SIGNATURE
# ---------------------------------------------------------
def normalize_signature(image_path):
    """
    Loads and normalizes a signature:
    - grayscale
    - binarize
    - crop to content
    - resize to fixed size
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Binarize
    _, th = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return cv2.resize(img, (400, 120))

    # Bounding box around all contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = img[y:y+h, x:x+w]

    # Resize to fixed size
    resized = cv2.resize(cropped, (400, 120), interpolation=cv2.INTER_AREA)

    return resized


# ---------------------------------------------------------
# 2. SEGMENT LETTERS
# ---------------------------------------------------------
def segment_letters(img):
    """
    Splits a normalized signature into individual letters.
    Returns a list of letter images ordered left-to-right.
    """

    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter out noise
        if w < 5 or h < 20:
            continue

        letter = img[y:y+h, x:x+w]
        letters.append((x, letter))

    # Sort by x position (left to right)
    letters.sort(key=lambda x: x[0])

    return [l for _, l in letters]


# ---------------------------------------------------------
# 3. COMPARE LETTERS
# ---------------------------------------------------------
def compare_letters(letters1, letters2):
    """
    Compares two lists of letters using template matching.
    Returns similarity 0–100.
    """

    if len(letters1) == 0 or len(letters2) == 0:
        return 0

    n = min(len(letters1), len(letters2))
    scores = []

    for i in range(n):
        l1 = cv2.resize(letters1[i], (40, 60))
        l2 = cv2.resize(letters2[i], (40, 60))

        res = cv2.matchTemplate(l1, l2, cv2.TM_CCOEFF_NORMED)
        scores.append(res.max())

    return round(sum(scores) / len(scores) * 100, 2)


# ---------------------------------------------------------
# 4. MAIN COMPARISON FUNCTION
# ---------------------------------------------------------
def compare_signatures_letters(img1, img2):
    """
    Full signature comparison based on letter shapes.
    """

    letters1 = segment_letters(img1)
    letters2 = segment_letters(img2)

    return compare_letters(letters1, letters2)


# ---------------------------------------------------------
# 5. EXTRACT FEATURES (wrapper)
# ---------------------------------------------------------
def extract_features(image_path):
    img = normalize_signature(image_path)

    # Quality check: enough ink pixels
    non_white = cv2.countNonZero(255 - img)
    quality_good = non_white > 300

    return img, quality_good


# ---------------------------------------------------------
# 6. COMPARE AGAINST DATABASE
# ---------------------------------------------------------
def compare_all_signatures(query_signature_path, database_path="../generated_signatures"):
    """
    Compares a new signature against all stored signatures using letter-based matching.
    Returns the top 3 most similar matches.
    """

    query_img, quality = extract_features(query_signature_path)
    if not quality:
        return {"status": "error", "message": "Signature quality too low."}

    results = []

    for filename in os.listdir(database_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(database_path, filename)
        db_img, db_quality = extract_features(file_path)

        similarity = compare_signatures_letters(query_img, db_img)
        results.append((filename, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    top_3 = results[:3]

    # Load name mapping
    mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "signature_names.json"))
    with open(mapping_path, "r", encoding="utf-8") as f:
        name_map = json.load(f)

    top_3_named = [(name_map.get(f, "Unknown"), score) for f, score in top_3]

    return {"top_3_matches": top_3_named}

