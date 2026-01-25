import cv2
import numpy as np
import os
import json

from skimage.metrics import structural_similarity as ssim

# 1. NORMALIZE SIGNATURE
def normalize_signature(image_path):
    """
    Loads and normalizes a signature:
    - grayscale
    - binarize
    - morphology closing (fix broken strokes)
    - crop to content
    - center on a fixed canvas
    - resize to consistent size
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # 1. Binarize (invert so ink = white)
    _, th = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Morphology closing to connect broken strokes
    kernel = np.ones((5, 15), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # 3. Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # fallback: resize original
        resized = cv2.resize(img, (600, 180), interpolation=cv2.INTER_AREA)
        return resized

    # 4. Bounding box around all contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = img[y:y+h, x:x+w]

    # 5. Resize cropped signature (preserving aspect ratio)
    target_w, target_h = 600, 180
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6. Center the resized signature on a fixed canvas
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


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


# 4. MAIN COMPARISON FUNCTION
# ---------------------------------------------------------
def compare_signatures_letters(img1, img2):
    """
    Full signature comparison based on letter shapes.
    """

    letters1 = segment_letters(img1)
    letters2 = segment_letters(img2)

    return compare_letters(letters1, letters2)

# 5. EXTRACT FEATURES (wrapper)
# ---------------------------------------------------------
def extract_features(image_path):
    img = normalize_signature(image_path)

    # Quality check: enough ink pixels
    non_white = cv2.countNonZero(255 - img)
    quality_good = non_white > 300

    return img, quality_good

def compare_ssim(img1, img2):
    img1 = cv2.resize(img1, (400, 120))
    img2 = cv2.resize(img2, (400, 120))
    score, _ = ssim(img1, img2, full=True)
    return score * 100

def compare_ssim_full(img1, img2):
    img1 = cv2.resize(img1, (600, 180))
    img2 = cv2.resize(img2, (600, 180))
    score, _ = ssim(img1, img2, full=True)
    return score * 100

def compare_template_full(img1, img2):
    img1 = cv2.resize(img1, (600, 180))
    img2 = cv2.resize(img2, (600, 180))

    res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    return float(res.max() * 100)

def is_cursive(img):
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assinaturas cursivas tendem a ter 1–3 contornos grandes
    return len(contours) <= 3

# 6. COMPARE ALL SIGNATURES IN DB
def compare_all_signatures(query_signature_path, database_path="../generated_signatures"):
    """
    Hybrid signature comparison:
    - If signature is cursive → use global SSIM + global template matching
    - If signature is non-cursive → use letters + SSIM
    """

    query_img, quality = extract_features(query_signature_path)
    if not quality:
        return {"status": "error", "message": "Signature quality too low."}

    query_is_cursive = is_cursive(query_img)

    results = []

    for filename in os.listdir(database_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(database_path, filename)
        db_img, db_quality = extract_features(file_path)

        db_is_cursive = is_cursive(db_img)

        # --- Cursive signatures: global comparison ---
        if query_is_cursive or db_is_cursive:
            similarity = compare_template_full(query_img, db_img)

        # --- Non-cursive: letter-based + SSIM ---
        else:
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

