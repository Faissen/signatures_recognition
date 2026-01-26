import streamlit as st
import cv2
import numpy as np
from signature_utils import compare_all_signatures, extract_text_from_image

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Signature Recognition System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Hide Streamlit default menu + footer for a cleaner UI
# ---------------------------------------------------------
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Recognition Mode",
    ["Hybrid (Recommended)", "OCR Only", "Visual Only"]
)

st.sidebar.markdown("### Confidence Threshold")
st.sidebar.info("The system uses a fixed threshold of **60%** to validate matches.")
threshold = 0.60


debug = st.sidebar.checkbox("Show Debug Information", value=False)

st.sidebar.markdown("---")
st.sidebar.info("Upload a signature image to begin.")

# ---------------------------------------------------------
# Main Title
# ---------------------------------------------------------
st.title("🔍 Signature Recognition System")
st.write("Upload a signature image to identify the most likely matching name from the database.")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a signature image",
    type=["png", "jpg", "jpeg"]
)

# ---------------------------------------------------------
# Process uploaded image
# ---------------------------------------------------------
if uploaded_file is not None:

    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display uploaded image
    st.image(img, channels="BGR", caption="Uploaded Signature", use_column_width=True)

    # Save temporarily for processing
    temp_path = "temp_signature.png"
    cv2.imwrite(temp_path, img)

    # ---------------------------------------------------------
    # OCR Section
    # ---------------------------------------------------------
    st.subheader("📘 Extracted Text (OCR)")

    with st.spinner("Running OCR..."):
        text = extract_text_from_image(temp_path)

    if text:
        st.success(f"Detected text: **{text}**")
    else:
        st.warning("No readable text detected.")

    # ---------------------------------------------------------
    # Signature Recognition Section
    # ---------------------------------------------------------
    st.subheader("📊 Signature Identification Results")

    with st.spinner("Analyzing signature..."):
        result = compare_all_signatures(temp_path)

    # Display top matches
    for name, score in result["top_3_matches"]:
        st.markdown(f"### 🧾 {name}")
        st.progress(score / 100)
        st.write(f"Confidence: **{score:.1f}%**")
        st.markdown("---")

    # ---------------------------------------------------------
    # Debug Information
    # ---------------------------------------------------------
    if debug:
        with st.expander("🔧 Debug Information"):
            st.write("OCR Raw Output:", text)
            st.write("Recognition Mode:", mode)
            st.write("Threshold:", threshold)
            st.write("Raw Result Object:", result)
