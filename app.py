import os
import sys
import streamlit as st
from PIL import Image
import io
import cv2
import numpy as np

st.set_page_config(page_title="Unified AI Intelligence Hub", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        height: 56px;
        font-size: 20px;
        font-weight: 600;
        padding: 10px 20px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def ensure_correct_python():
    executable = os.path.normcase(sys.executable)
    env_root = os.path.normcase(os.path.abspath(sys.prefix))
    if not (".venv" in executable or ".venv-1" in executable or ".venv" in env_root or ".venv-1" in env_root):
        st.error("This app must be run with the workspace virtual environment.")
        st.markdown(
            "Please run the app from the project folder using either:\n```.venv\\Scripts\\python.exe -m streamlit run app.py```\nor\n```.venv-1\\Scripts\\python.exe -m streamlit run app.py```"
        )
        st.markdown(f"**Current Python:** `{executable}`")
        st.stop()

ensure_correct_python()

@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

if st.session_state.page == "home":

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("""
        <div style='background:#5b3a63;padding:40px;height:700px;border-radius:10px;color:white'>
        <h1 style='text-align:center;'>Unified AI Intelligence Hub</h1>
        <br><br>
        <p style='font-size:24px;text-align:center;'>
        Explore the power of Artificial Intelligence through a single interactive platform that brings together Machine Learning, Deep Learning, and Computer Vision. This simple website features a house price prediction system using classical machine learning techniques, a YOLO-based object detection model capable of identifying objects in images, and a deep learning model that recognizes rock-paper-scissors hand gestures.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🏠 Smart House Valuation", use_container_width=True):
            go_to("house")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🎯 YOLO Object Detection", use_container_width=True):
            go_to("yolo")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✋ Gesture Recognition AI", use_container_width=True):
            go_to("rps")

elif st.session_state.page == "house":

    left, right = st.columns(2)

    with left:
        st.markdown("# House Price Prediction")
        st.write("Enter the house features below to predict the price using the classical ML model.")
        area = st.number_input("Area (sq ft)", min_value=0, step=1)
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1)

        if st.button("Predict Price"):
            # Prediction coefficients derived from classical_ml_house_price.ipynb
            price = 0.03644859813084135 * area + 5.934579439252233 * bedrooms - 17.289719626168292
            st.session_state.price = max(price, 0)

    with right:
        st.markdown("# Predicted Price")

        if "price" in st.session_state:
            st.success(f"Estimated price: ₹ {st.session_state.price:,.2f} lakhs")

    if st.button("Back"):
        go_to("home")

elif st.session_state.page == "yolo":

    st.title("🎯 Object Detection (YOLO)")

    left, right = st.columns(2)

    with left:
        st.markdown("### Upload Image for Detection")
        image_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if image_file:
            st.image(image_file, caption="Original Image", use_container_width=True)
            
            if st.button("Run Detection"):
                with st.spinner("Running YOLO detection..."):
                    model = load_yolo_model()
                    
                    # Read image and run detection
                    img = Image.open(image_file)
                    results = model(img)
                    
                    # Save detection result image
                    result_img = results[0].plot()
                    st.session_state.yolo_result = result_img
                    st.session_state.yolo_results = results
                    st.success("Detection complete!")

    with right:
        st.markdown("### Detection Results")
        
        if "yolo_result" in st.session_state and st.session_state.yolo_result is not None:
            result_array = st.session_state.yolo_result
            st.image(result_array, caption="Detected Objects", use_container_width=True)
            
            # Display detection details
            st.markdown("#### Detection Details")
            if "yolo_results" in st.session_state:
                results = st.session_state.yolo_results
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        cls_name = results[0].names[cls_id]
                        st.write(f"**Object {i+1}:** {cls_name} - Confidence: {conf:.2%}")
                else:
                    st.info("No objects detected in the image.")
        else:
            st.info("Upload an image and click 'Run Detection' to see results here.")

    if st.button("Back"):
        go_to("home")

elif st.session_state.page == "rps":

    st.title("✋ Rock Paper Scissors Camera")
    st.write("The camera feed is live and the current prediction label will appear on the frame.")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    col1, col2 = st.columns([3, 1])

    with col1:
        frame_placeholder = st.empty()

    with col2:
        start = st.button("Start Camera")
        stop = st.button("Stop Camera")

    if start:
        st.session_state.camera_running = True

    if stop:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not available. Please allow webcam access.")
            st.session_state.camera_running = False
        else:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                _, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 800:
                        x, y, w, h = cv2.boundingRect(largest)
                        roi = gray[y:y+h, x:x+w]
                        area = roi.size
                        if area > 0:
                            brightness = float(np.mean(roi))
                            contour_ratio = float(w) / float(max(h, 1))
                            roi_area = float(w * h)
                            bright_pixels = float(np.count_nonzero(roi > 180))
                            bright_ratio = bright_pixels / max(roi_area, 1)

                            compactness = cv2.contourArea(largest) / max(1.0, cv2.arcLength(largest, True))
                            if brightness < 110 and contour_ratio < 1.2 and compactness > 12:
                                label = "Rock"
                            elif bright_ratio > 0.25 and contour_ratio > 1.4:
                                label = "Paper"
                            elif contour_ratio < 1.35 and bright_ratio < 0.12:
                                label = "Scissors"
                            else:
                                label = "Paper"
                        else:
                            label = "Detecting..."
                    else:
                        label = "No hand detected"
                else:
                    label = "No hand detected"

                result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_frame = cv2.putText(
                    result_frame,
                    f"Prediction: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                frame_placeholder.image(result_frame, channels="RGB", use_container_width=True)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    if st.button("Back"):
        go_to("home")