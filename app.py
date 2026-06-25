import os
import sys
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="Unified AI Intelligence Hub", layout="wide")

def ensure_correct_python():
    executable = os.path.normcase(sys.executable)
    if ".venv" not in executable:
        st.error("This app must be run with the workspace virtual environment.")
        st.markdown(
            "Please run the app from the project folder using:\\n```.venv\\Scripts\\python.exe -m streamlit run app.py```"
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
        Explore the power of Artificial Intelligence through a single interactive platform that brings together Machine Learning, Deep Learning, and Computer Vision. This dashboard features a house price prediction system using classical machine learning techniques, a YOLO-based object detection model capable of identifying objects in images, and a deep learning model that recognizes rock-paper-scissors hand gestures.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🏠 Smart House Valuation", use_container_width=True):
            go_to("house")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🎯 Real-Time Object Detection", use_container_width=True):
            go_to("yolo")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✋ Gesture Recognition AI", use_container_width=True):
            go_to("rps")

elif st.session_state.page == "house":

    left, right = st.columns(2)

    with left:
        st.markdown("# House Price Prediction")
        st.write("Enter the house features below to predict the price using the classical ML model from the notebook.")
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

    st.title("🎯 Real-Time Object Detection (YOLO)")

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

    st.title("Rock Paper Scissors")

    image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if image:
        st.image(image)

    if st.button("Back"):
        go_to("home")