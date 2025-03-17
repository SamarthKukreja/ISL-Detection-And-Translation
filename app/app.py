import streamlit as st
import requests
import io
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# FastAPI Backend URL
API_URL = "http://16.170.236.135:8000/predict/"  # Change this after deployment

st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🖐️ Sign Language Detection")
st.markdown("Upload an image or start real-time detection!")

# File Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Initialize session state for video streaming
if "video_active" not in st.session_state:
    st.session_state.video_active = False


# Function to preprocess image
def preprocess_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    return {"file": image_bytes.getvalue()}


# Video Frame Processing Class
class SignLanguageProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV BGR format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Convert Image to Bytes for API
        files = preprocess_image(pil_img)

        # Send to API
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            if result:
                predicted_sign = result["prediction"]
                confidence = result.get("confidence", 1.0) * 100
                bbox = result.get("bbox", None)

                if bbox:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{predicted_sign} ({confidence:.2f}%)",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
        return frame.from_ndarray(img, format="bgr24")


# Start Video Streaming
if st.button("Start Video"):
    st.session_state.video_active = not st.session_state.video_active  # Toggle state

if st.session_state.video_active:
    webrtc_streamer(
        key="sign-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

if uploaded_file is not None:
    # Read image & Display
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_container_width =True)
    st.write("")
    st.markdown("### Processing Image... 🔄")

    # Convert image to bytes for API request
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    files = {"file": image_bytes.getvalue()}

    # Send Image to FastAPI Backend
    with st.spinner("Detecting Sign... ⏳"):
        response = requests.post(API_URL, files=files)

    # Process Response
    if response.status_code == 200:
        result = response.json()
        if result == {}:
            st.error("No Hands Detected")
        else:
            predicted_sign = result["prediction"]
            confidence = result.get("confidence", 1.0) * 100
            bbox = result.get("bbox", None)

            # Display Prediction Result
            st.success(f"Predicted Sign: **{predicted_sign}**")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

            # Draw Bounding Box on Image
            if bbox:
                image_cv = np.array(image)
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                image = Image.fromarray(image_cv)
                st.image(image, caption="Detected Hand")

    else:
        st.error("⚠️ Failed to process the image. Please try again!")

# Sidebar for Info
st.sidebar.header("About the Project")
st.sidebar.info(
    "This is a Sign Language Detection app built using **FastAPI & Streamlit**. "
    "Upload an image, and the model will predict the sign language character."
)
