import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import uvicorn
from fastapi.responses import JSONResponse
import time
import sys

# Initialize FastAPI app
app = FastAPI()

# Load trained model
MODEL_PATH = "/home/ubuntu/isl_backend/model/sign_language_model5.h5"
model = load_model(MODEL_PATH, compile=False)  # Load without compiling

# Manually compile (if needed)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load actions (A-Z labels)
actions = np.array([chr(i) for i in range(65, 91)])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)


# Function to extract hand keypoints
def extract_hand_keypoints(image):
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image
    # results = hands.process(image_rgb)  # Use the correct image
    for _ in range(1):
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            # Bounding box initialization
            x_min_global, y_min_global = image.shape[1], image.shape[0]
            x_max_global, y_max_global = 0, 0
            padding = 10  # Minimum padding for bounding box

            keypoints = np.zeros(126)  # Default empty keypoints
            hand_areas = []  # Store areas of detected hands
            hand_bboxes = []  # Store bounding boxes of hands
            all_keypoints = []  # Store keypoints of hands

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = image.shape
                    hand_keypoints = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    ).flatten()
                    all_keypoints.append(hand_keypoints)

                    # Compute bounding box
                    x_min, y_min, x_max, y_max = w, h, 0, 0
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, y_min = min(x, x_min), min(y, y_min)
                        x_max, y_max = max(x, x_max), max(y, y_max)

                    # Apply padding in all directions
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Compute area of the bounding box
                    area = (x_max - x_min) * (y_max - y_min)
                    hand_areas.append(area)
                    hand_bboxes.append((x_min, y_min, x_max, y_max))

                if len(all_keypoints) == 2:
                    # Calculate distance between the closest points of the two hands
                    bbox1, bbox2 = hand_bboxes
                    distance = max(0, bbox2[0] - bbox1[2], bbox1[0] - bbox2[2]) + max(
                        0, bbox2[1] - bbox1[3], bbox1[1] - bbox2[3]
                    )

                    if distance > 20:  # Threshold for contact (adjustable)
                        larger_hand_idx = np.argmax(hand_areas)
                        keypoints[
                            : len(all_keypoints[larger_hand_idx])
                        ] = all_keypoints[larger_hand_idx]
                        (
                            x_min_global,
                            y_min_global,
                            x_max_global,
                            y_max_global,
                        ) = hand_bboxes[larger_hand_idx]
                    else:
                        keypoints[: len(all_keypoints[0])] = all_keypoints[0]
                        keypoints[63 : 63 + len(all_keypoints[1])] = all_keypoints[1]
                        x_min_global = (
                            min(hand_bboxes[0][0], hand_bboxes[1][0]) - padding
                        )
                        y_min_global = (
                            min(hand_bboxes[0][1], hand_bboxes[1][1]) - padding
                        )
                        x_max_global = (
                            max(hand_bboxes[0][2], hand_bboxes[1][2]) + padding
                        )
                        y_max_global = (
                            max(hand_bboxes[0][3], hand_bboxes[1][3]) + padding
                        )
                else:
                    keypoints[: len(all_keypoints[0])] = all_keypoints[0]
                    x_min_global = hand_bboxes[0][0] - padding
                    y_min_global = hand_bboxes[0][1] - padding
                    x_max_global = hand_bboxes[0][2] + padding
                    y_max_global = hand_bboxes[0][3] + padding

            return keypoints, [x_min_global, y_min_global, x_max_global, y_max_global]
        else:
            time.sleep(0.1)  # Small delay before retrying

    return None, None


# Function to predict sign language character
# def predict_sign(image):
#     keypoints, bbox = extract_hand_keypoints(image)
#     if keypoints is None:
#         return None, None, None

#     # if np.max(keypoints) > 0:  # Avoid division by zero
#     #     keypoints = keypoints / np.max(keypoints)

#     keypoints = keypoints.reshape(1, -1)  # Reshape for model
#     prediction = model.predict(keypoints)
#     confidence_score = np.max(prediction)
#     if confidence_score <= 20:
#         prediction = model.predict(keypoints)

#     predicted_class = np.argmax(prediction)

#     print(
#         f"Predicted Class: {predicted_class}, Confidence: {confidence_score}"
#     )  # Debugging
#     return actions[predicted_class], bbox, confidence_score


def predict_sign(image):
    """Predicts sign from image with retries if confidence is too low."""

    for _ in range(3):
        keypoints, bbox = extract_hand_keypoints(image)

        if keypoints is None:
            print("No Hands Detected, retrying...")
            time.sleep(0.1)  # Small delay before retrying
            continue  # Retry if no keypoints found

        # Normalize keypoints (Prevent division by zero)
        # norm_factor = np.linalg.norm(keypoints)
        # if norm_factor > 0:
        #     keypoints = keypoints / norm_factor

        keypoints = keypoints.reshape(1, -1)  # Reshape for model input

        # Predict
        prediction = model.predict(keypoints)
        confidence_score = np.max(prediction)

        if confidence_score > 0.2:  # Threshold for confidence
            predicted_class = np.argmax(prediction)
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence_score}")
            return actions[predicted_class], bbox, confidence_score

        print(f"Low confidence ({confidence_score}), retrying...")
        time.sleep(0.1)  # Small delay before retrying

    # Return failure case after retries
    print("Failed to classify sign after retries.")
    return None, None, None


# FastAPI endpoint for image upload
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = np.array(image, dtype=np.uint8)  # Convert PIL to NumPy

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(f"Image shape: {image.shape}")

        prediction, bbox, confidence_score = predict_sign(image)
        if prediction is None:
            return {}

        response = {
            "prediction": str(prediction),
            "bbox": [int(coord) for coord in bbox],
            "confidence": float(confidence_score),
        }
        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
