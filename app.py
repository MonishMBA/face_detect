from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import random
import json
import pickle
from deepface import DeepFace
import requests

app = Flask(__name__)

# Classes 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

COMICAL_MESSAGES = {
    "happy": ["Keep smiling, it's contagious!", "Happiness looks great on you!"],
    "sad": ["Cheer up! Better days are coming.", "Don't worry, be happy!"],
    "angry": ["Take a deep breath... and let it go.", "Channel that energy wisely!"],
    "surprise": ["Whoa! What a surprise!", "Expect the unexpected!"],
    "neutral": ["Calm and composed, nice!", "Zen mode activated."],
    "disgust": ["That expression says, 'I'd rather not, thank you.'", "Someone must’ve told you a terrible joke."],
    "fear": ["Blink twice if you’re nervous!", "Ghosts are just shy introverts!"]
}

# # Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dummy function to simulate ML model
def detect_emotion(frame):
    # Replace this with your ML model logic
    start = time.time()
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    emotion_label = "Detecting.."
    elapsed = 0
    comical_message = "Let's see?!"
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion_label = result[0]['dominant_emotion']
        end = time.time(); elapsed = round((end-start) * 10**3,2)
        
        # Get a random comical message for the detected emotion
        comical_message = random.choice(COMICAL_MESSAGES.get(emotion_label, ["You're unique!"]))
        
        # Display the emotion label on the frame
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    return frame, emotion_label, elapsed, comical_message


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Open the uploaded image
            image = Image.open(BytesIO(file.read()))
            image_np = np.array(image)

            # Process the image using the emotion detection function
            processed_image, emotion_label, elapsed, comical_message = detect_emotion(image_np)

            # Convert processed image to a format that can be displayed directly
            output_image = Image.fromarray(processed_image)
            buffered = BytesIO()
            output_image.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode()

            # Embed the image in HTML using a data URI
            return render_template('upload.html', output_image=f"data:image/jpeg;base64,{encoded_image}",
                                   emotion_label=emotion_label, elapsed_time=elapsed, comical_message=comical_message)
    return render_template('upload.html')


@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame = request.files['frame']
    
    # Convert the received frame to a numpy array
    image = Image.open(frame)
    frame_np = np.array(image)

    # Process the frame to detect emotion
    processed_frame, emotion_label, elapsed, comical_message = detect_emotion(frame_np)

    # Return the result as a JSON response
    return jsonify({
        'emotion': emotion_label,
        'time_ms': elapsed
    })

if __name__ == '__main__':
    app.run()
