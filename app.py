import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def process_image(image, pose, draw_landmarks=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        if draw_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Calculate angles
        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        left_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y]
        
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        right_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Add angle text to image
        cv2.putText(image, f"L Elbow: {left_elbow_angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"R Elbow: {right_elbow_angle:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image, results.pose_landmarks is not None

def main():
    st.title("Advanced Human Pose Estimation")
    
    app_mode = st.sidebar.selectbox("Choose the App Mode",
        ["About App", "Run on Image", "Run on Video", "Run on Webcam"])
    
    if app_mode == "About App":
        st.markdown("This application uses MediaPipe for Human Pose Estimation.")
        st.markdown("**Created by Dhruv Tiwari**")
        st.markdown("---")
        st.markdown("## Instructions")
        st.markdown("1. Select Run on Image/Video/Webcam from the sidebar")
        st.markdown("2. Upload an image/video or use your webcam")
        st.markdown("3. Adjust the confidence threshold as needed")
        st.markdown("4. View the results with pose landmarks and angle measurements")
    
    elif app_mode == "Run on Image":
        confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown("## Output")
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            st.sidebar.text("Original Image")
            st.sidebar.image(image)
            
            with mp_pose.Pose(min_detection_confidence=confidence, min_tracking_confidence=confidence) as pose:
                image, pose_detected = process_image(image, pose)
            
            if pose_detected:
                st.image(image, channels="BGR", caption="Processed Image")
            else:
                st.warning("No pose detected in the image.")
    
    elif app_mode == "Run on Video":
        confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown("## Output")
        video_file_buffer = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        
        if video_file_buffer is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file_buffer.read())
            
            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            with mp_pose.Pose(min_detection_confidence=confidence, min_tracking_confidence=confidence) as pose:
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    frame, _ = process_image(frame, pose)
                    stframe.image(frame, channels="BGR")
    
    elif app_mode == "Run on Webcam":
        confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown("## Output")
        
        run = st.checkbox("Run on Webcam")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        with mp_pose.Pose(min_detection_confidence=confidence, min_tracking_confidence=confidence) as pose:
            while run:
                _, frame = camera.read()
                frame, _ = process_image(frame, pose)
                FRAME_WINDOW.image(frame, channels="BGR")
            
        camera.release()

if __name__ == "__main__":
    main()
