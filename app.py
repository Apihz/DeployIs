import time
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import asyncio
import av
from typing import Optional, Tuple, List, Dict, Any

# Page config
st.set_page_config(page_title="Real-time Student Attention Detection", layout="wide", page_icon="📸")

# Classes for the model - make sure these match your model training
classes = ['bored', 'confused', 'drowsy', 'engaged', 'frustrated', 'Looking away']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with better error handling
@st.cache_resource
def load_model():
    """Load the attention detection model"""
    model_path = "attention_resnet18.pth"
    
    try:
        # Create model architecture
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        return model
        
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()

# Load face cascade
@st.cache_resource
def load_face_cascade():
    """Load Haar Cascade for face detection"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("Failed to load face cascade classifier")
            return None
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return None

face_cascade = load_face_cascade()

# Transform for preprocessing
webcam_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Using the same normalization as your notebook
])
transform = webcam_transform

# Video Transformer for real-time prediction
class AttentionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.transform = webcam_transform
        self.device = device
        self.classes = classes
        self.face_cascade = face_cascade
        
    def recv(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # If model or face cascade is not loaded, just return original frame
            if self.model is None or self.face_cascade is None:
                cv2.putText(img, "Model or Face Cascade not loaded", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                # No faces detected
                cv2.putText(img, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Crop face from original frame
                face_img = img[y:y+h, x:x+w]
                
                # Skip if face is too small
                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    continue
                
                # Convert BGR to RGB for model (face_img is already in BGR format)
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Preprocess face image
                input_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    pred_idx = predicted.item()
                    confidence = probs[pred_idx].item()
                
                # Get emotion label
                emotion = self.classes[pred_idx]
                
                # Choose color based on attention state
                if emotion == 'engaged':
                    color = (0, 255, 0)  # Green for engaged
                elif emotion in ['bored', 'drowsy', 'Looking away']:
                    color = (0, 0, 255)  # Red for negative states
                elif emotion in ['confused', 'frustrated']:
                    color = (0, 165, 255)  # Orange for neutral/concerning states
                else:
                    color = (255, 255, 0)  # Yellow for other states
                
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Create label with confidence
                label = f"{emotion}: {confidence:.1%}"
                
                # Put emotion text above rectangle
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
                
                # Add confidence bar below the face
                bar_width = int((w * confidence))
                cv2.rectangle(img, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1)
                cv2.rectangle(img, (x, y + h + 5), (x + w, y + h + 15), (255, 255, 255), 1)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # If any error occurs, return original frame with error message
            cv2.putText(img, f"Error: {str(e)[:40]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
def predict_attention_state(image: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Predict attention state from face image
    
    Args:
        image: Face image as numpy array (RGB format)
    
    Returns:
        Tuple of (predicted_class, confidence_score)
    """
    if model is None:
        return None, 0.0
    
    try:
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        return classes[predicted_class_idx], confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0.0

def detect_and_analyze_faces(image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Detect faces in image and analyze attention state
    
    Args:
        image: Input image as numpy array (BGR format)
    
    Returns:
        Tuple of (processed_image, analysis_results)
    """
    if face_cascade is None:
        return image, [{"error": "Face detector not available"}]
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    processed_image = image.copy()
    
    if len(faces) == 0:
        results.append({
            "message": "No faces detected",
            "type": "info",
            "color": (255, 165, 0)  # Orange
        })
        return processed_image, results
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        try:
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            # Skip very small faces
            if face_region.shape[0] < 50 or face_region.shape[1] < 50:
                continue
            
            # Convert BGR to RGB for model
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Predict attention state
            attention_state, confidence = predict_attention_state(face_rgb)
            
            if attention_state:
                # Define colors for different states
                color_map = {
                    'engaged': (0, 255, 0),      # Green
                    'bored': (0, 0, 255),        # Red
                    'drowsy': (0, 0, 255),       # Red
                    'Looking away': (0, 0, 255), # Red
                    'confused': (0, 165, 255),   # Orange
                    'frustrated': (0, 165, 255)  # Orange
                }
                
                color = color_map.get(attention_state, (255, 255, 0))  # Default: Yellow
                
                # Draw face rectangle
                cv2.rectangle(processed_image, (x, y), (x+w, y+h), color, 3)
                
                # Add label
                label = f"{attention_state}: {confidence:.1%}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Background for text
                cv2.rectangle(processed_image, (x, y-35), (x + label_size[0] + 10, y), color, -1)
                cv2.putText(processed_image, label, (x + 5, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Confidence bar
                bar_width = int(w * confidence)
                cv2.rectangle(processed_image, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1)
                cv2.rectangle(processed_image, (x, y + h + 5), (x + w, y + h + 15), (255, 255, 255), 2)
                
                results.append({
                    "face_id": i + 1,
                    "attention_state": attention_state,
                    "confidence": confidence,
                    "bbox": (x, y, w, h),
                    "color": color,
                    "type": "prediction"
                })
            
        except Exception as e:
            results.append({
                "face_id": i + 1,
                "error": f"Analysis failed: {str(e)}",
                "type": "error"
            })
    
    return processed_image, results

class CameraManager:
    """Simple camera management class"""
    
    def __init__(self):
        self.cap = None
        self.is_active = False
    
    def start(self) -> bool:
        """Start camera capture, automatically detect available camera"""
        try:
            # Try opening the first available camera index
            for camera_index in range(10):  # Check up to 10 camera indexes
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 60)
                    self.is_active = True
                    return True
            
            # If no camera found
            return False
            
        except Exception:
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if self.cap and self.is_active:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None
    
    def stop(self):
        """Stop camera and release resources"""
        self.is_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
      
def display_analysis_results(results: List[Dict[str, Any]]):
    """Display analysis results in a formatted way"""
    
    if not results:
        st.warning("No analysis results available")
        return
    
    predictions = [r for r in results if r.get("type") == "prediction"]
    errors = [r for r in results if r.get("type") == "error"]
    info_messages = [r for r in results if r.get("type") == "info"]
    
    # Display predictions
    if predictions:
        st.subheader("Attention Analysis Results")
        
        # Break the layout into a single column, avoiding nested columns
        for result in predictions:
            face_id = result["face_id"]
            state = result["attention_state"]
            confidence = result["confidence"]
            
            # Display the results using different Streamlit components
            st.metric(f"Face {face_id}", state.title())
            
            # Progress bar for confidence
            st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            
            # Status indicator based on attention state
            if state == 'engaged':
                st.success("✅ Focused")
            elif state in ['bored', 'drowsy', 'Looking away']:
                st.error("Please lock in bro 🙏🥀🥀")
            else:
                st.warning("⚠️ Need focus more")

    # Display info messages
    for msg in info_messages:
        st.info(msg["message"])
    
    # Display errors
    for error in errors:
        st.error(f"Face {error['face_id']}: {error['error']}")

# Main function
def main():
    st.title("Real-time Student Attention Detection")
    st.markdown("---")
    # Sidebar content
    with st.sidebar:
        st.header("Information")
        st.write("This app detects student attention states in real-time using your webcam.")
        
        st.subheader("Detected States: ")
        state_colors = {
            'engaged': '🟢',
            'bored': '🔴',
            'confused': '🟡',
            'drowsy': '🔴',
            'frustrated': '🟡',
            'Looking away': '🔴'
        }
        
        for class_name in classes:
            emoji = state_colors.get(class_name, '⚪')
            st.write(f"{emoji} {class_name}")
        
        st.subheader("Requirements:")
        st.write("- Webcam access")
        st.write("- Good lighting")
        st.write("- Clear face visibility")
        st.write("- Model file: **attention_resnet18.pth**")
        
        st.subheader("System Status:")
        st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"Model: {'Loaded' if model is not None else 'Not Loaded'}")
        st.write(f"Face Detection: {'Ready' if face_cascade is not None else 'Failed'}")

    # Create layout with two columns
    colx1, colx2 = st.columns([4, 1])

    with colx1:
        tab1, tab2 = st.tabs(["📹 Live Camera", "📁 Upload Image"])

        with tab1:
                st.subheader("Live Camera Feed")        
                if model is None:
                    st.error("Model not loaded! Please ensure 'attention_resnet18.pth' is in the app directory.")
                elif face_cascade is None:
                    st.error("Face detection not available!")
                else:
                    st.success("System ready for attention detection!")
                
                RTC_CONFIGURATION = {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:stun3.l.google.com:19302"]},
                        {"urls": ["stun:stun4.l.google.com:19302"]},
                    ]
                }

                ctx = webrtc_streamer(
                    key="attention-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=AttentionTransformer,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 480, "ideal": 640, "max": 1280},
                            "height": {"min": 360, "ideal": 480, "max": 720},
                            "frameRate": {"min": 30, "ideal": 30, "max": 60}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )


                st.markdown("---")
                st.subheader("Frame capture")        


                        # Check system readiness
                if model is None or face_cascade is None:
                    st.error("System not ready. Please check model and face detection status in sidebar.")
                    return
                
                # Initialize camera manager in session state
                if 'camera_manager' not in st.session_state:
                    st.session_state.camera_manager = CameraManager()

                col1, col2, col3 = st.columns(3)
        
                with col1:
                    if st.button("Start Camera", type="primary", use_container_width=True):
                        if st.session_state.camera_manager.start():
                            st.success("Camera started!")
                            st.rerun()
                        else:
                            st.error("Failed to start camera")
                
                with col2:
                    if st.button("Capture & Analyze", use_container_width=True):
                        frame = st.session_state.camera_manager.capture_frame()
                        if frame is not None:
                            st.session_state.captured_frame = frame
                            st.session_state.frame_timestamp = time.time()
                            st.success("Frame captured!")
                            st.rerun()
                        else:
                            st.error("No frame available")
                
                with col3:
                    if st.button("Stop Camera", use_container_width=True):
                        st.session_state.camera_manager.stop()
                        st.success("Camera stopped!")
                        if 'captured_frame' in st.session_state:
                            del st.session_state.captured_frame
                        st.rerun()
                
                
                # Display captured frame and analysis
                if 'captured_frame' in st.session_state:
                    
                    # Process the frame
                    with st.spinner("Analyzing attention states..."):
                        processed_frame, analysis_results = detect_and_analyze_faces(
                            st.session_state.captured_frame
                        )
                    
                    # Display results
                    col_img, col_results = st.columns([3, 2])
                    
                    with col_img:
                        st.subheader("Analysis Result")
                        # Convert BGR to RGB for display
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        st.image(display_frame, caption=f"Captured at {time.ctime(st.session_state.frame_timestamp)}", 
                                use_container_width=True)
                    
                    with col_results:
                        display_analysis_results(analysis_results)
                
                else:
                    st.info("Click 'Start Camera' then 'Capture & Analyze' to begin analysis")
    
        
        with tab2:
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                input_image = webcam_transform(image_np).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_image)
                    _, predicted = torch.max(output, 1)
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    pred_idx = predicted.item()
                    confidence = probs[pred_idx].item()
                
                attention_state = classes[pred_idx]
                
                st.markdown("---")
                img_col, result_col = st.columns([1.2, 1])
                with img_col:
                    st.subheader("📸 Analyzed Image")
                    st.image(image, use_container_width=True)
                
                with result_col:
                    state_config = {
                        'engaged': {'emoji': '🎯', 'color': 'green', 'status': 'success'},
                        'bored': {'emoji': '😴', 'color': 'red', 'status': 'error'},
                        'confused': {'emoji': '🤔', 'color': 'orange', 'status': 'warning'},
                        'drowsy': {'emoji': '😪', 'color': 'red', 'status': 'error'},
                        'frustrated': {'emoji': '😤', 'color': 'orange', 'status': 'warning'},
                        'Looking away': {'emoji': '👀', 'color': 'violet', 'status': 'info'}
                    }
                    
                    config = state_config.get(attention_state, {'emoji': '🧠', 'color': 'blue', 'status': 'info'})


                    st.subheader("Detection Result")
                    st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{config['emoji']}</div>", 
                               unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: center; color: {config['color']};'>{attention_state.upper()}</h1>", 
                               unsafe_allow_html=True)
                    st.subheader("Confidence Level")
                    st.progress(confidence)
                    st.metric("Confidence Score", f"{confidence:.1%}")
                    if config['status'] == 'success':
                        st.success(f"Student is {attention_state.lower()} - Great focus!")
                    elif config['status'] == 'error':
                        st.error(f"Student appears {attention_state.lower()} - May need attention")
                    elif config['status'] == 'warning':
                        st.warning(f"Student seems {attention_state.lower()} - Consider checking in")
                    else:
                        st.info(f"Student is {attention_state.lower()}")
                
                st.markdown("---")
                st.subheader("Summary Report")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.metric("Detected State", attention_state.title())
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with summary_col2:
                    if attention_state == 'engaged':
                        overall = "POSITIVE"
                        st.success(f"Overall Assessment: {overall}")
                    elif attention_state in ['bored', 'drowsy', 'Looking away']:
                        overall = "NEEDS ATTENTION"
                        st.error(f"Overall Assessment: {overall}")
                    else:
                        overall = "NEUTRAL"
                        st.warning(f"Overall Assessment: {overall}")
                    
                    if confidence >= 0.7:
                        st.info("AI is confident in this prediction")
                    else:
                        st.info("AI has moderate confidence in this prediction")
                
                with st.expander("Detailed Analysis"):
                    st.write("**Model Information:**")
                    st.write(f"- Architecture: ResNet-18")
                    st.write(f"- Input Resolution: 224x224")
                    st.write(f"- Processing Device: {device}")
                    st.write(f"- Classes Detected: {len(classes)}")
                    
                    st.write("**Probability Distribution:**")
                    probs_all = torch.nn.functional.softmax(output[0], dim=0)
                    for i, class_name in enumerate(classes):
                        prob = probs_all[i].item()
                        st.write(f"- {class_name}: {prob:.2%}")
                        st.progress(prob)

    with colx2:
        st.subheader("Detection Tips")
        
        st.write(""" 
        **For better detection:**
        - Ensure good lighting on your face
        - Keep your face clearly visible
        - Maintain proper distance from camera
        - Look directly at the camera for 'engaged'
        - Detection works best with clear facial expressions
        """)
       
        if model is not None:
            st.subheader("Model Info")
            st.write(f"Classes: {len(classes)}")
            st.write("Architecture: ResNet-18")
            st.write("Input size: 224x224")
        
        with st.expander("Troubleshooting"):
            st.write(""" 
            **If detection isn't working:**
            - Make sure your face is clearly visible
            - Check lighting conditions
            - Ensure you're facing the camera
            - Wait a few seconds for processing
            - Try different facial expressions
            
            **If camera won't connect:**
            - Refresh the page and try again
            - Check camera permissions in browser
            - Try a different browser (Chrome recommended)
            - Ensure you're using HTTPS (not HTTP)
            - Wait up to 30 seconds for connection
            
            **Supported browsers:**
            - Chrome/Chromium
            - Firefox
            - Safari (macOS/iOS)
            - Edge (may have issues)
            """)
            

if __name__ == "__main__":
    main()
