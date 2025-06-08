import streamlit as st
import sys
import os
import asyncio
import threading
from typing import Optional, Tuple, List, Dict, Any

# Fix for Windows asyncio event loop policy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Fix for PyTorch/Streamlit compatibility issue
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Monkey patch to fix torch.classes issue with Streamlit
try:
    import torch
    if hasattr(torch, '_classes'):
        # Create a mock __path__ attribute if it doesn't exist
        if not hasattr(torch._classes, '__path__'):
            class MockPath:
                def __init__(self):
                    self._path = []
                
                @property
                def _path(self):
                    return []
            
            torch._classes.__path__ = MockPath()
except Exception as e:
    print(f"Warning: Could not apply torch.classes fix: {e}")

# Now import other dependencies
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import time

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="📸 Student Attention Detection", 
    layout="wide", 
    page_icon="📸",
    initial_sidebar_state="expanded"
)

# Constants
CLASSES = ['bored', 'confused', 'drowsy', 'engaged', 'frustrated', 'Looking away']
MODEL_PATH = "attention_resnet18.pth"

# Device setup
@st.cache_data
def get_device():
    """Get the computing device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Load model with enhanced error handling
@st.cache_resource
def load_attention_model():
    """Load the attention detection model with comprehensive error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file '{MODEL_PATH}' not found in current directory.")
            st.info(f"📁 Current directory: {os.getcwd()}")
            st.info("📋 Please ensure the model file is in the same directory as this script.")
            return None
        
        # Create model architecture
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        
        # Load weights with proper error handling
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        except Exception as load_error:
            # Try loading without weights_only for older PyTorch versions
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully on {device}")
        return model
        
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_PATH}' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Try checking your PyTorch installation and model file integrity.")
        return None

# Load face cascade
@st.cache_resource
def load_face_detector():
    """Load Haar Cascade for face detection"""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            st.error("❌ Failed to load face cascade classifier")
            return None
            
        st.success("✅ Face detector loaded successfully")
        return face_cascade
        
    except Exception as e:
        st.error(f"❌ Error loading face detector: {str(e)}")
        return None

# Initialize components
model = load_attention_model()
face_cascade = load_face_detector()

# Image preprocessing pipeline
@st.cache_data
def get_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

transform = get_transform()

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
        
        return CLASSES[predicted_class_idx], confidence
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
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
    
    def start(self, camera_index: int = 0) -> bool:
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_active = True
            return True
            
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
        st.subheader("🎯 Attention Analysis Results")
        
        for result in predictions:
            face_id = result["face_id"]
            state = result["attention_state"]
            confidence = result["confidence"]
            
            # Create columns for each face analysis
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.metric(f"Face {face_id}", state.title())
            
            with col2:
                # Progress bar for confidence
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            
            with col3:
                # Status indicator
                if state == 'engaged':
                    st.success("✅ Focused")
                elif state in ['bored', 'drowsy', 'Looking away']:
                    st.error("⚠️ Attention needed")
                else:
                    st.warning("⚡ Check in")
    
    # Display info messages
    for msg in info_messages:
        st.info(msg["message"])
    
    # Display errors
    for error in errors:
        st.error(f"Face {error['face_id']}: {error['error']}")

def main():
    """Main application function"""
    
    # Title and header
    st.title("📸 Student Attention Detection System")
    st.markdown("### *Real-time attention state analysis using computer vision*")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # System status
        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
            st.metric("Model", "✅" if model else "❌")
        with col2:
            st.metric("Face Detection", "✅" if face_cascade else "❌")
            st.metric("Classes", len(CLASSES))
        
        # Detection states info
        st.subheader("🎯 Detection States")
        state_emojis = {
            'engaged': '🟢 Focused',
            'bored': '🔴 Disengaged', 
            'confused': '🟡 Needs help',
            'drowsy': '🔴 Tired',
            'frustrated': '🟡 Struggling',
            'Looking away': '🔴 Distracted'
        }
        
        for state in CLASSES:
            st.write(f"{state_emojis.get(state, '⚪ ' + state)}")
        
        st.markdown("---")
        st.subheader("💡 Tips for Better Detection")
        st.write("""
        • Ensure good lighting on face
        • Keep face clearly visible  
        • Maintain 2-4 feet from camera
        • Avoid excessive head movement
        • Look towards camera for 'engaged'
        """)
    
    # Main interface
    tab1, tab2 = st.tabs(["📹 Live Camera", "📁 Upload Image"])
    
    with tab1:
        st.subheader("📹 Camera-based Analysis")
        
        # Check system readiness
        if model is None or face_cascade is None:
            st.error("❌ System not ready. Please check model and face detection status in sidebar.")
            return
        
        # Initialize camera manager in session state
        if 'camera_manager' not in st.session_state:
            st.session_state.camera_manager = CameraManager()
        
        # Camera controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎥 Start Camera", type="primary", use_container_width=True):
                if st.session_state.camera_manager.start():
                    st.success("✅ Camera started!")
                    st.rerun()
                else:
                    st.error("❌ Failed to start camera")
        
        with col2:
            if st.button("📸 Capture & Analyze", use_container_width=True):
                frame = st.session_state.camera_manager.capture_frame()
                if frame is not None:
                    st.session_state.captured_frame = frame
                    st.session_state.frame_timestamp = time.time()
                    st.success("✅ Frame captured!")
                    st.rerun()
                else:
                    st.error("❌ No frame available")
        
        with col3:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                st.session_state.camera_manager.stop()
                st.success("✅ Camera stopped!")
                if 'captured_frame' in st.session_state:
                    del st.session_state.captured_frame
                st.rerun()
        
        with col4:
            camera_index = st.selectbox("📷 Camera", [0, 1, 2], index=0)
        
        # Display captured frame and analysis
        if 'captured_frame' in st.session_state:
            st.markdown("---")
            
            # Process the frame
            with st.spinner("🔍 Analyzing attention states..."):
                processed_frame, analysis_results = detect_and_analyze_faces(
                    st.session_state.captured_frame
                )
            
            # Display results
            col_img, col_results = st.columns([3, 2])
            
            with col_img:
                st.subheader("📸 Analysis Result")
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(display_frame, caption=f"Captured at {time.ctime(st.session_state.frame_timestamp)}", 
                        use_container_width=True)
            
            with col_results:
                display_analysis_results(analysis_results)
        
        else:
            st.info("📋 Click 'Start Camera' then 'Capture & Analyze' to begin analysis")
    
    with tab2:
        st.subheader("📁 Image Upload Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image containing faces for attention analysis"
        )
        
        if uploaded_file is not None:
            # Load and process image
            try:
                image = Image.open(uploaded_file)
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                with st.spinner("🔍 Analyzing uploaded image..."):
                    processed_frame, analysis_results = detect_and_analyze_faces(image_np)
                
                # Display results
                col_orig, col_processed = st.columns(2)
                
                with col_orig:
                    st.subheader("📤 Original Image")
                    st.image(image, use_container_width=True)
                
                with col_processed:
                    st.subheader("🎯 Analysis Result")
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(display_frame, use_container_width=True)
                
                # Analysis results
                st.markdown("---")
                display_analysis_results(analysis_results)
                
            except Exception as e:
                st.error(f"❌ Error processing uploaded image: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "💡 Powered by PyTorch, OpenCV, and Streamlit | "
        "🔬 Computer Vision for Education Technology"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()