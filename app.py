import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from PIL import Image
import cv2
import io
import zipfile
import av
from torch.utils.data import DataLoader
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import numpy as np

# Page config must be the first Streamlit command
st.set_page_config(page_title="📸 Real-time Student Attention Detection", layout="wide", page_icon="📸")

# Classes for the model - make sure these match your model training
classes = ['bored', 'confused', 'drowsy', 'engaged', 'frustrated', 'Looking away']

# Device
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

# Main function
def main():
    st.title("📸 Real-time Student Attention Detection")
    st.markdown("---")
    
    # Sidebar for selecting input method
    input_method = st.sidebar.radio("Select Input Method", ("Webcam", "Upload File"))
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.write("This app detects student attention states in real-time using your webcam.")
        
        st.subheader("Detected States:")
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
        st.write(f"🔧 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"🤖 Model: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
        st.write(f"👤 Face Detection: {'✅ Ready' if face_cascade is not None else '❌ Failed'}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed" if input_method == "Webcam" else "📂 Upload Image")
        
        if input_method == "Webcam":
            # Webcam option
            if model is None:
                st.error("⚠️ Model not loaded! Please ensure 'attention_resnet18.pth' is in the app directory.")
            elif face_cascade is None:
                st.error("⚠️ Face detection not available!")
            else:
                st.success("✅ System ready for attention detection!")
            
            # WebRTC streamer with STUN/TURN configuration for deployment
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
        else:
            # Upload option
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Process uploaded image here
                image = Image.open(uploaded_file)
                
                # Convert the PIL image to numpy array before passing to the transform function
                image_np = np.array(image)

                # Preprocess the image (same transformation as for webcam)
                input_image = webcam_transform(image_np).unsqueeze(0).to(device)
                
                # Make prediction on the uploaded image
                with torch.no_grad():
                    output = model(input_image)
                    _, predicted = torch.max(output, 1)
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    pred_idx = predicted.item()
                    confidence = probs[pred_idx].item()
                
                # Get the predicted attention state
                attention_state = classes[pred_idx]
                
                # Create spacing
                st.markdown("---")
                
                # Main layout with columns
                img_col, result_col = st.columns([1.2, 1])
                
                with img_col:
                    st.subheader("📸 Analyzed Image")
                    st.image(image, use_container_width=True)
                
                with result_col:
                    # Attention state configuration
                    state_config = {
                        'engaged': {'emoji': '🎯', 'color': 'green', 'status': 'success'},
                        'bored': {'emoji': '😴', 'color': 'red', 'status': 'error'},
                        'confused': {'emoji': '🤔', 'color': 'orange', 'status': 'warning'},
                        'drowsy': {'emoji': '😪', 'color': 'red', 'status': 'error'},
                        'frustrated': {'emoji': '😤', 'color': 'orange', 'status': 'warning'},
                        'Looking away': {'emoji': '👀', 'color': 'violet', 'status': 'info'}
                    }
                    
                    config = state_config.get(attention_state, {'emoji': '🧠', 'color': 'blue', 'status': 'info'})
                    
                    # Main result display
                    st.subheader("🎯 Detection Result")
                    
                    # Large emoji and state
                    st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{config['emoji']}</div>", 
                               unsafe_allow_html=True)
                    
                    # State name in large text
                    st.markdown(f"<h1 style='text-align: center; color: {config['color']};'>{attention_state.upper()}</h1>", 
                               unsafe_allow_html=True)
                    
                    # Confidence as a progress bar
                    st.subheader("📊 Confidence Level")
                    st.progress(confidence)
                    st.metric("Confidence Score", f"{confidence:.1%}")
                    
                    # Status message using Streamlit's alert components
                    if config['status'] == 'success':
                        st.success(f"✅ Student is {attention_state.lower()} - Great focus!")
                    elif config['status'] == 'error':
                        st.error(f"⚠️ Student appears {attention_state.lower()} - May need attention")
                    elif config['status'] == 'warning':
                        st.warning(f"⚡ Student seems {attention_state.lower()} - Consider checking in")
                    else:
                        st.info(f"ℹ️ Student is {attention_state.lower()}")
                
                # Additional analysis section
                st.markdown("---")
                
                # Three columns for detailed breakdown
                conf_col, state_col, advice_col = st.columns(3)
                
                with conf_col:
                    st.subheader("🎯 Accuracy")
                    if confidence >= 0.8:
                        st.success("Very High")
                        st.write("🟢 Highly reliable")
                    elif confidence >= 0.6:
                        st.success("High") 
                        st.write("🟡 Reliable")
                    elif confidence >= 0.4:
                        st.warning("Moderate")
                        st.write("🟠 Somewhat reliable")
                    else:
                        st.error("Low")
                        st.write("🔴 Less reliable")
                
                with state_col:
                    st.subheader("📋 Analysis")
                    
                    # Description based on attention state
                    descriptions = {
                        'engaged': "Student is actively focused and participating",
                        'bored': "Student may be losing interest in the material",
                        'confused': "Student might need clarification or help",
                        'drowsy': "Student appears tired and may need a break",
                        'frustrated': "Student may be struggling with the content",
                        'Looking away': "Student's attention is directed elsewhere"
                    }
                    
                    st.write(descriptions.get(attention_state, "Attention state detected"))
                
                with advice_col:
                    st.subheader("💡 Suggestion")
                    
                    # Recommendations based on state
                    recommendations = {
                        'engaged': "Continue with current approach",
                        'bored': "Try interactive activities or change pace",
                        'confused': "Pause and provide additional explanation",
                        'drowsy': "Consider a short break or energizing activity",
                        'frustrated': "Offer individual support or simplify concept",
                        'Looking away': "Redirect attention with engaging content"
                    }
                    
                    st.write(recommendations.get(attention_state, "Monitor student engagement"))
                
                # Summary card
                st.markdown("---")
                st.subheader("📈 Summary Report")
                
                # Use columns for a clean summary layout
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.metric("Detected State", attention_state.title())
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with summary_col2:
                    # Overall assessment
                    if attention_state == 'engaged':
                        overall = "POSITIVE"
                        st.success(f"Overall Assessment: {overall}")
                    elif attention_state in ['bored', 'drowsy', 'Looking away']:
                        overall = "NEEDS ATTENTION"
                        st.error(f"Overall Assessment: {overall}")
                    else:
                        overall = "NEUTRAL"
                        st.warning(f"Overall Assessment: {overall}")
                    
                    # AI confidence level
                    if confidence >= 0.7:
                        st.info("🤖 AI is confident in this prediction")
                    else:
                        st.info("🤖 AI has moderate confidence in this prediction")
                
                # Expandable detailed analysis
                with st.expander("🔍 Detailed Analysis"):
                    st.write("**Model Information:**")
                    st.write(f"- Architecture: ResNet-18")
                    st.write(f"- Input Resolution: 224x224")
                    st.write(f"- Processing Device: {device}")
                    st.write(f"- Classes Detected: {len(classes)}")
                    
                    st.write("**Probability Distribution:**")
                    # Show all class probabilities
                    probs_all = torch.nn.functional.softmax(output[0], dim=0)
                    for i, class_name in enumerate(classes):
                        prob = probs_all[i].item()
                        st.write(f"- {class_name}: {prob:.2%}")
                        st.progress(prob)
    with col2:
        st.subheader("📊 Detection Tips")
        
        st.write(""" 
        **For better detection:**
        - 💡 Ensure good lighting on your face
        - 👤 Keep your face clearly visible
        - 📏 Maintain proper distance from camera
        - 🎯 Look directly at the camera for 'engaged'
        - 😴 Detection works best with clear facial expressions
        """)
        
        # Instructions
        st.subheader("📋 Instructions")
        st.write("""
        1. Click **START** to begin camera feed
        2. Allow camera permissions when prompted
        3. Wait for connection (may take 10-30 seconds)
        4. Position yourself in the camera view
        5. Watch real-time attention detection
        6. Click **STOP** to end session
        """)
        
        # Model info
        if model is not None:
            st.subheader("🧠 Model Info")
            st.write(f"Classes: {len(classes)}")
            st.write("Architecture: ResNet-18")
            st.write("Input size: 224x224")
        
        # Troubleshooting section
        with st.expander("🔧 Troubleshooting"):
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
            - ✅ Chrome/Chromium
            - ✅ Firefox
            - ✅ Safari (macOS/iOS)
            - ⚠️ Edge (may have issues)
            """)

if __name__ == "__main__":
    main()
