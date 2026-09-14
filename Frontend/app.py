import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import time

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="X-MAL | Malware Classifier",
    page_icon="🦠",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #FF4B4B; color: white; font-weight: bold; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #FF6B6B; border-color: #FF6B6B; }
    .prediction-box { padding: 20px; border-radius: 10px; background-color: #1E2127; border-left: 5px solid #FF4B4B; margin-top: 20px; }
    .title-text { font-family: 'Courier New', Courier, monospace; color: #FF4B4B; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. MODEL DEFINITION & LOADING
# ==========================================
MALWARE_CLASSES = ['adware', 'backdoor', 'downloader', 'spyware', 
                'trojan', 'virus', 'worm', 'benign']

@st.cache_resource
def load_model():
    try:
        # 1. Load the EfficientNet-B3 blueprint (weights=None saves download time in production)
        model = models.efficientnet_b3(weights=None)
        
        # 2. Modify the final classification layer to output 5 classes instead of ImageNet's 1000
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(MALWARE_CLASSES))
        
        # 3. Load your trained weights from the local folder
        # Make sure your weights file is inside a folder named 'models' or update the path below
        model.load_state_dict(torch.load('models/malware_model.pth', map_location=torch.device('cpu')))
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================
# EfficientNet expects RGB images and standard ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)), # EfficientNet-B3 standard resolution (change to 224 if you trained on 224)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image, model):
    """Preprocesses the image and returns the predicted class."""
    # Ensure image is RGB (removes alpha channels from PNGs or converts grayscale)
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return MALWARE_CLASSES[predicted.item()]


# ==========================================
# 4. USER INTERFACE
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100) 
    st.markdown("<h2 class='title-text'>X-MAL Engine</h2>", unsafe_allow_html=True)
    st.info("Upload a binary-to-image representation of a file to classify its malware family using EfficientNet-B3.")
    st.divider()
    st.markdown("**Supported Classes:**")
    for cls in MALWARE_CLASSES:
        st.markdown(f"- `{cls}`")

st.markdown("<h1 style='text-align: center;'><span class='title-text'>X-MAL</span> Vision Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Threat detection via structural image analysis.</p>", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("Upload Malware Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Target File")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col2:
        st.subheader("Analysis")
        analyze_button = st.button("Initialize Scan")
        
        if analyze_button:
            if model is None:
                st.error("Model failed to load. Check your models/malware_model.pth file path.")
            else:
                with st.spinner("Analyzing neural patterns..."):
                    time.sleep(0.5) # Slight UI delay for effect
                    prediction = predict_image(image, model)
                
                st.success("Scan Complete.")
                st.markdown(f"""
                    <div class="prediction-box">
                        <h4 style="margin:0; color:#888;">Detected Signature:</h4>
                        <h1 style="margin:0; color:#FF4B4B;">{prediction}</h1>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.info("Awaiting file upload...")