import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from torchvision.utils import make_grid
import os

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .digit-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1em 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75em 2em;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .generated-image {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# Generator Model Class (same as training script)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_channels=1, img_size=28):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.img_channels = img_channels
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Generator layers
        self.model = nn.Sequential(
            # Input: (latent_dim + num_classes) -> 256
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 -> 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 1024 -> 784 (28*28)
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate noise and label embeddings
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_generator_model():
    """Load the trained generator model"""
    try:
        device = torch.device('cpu')  # Use CPU for deployment
        generator = Generator()
        
        # Try to load the model weights
        if os.path.exists('generator.pth'):
            generator.load_state_dict(torch.load('generator.pth', map_location=device))
            generator.eval()
            return generator, True
        else:
            # If no trained model, create a dummy one for demonstration
            st.warning("⚠️ Pre-trained model not found. Using demonstration mode.")
            return generator, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return Generator(), False

def generate_digit_images(digit, num_images=5, use_trained_model=True):
    """Generate images for a specific digit"""
    device = torch.device('cpu')
    
    if use_trained_model and st.session_state.generator is not None:
        # Use the trained model
        generator = st.session_state.generator
        generator.eval()
        
        with torch.no_grad():
            # Create labels for the specific digit
            labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
            
            # Generate random noise
            noise = torch.randn(num_images, 100, device=device)
            
            # Generate images
            gen_imgs = generator(noise, labels)
            
            # Denormalize from [-1, 1] to [0, 1]
            gen_imgs = (gen_imgs + 1) / 2
            
            return gen_imgs.numpy()
    else:
        # Use pattern-based generation for demonstration
        return generate_demo_digits(digit, num_images)

def generate_demo_digits(digit, num_images=5):
    """Generate demonstration digit patterns"""
    images = []
    
    for i in range(num_images):
        # Create a 28x28 image
        img = np.zeros((1, 28, 28))
        
        # Add some variation
        offset_x = np.random.randint(-2, 3)
        offset_y = np.random.randint(-2, 3)
        
        # Create simple patterns for each digit
        pattern = create_digit_pattern(digit, offset_x, offset_y)
        
        # Add the pattern to the image
        img[0] = pattern
        
        # Add some noise
        noise = np.random.normal(0, 0.1, (28, 28))
        img[0] = np.clip(img[0] + noise, 0, 1)
        
        images.append(img)
    
    return np.array(images)

def create_digit_pattern(digit, offset_x=0, offset_y=0):
    """Create a simple pattern for each digit"""
    img = np.zeros((28, 28))
    center_x, center_y = 14 + offset_x, 14 + offset_y
    
    # Simple patterns for each digit
    if digit == 0:
        for angle in np.linspace(0, 2*np.pi, 50):
            x = int(center_x + 8 * np.cos(angle))
            y = int(center_y + 10 * np.sin(angle))
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 1.0
    
    elif digit == 1:
        for y in range(max(0, center_y-10), min(28, center_y+10)):
            if 0 <= center_x < 28:
                img[y, center_x] = 1.0
    
    elif digit == 2:
        # Top curve
        for x in range(max(0, center_x-8), min(28, center_x+8)):
            y = int(center_y - 8 + 3 * np.sin(np.pi * (x - center_x + 8) / 16))
            if 0 <= y < 28:
                img[y, x] = 1.0
        # Bottom line
        for x in range(max(0, center_x-8), min(28, center_x+8)):
            if 0 <= center_y+8 < 28:
                img[center_y+8, x] = 1.0
    
    # Add patterns for other digits (3-9)
    elif digit == 3:
        for y in range(max(0, center_y-8), min(28, center_y+8)):
            x1 = int(center_x + 6 * np.cos(np.pi * (y - center_y) / 16))
            if 0 <= x1 < 28:
                img[y, x1] = 1.0
    
    elif digit == 4:
        # Vertical line
        for y in range(max(0, center_y-10), min(28, center_y+10)):
            if 0 <= center_x+5 < 28:
                img[y, center_x+5] = 1.0
        # Horizontal line
        for x in range(max(0, center_x-8), min(28, center_x+8)):
            if 0 <= center_y < 28:
                img[center_y, x] = 1.0
    
    elif digit == 5:
        # Top and middle lines
        for x in range(max(0, center_x-6), min(28, center_x+6)):
            if 0 <= center_y-6 < 28:
                img[center_y-6, x] = 1.0
            if 0 <= center_y < 28:
                img[center_y, x] = 1.0
    
    elif digit == 6:
        # Circle pattern
        for angle in np.linspace(0, 2*np.pi, 40):
            x = int(center_x + 6 * np.cos(angle))
            y = int(center_y + 8 * np.sin(angle))
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 1.0
    
    elif digit == 7:
        # Top line and diagonal
        for x in range(max(0, center_x-8), min(28, center_x+8)):
            if 0 <= center_y-8 < 28:
                img[center_y-8, x] = 1.0
        for i in range(16):
            x = int(center_x + 8 - i)
            y = int(center_y - 8 + i)
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 1.0
    
    elif digit == 8:
        # Two circles
        for angle in np.linspace(0, 2*np.pi, 30):
            x1 = int(center_x + 5 * np.cos(angle))
            y1 = int(center_y - 4 + 4 * np.sin(angle))
            x2 = int(center_x + 5 * np.cos(angle))
            y2 = int(center_y + 4 + 4 * np.sin(angle))
            if 0 <= x1 < 28 and 0 <= y1 < 28:
                img[y1, x1] = 1.0
            if 0 <= x2 < 28 and 0 <= y2 < 28:
                img[y2, x2] = 1.0
    
    elif digit == 9:
        # Circle with stem
        for angle in np.linspace(0, 2*np.pi, 30):
            x = int(center_x + 5 * np.cos(angle))
            y = int(center_y - 4 + 4 * np.sin(angle))
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 1.0
    
    return img

def plot_images(images, digit):
    """Plot generated images in a grid"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f'Generated Images for Digit {digit}', fontsize=16, fontweight='bold')
    
    for i in range(5):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate handwritten digits using a trained GAN model</p>', unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            st.session_state.generator, st.session_state.model_loaded = load_generator_model()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    st.sidebar.markdown("---")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate:",
        options=list(range(10)),
        index=0,
        help="Choose which digit (0-9) you want to generate"
    )
    
    # Number of images
    num_images = st.sidebar.slider(
        "Number of images:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many variations to generate"
    )
    
    # Generation button
    generate_button = st.sidebar.button(
        "🎲 Generate Images",
        help="Click to generate new images"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    if st.session_state.model_loaded and st.session_state.generator is not None:
        st.sidebar.success("✅ Trained model loaded")
        st.sidebar.info("Using GAN model trained on MNIST dataset")
    else:
        st.sidebar.warning("⚠️ Demo mode active")
        st.sidebar.info("Upload trained model weights for full functionality")
    
    # File upload for model weights
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Upload Model")
    uploaded_file = st.sidebar.file_uploader(
        "Upload generator.pth file:",
        type=['pth'],
        help="Upload your trained generator model"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file
            with open("generator.pth", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Reload model
            st.session_state.generator, st.session_state.model_loaded = load_generator_model()
            st.sidebar.success("Model uploaded successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error uploading model: {str(e)}")
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Display current selection
        st.markdown(f"### Selected Digit: **{selected_digit}**")
        
        # Generate images
        if generate_button or 'generated_images' not in st.session_state:
            with st.spinner(f"Generating {num_images} images of digit {selected_digit}..."):
                # Generate images
                use_trained = st.session_state.model_loaded and st.session_state.generator is not None
                generated_images = generate_digit_images(selected_digit, num_images, use_trained)
                
                # Store in session state
                st.session_state.generated_images = generated_images
                st.session_state.current_digit = selected_digit
        
        # Display results
        if 'generated_images' in st.session_state:
            st.markdown("### 🖼️ Generated Images")
            
            # Plot images
            fig = plot_images(st.session_state.generated_images, st.session_state.current_digit)
            st.pyplot(fig)
            
            # Individual images in columns
            st.markdown("### 🔍 Individual Samples")
            cols = st.columns(min(5, num_images))
            
            for i in range(min(len(st.session_state.generated_images), num_images)):
                with cols[i % 5]:
                    st.image(
                        st.session_state.generated_images[i][0],
                        caption=f"Sample {i+1}",
                        use_column_width=True,
                        clamp=True
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📝 About This App
    
    This application uses a **Conditional Generative Adversarial Network (cGAN)** trained on the MNIST dataset to generate handwritten digits. 
    
    **Features:**
    - Generate 1-10 variations of any digit (0-9)
    - Upload your own trained model weights
    - View individual samples and grid layouts
    - Responsive design for all devices
    
    **Model Architecture:**
    - Generator: Fully connected layers with batch normalization
    - Training: 50 epochs on MNIST dataset
    - Input: Random noise + digit label
    - Output: 28×28 grayscale images
    """)

if __name__ == "__main__":
    main()