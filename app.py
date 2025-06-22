"""
MNIST Handwritten Digit Generator Web App
Built with Streamlit
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Model definition (same as training script)
class ConditionalVAE(nn.Module):
    """Conditional VAE for digit-specific generation"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.num_classes = num_classes
        
    def encode(self, x, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        x_labeled = torch.cat([x, labels_onehot], dim=1)
        h = self.encoder(x_labeled)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        z_labeled = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_labeled)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x.view(-1, 784), labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = ConditionalVAE(latent_dim=20).to(device)
    try:
        # Try to load the model weights
        model.load_state_dict(torch.load('conditional_vae_mnist.pth', map_location=device))
        model.eval()
        return model, True
    except FileNotFoundError:
        st.error("Model file 'conditional_vae_mnist.pth' not found. Please upload the trained model.")
        return model, False

def generate_digits(model, digit, num_samples=5, seed=None):
    """Generate handwritten digits"""
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    with torch.no_grad():
        # Create labels for the desired digit
        labels = torch.tensor([digit] * num_samples).to(device)
        
        # Sample from latent space
        z = torch.randn(num_samples, 20).to(device)
        
        # Generate images
        generated = model.decode(z, labels)
        generated = generated.view(num_samples, 28, 28)
        
        return generated.cpu().numpy()

def create_image_grid(images):
    """Create a grid of images"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

def main():
    # Title and description
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("""
    This app generates handwritten digits (0-9) using a Conditional Variational Autoencoder (CVAE) 
    trained on the MNIST dataset. Select a digit and generate 5 unique samples!
    """)
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.warning("⚠️ Model not loaded. Please ensure 'conditional_vae_mnist.pth' is in the app directory.")
        
        # File uploader for model
        uploaded_file = st.file_uploader("Upload your trained model (.pth)", type=['pth'])
        if uploaded_file is not None:
            # Save uploaded file
            with open("conditional_vae_mnist.pth", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Model uploaded successfully! Please refresh the page.")
            st.experimental_rerun()
        
        # Show demo functionality anyway
        st.info("Demo mode: Showing random noise as placeholder")
        model_loaded = False
    
    # Sidebar controls
    st.sidebar.header("Generation Controls")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate:",
        options=list(range(10)),
        format_func=lambda x: f"Digit {x}"
    )
    
    # Seed for reproducibility
    use_seed = st.sidebar.checkbox("Use seed for reproducible results")
    seed_value = None
    if use_seed:
        seed_value = st.sidebar.number_input("Seed value:", min_value=0, max_value=9999, value=42)
    
    # Generation button
    generate_button = st.sidebar.button("🎲 Generate 5 Samples", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Generated Digit: {selected_digit}")
        
        if generate_button or 'generated_images' not in st.session_state:
            if model_loaded:
                # Generate real images
                with st.spinner("Generating digits..."):
                    generated_images = generate_digits(model, selected_digit, 5, seed_value)
                    st.session_state.generated_images = generated_images
                    st.session_state.current_digit = selected_digit
            else:
                # Generate dummy images for demo
                generated_images = np.random.rand(5, 28, 28) * 0.3
                st.session_state.generated_images = generated_images
                st.session_state.current_digit = selected_digit
        
        # Display images
        if 'generated_images' in st.session_state:
            image_grid = create_image_grid(st.session_state.generated_images)
            st.image(image_grid, use_column_width=True)
            
            # Individual images
            st.subheader("Individual Samples")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    # Create individual image
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(st.session_state.generated_images[i], cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    
                    img = Image.open(buf)
                    st.image(img, caption=f"Sample {i+1}")
    
    with col2:
        st.subheader("Model Information")
        
        st.info("""
        **Model Architecture:**
        - Conditional Variational Autoencoder
        - Input: 784 dimensions (28×28 pixels)
        - Latent space: 20 dimensions
        - Hidden layers: 400 units each
        - Trained on MNIST dataset
        """)
        
        if model_loaded:
            st.success("✅ Model loaded successfully")
            st.metric("Device", str(device).upper())
            
            # Model parameters
            total_params = sum(p.numel() for p in model.parameters())
            st.metric("Total Parameters", f"{total_params:,}")
        else:
            st.error("❌ Model not loaded")
        
        st.subheader("How it works")
        st.markdown("""
        1. **Select** a digit (0-9)
        2. **Click** Generate to create 5 samples
        3. The model samples from a learned latent space
        4. Each generation is unique but represents the selected digit
        """)
        
        # Statistics
        if 'generated_images' in st.session_state:
            images = st.session_state.generated_images
            st.subheader("Image Statistics")
            st.metric("Image Size", "28×28 pixels")
            st.metric("Pixel Range", f"{images.min():.3f} - {images.max():.3f}")
            st.metric("Mean Intensity", f"{images.mean():.3f}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit • Powered by PyTorch • Trained on MNIST Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
