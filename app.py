import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# -----------------------------
# Load your trained model
# ----------------------------
# Class names
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
from function import CNN  # or define CNN above
model = CNN(num_classes=4)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# Image transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Adjust size to your model input
    transforms.ToTensor(),
])
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_tumor_type(image):
    image = image.convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


# -----------------------------
# Streamlit Web Interface
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI scan image and let the model predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            prediction = predict_tumor_type(image)
        st.success(f"🩺 **Predicted Tumor Type:** {prediction}")