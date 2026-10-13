import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import numpy as np

from model import Net  # sənin model faylın

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yüklə
model = Net().to(device)
model.load_state_dict(torch.load("cnn_mnist.pth", map_location=device))
model.eval()

st.title("Handwritten Digit Recognizer")

st.write("Draw digit (0-9)")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # numpy → PIL
        img = Image.fromarray(img.astype("uint8"))
        img = img.convert("L")          # grayscale
        img = ImageOps.invert(img)      # ağ fon, qara rəqəm üçün

        img = transform(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()

        st.success(f"Modelin tapdığı rəqəm: {pred}")


# streamlit run app.py