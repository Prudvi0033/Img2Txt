from PIL import Image
from io import BytesIO
import torch
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
import streamlit as st

# Load environment variables
load_dotenv(find_dotenv(".env"))

def img2text(image):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(image)
    return text

def main():
    st.set_page_config(page_title="Image to Text")

    st.header("Turn your image into text")
    up_file = st.file_uploader("Choose an image....", type=['png', 'jpg', 'jpeg'])
  
    if up_file is not None:
        try:
            # Read the uploaded image bytes
            image_bytes = up_file.read()

            # Convert the image bytes to a PIL image object
            pil_image = Image.open(BytesIO(image_bytes))

            # Display the uploaded image
            st.image(pil_image, caption="Uploaded Image..", use_column_width=True)

            text = img2text(pil_image)

            # Display the extracted text
            with st.expander("Extracted Text"):
                st.write(text)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    # Run the Streamlit app
    main()
