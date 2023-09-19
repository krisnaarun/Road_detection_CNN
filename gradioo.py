
import gradio as gr
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model


# Load your trained model
model_path = r"C:\Users\Dell\Desktop\pjct_cnn\trained_model.h5"
loaded_model = load_model(model_path)

# Define a function to make predictions
def classify_road_condition(image_pil):
    # Resize image to match model input size
    img = image_pil.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize image data

    prediction = loaded_model.predict(img)
    road_conditions = ["Clean Road", "Dirty Road", "Potholes"]
    result = road_conditions[np.argmax(prediction)]

    # Create a descriptive message
    if result == "Clean Road":
        message = "The road is clean and well-maintained."
    elif result == "Dirty Road":
        message = "The road is dirty and may need cleaning."
    elif result == "Potholes":
        message = "The road contains potholes and needs repair."

    return f"Result: {result}\nMessage: {message}"

# Create a Gradio interface
iface = gr.Interface(fn=classify_road_condition,
                     inputs=gr.inputs.Image(type="pil", label="Select an image"),
                     outputs="text")

# Launch the Gradio app
iface.launch(share=True)