from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow import keras

st.set_page_config(page_title="Streamlit-HWR", layout="centered")
mapping = np.loadtxt('emnist-balanced-mapping.txt', dtype=int, usecols=(1), unpack=True)
char_labels = {}
for i in range(47):
    char_labels[i] = chr(mapping[i])

model = keras.models.load_model('emnist.model')

with st.container():
    st.subheader("Draw a number! Letter! I'll try and correctly guess it!")
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=True,
        key="canvas",
    )

    # Load our nn model

    # Do something interesting with the image data and paths
    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)
    # if canvas_result.json_data is not None:
    #     objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    #     for col in objects.select_dtypes(include=['object']).columns:
    #         objects[col] = objects[col].astype("str")
    #     st.dataframe(objects)

    guess = st.button("predict")

    if guess:
        # Image reshaping/resizing
        result = (canvas_result.image_data[:, :, :3]).astype(np.uint8)
        im = Image.fromarray(result).convert("L")
        print(result.shape, result.size)
        im = im.resize((28, 28))
        img = np.array(im)
        img = np.invert(np.array([img]))
        img = img.reshape(1, 28, 28, 1)
        img = img / 255
        predict = model.predict(img)  # Identify the class which resonates highest with the image
        st.write("Finished thinking....it is: " + char_labels[np.argmax(predict)])

st.subheader("How it works:")
st.write("Handwriting Recognition is the ability of a computer to interpret handwritten input correctly. For this "
         "application, we create a simple machine learning model using neural networks, which is able to identify an "
         "image and output the character the computer thinks the image closely resembles.")
st.write("To start off, a widely popular containing images of handwritten characters called MNIST was used as "
         "preparation data. The specific data used for this model was a superset including digits and letters. Since "
         "some letters look similar in upper and lowercase, the lowercase duplicate was omitted this dataset The "
         "data was trained on a convolutional neural network, as this type of network is most "
         "commonly applied to analyze visual imagery. After training over the 112800 different images, the model has "
         "achieved an accuracy of 87%.")
