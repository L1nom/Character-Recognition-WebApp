import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow import keras


mapping = np.loadtxt('emnist-balanced-mapping.txt', dtype=int, usecols=(1), unpack=True)
print(mapping)
char_labels = {}
for i in range(47):
    char_labels[i] = chr(mapping[i])
# print(char_labels)
print(tf.__version__)

# Specify canvas parameters in application
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    display_toolbar=True,
    key="canvas",
)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)

guess = st.button("predict")

model = keras.models.load_model('emnist.model')

if guess:
    # # test = np.array(canvas_result.image_data)
    # image_arr = canvas_result.image_data[:, :]
    # # st.write(image_arr)
    # # st.image(image_arr)
    # st.write(image_arr.size, image_arr.shape)
    # # st.table(image_arr)
    # # st.image(image_arr)
    # # Literal["1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"]
    # im = Image.fromarray(image_arr)
    # im.save("test.png")
    # # st.image(im)
    # # st.image(Image.fromarray(image_arr).convert("L"))
    #
    # im = Image.fromarray(image_arr)
    # st.image(im)
    # # st.image(Image.fromarray(image_arr).convert("L"))
    # st.write(image_arr.size, image_arr.shape)
    # image_arr.resize((28, 28, 1))
    # st.write(image_arr.size, image_arr.shape)
    #
    # image_arr = np.invert(np.array([image_arr]))
    # st.write("1", image_arr.shape)
    # image_arr.reshape(1, 28, 28, 1)
    # image_arr = image_arr / 255
    #
    # st.write("2", image_arr.shape)
    # #
    # # predict = model.predict(image_arr)
    # #
    # # final_predict = char_labels[np.argmax(predict)]
    # # if not final_predict:
    # #     st.stop()
    # # st.write("Finished thinking....it is: " + str(final_predict))
    #
    # # print(canvas_result.image_data)
    #
    # # grab = canvas_result.image_data
    # # st.write(grab.shape)
    # # st.write(grab.size)
    # # print(grab)
    # # grab = np.resize(grab, 1*28*28*1)
    # # # grab = grab.resize((1, 28, 28, 1))
    # # # grab.reshape((28, 28))
    # # st.write(grab.size, grab.shape)
    # # # grab = grab.reshape(1, 28, 28, 1)
    # # st.write("shape", grab.shape)
    #
    # # predict = model.predict(grab)
    # #
    # # final_predict = char_labels[np.argmax(predict)]
    # # if not final_predict:
    # #     st.stop()
    # # st.write("Finished thinking....it is: " + str(final_predict))
    result = (canvas_result.image_data[:,:,:3]).astype(np.uint8)
    im = Image.fromarray(result).convert("L")
    print(result.shape, result.size)
    im = im.resize((28,28))
    # st.image(im)
    # im.save("test1.png")

    # img = Image.open('test.png')
    # img = Image.open("test1.png")
    img = np.array(im)
    # print(np.matrix(img))
    # img = img.resize((28,28))
    # img = np.array(img)
    img = np.invert(np.array([img]))
    img = img.reshape(1, 28, 28, 1)
    img = img / 255
    predict = model.predict(img)  # identify the class which resonates highest with the image
    print(char_labels[np.argmax(predict)])
    st.write(char_labels[np.argmax(predict)])


    # img = result
    # st.write(img.size, img.shape)
    # img.resize()
    # img = np.invert(np.array([img]))
    # img = img.reshape(1, 28, 28, 1)
    # img = img / 255
    # predict = model.predict(img)  # identify the class which resonates highest with the image
    # print(char_labels[np.argmax(predict)])
    # if not predict:
    #     st.stop()
    # st.write("Finished thinking....it is: " + str(predict))