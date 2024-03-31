# app.py
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import os, base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

app = Flask(__name__)

# Load the YOLOv8 ONNX model
model_path = "best.onnx"
model = YOLO(model_path, task='detect')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        if file:
            image = Image.open(BytesIO(file.read()))
            result = model(image)

            for r in result:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                result_image = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # annotated_image.save('results.jpg')  # save image
                # Save the annotated image to a temporary buffer
                buffer = BytesIO()
                # annotated_image.save(buffer, format='PNG')
                annotated_image = plt.imshow(result_image)
                plt.axis('off')
                # buffer.seek(0)
            annotated_image = plot()
            return render_template('index.html', image=annotated_image)

    return render_template('index.html')

def plot():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.flush()
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.flush()
    buffer.close()
    return graph

if __name__ == '__main__':
    app.run(debug=True)
