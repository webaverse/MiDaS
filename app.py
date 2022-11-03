import cv2
import torch
# import gradio as gr
import numpy as np
from PIL import Image
import flask
from flask import request
import io

torch.hub.download_url_to_file('https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f', 'turtle.jpg')
torch.hub.download_url_to_file('https://images.unsplash.com/photo-1519066629447-267fffa62d4b', 'lions.jpg')

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

use_large_model = True

if use_large_model:
  midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
else:
  midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

device = "cpu"
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
  transform = midas_transforms.default_transform
else:
  transform = midas_transforms.small_transform

def depth(img):
  cv_image = np.array(img) 
  img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

  input_batch = transform(img).to(device)
  with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

  output = prediction.cpu().numpy()
  formatted = (output * 255 / np.max(output)).astype('uint8')
  img = Image.fromarray(formatted)
  return img

# start a flask app
app = flask.Flask(__name__)


# serve the depth endpoint under /depth
@app.route("/depth", methods=["GET", "POST", "OPTIONS"])
def depth2():
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        return response

    # catch errors and respond with 500
    try:
        # read the post body
        body = request.get_data()
        img = Image.open(io.BytesIO(body))
        cv_image = np.array(img)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        output = prediction.cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype('uint8')
        img = Image.fromarray(formatted)
        # respond with the image data to the web page
        # use cors headers
        response = flask.make_response(img.tobytes())
        response.headers.set('Access-Control-Allow-Origin', '*')
        response.headers.set('Access-Control-Allow-Methods', '*')
        response.headers.set('Access-Control-Allow-Headers', '*')
        # response.headers.set('Cross-Origin-Opener-Policy', 'same-origin')
        # response.headers.set('Cross-Origin-Embedder-Policy', 'require-corp')
        # response.headers.set('Cross-Origin-Resource-Policy', 'cross-origin')
        return response
        # return flask.send_file(img, mimetype='image/png')
    except Exception as e:
        print(e)
        # respond to the client with the error
        response = flask.make_response(str(e))
        response.headers.set('Access-Control-Allow-Origin', '*')
        response.headers.set('Access-Control-Allow-Methods', '*')
        response.headers.set('Access-Control-Allow-Headers', '*')
        return response
        # return flask.Response(status=500)

# listen on 0.0.0.0:80
if __name__ == "__main__":
    print("Starting server...")
    app.run(
        host="0.0.0.0",
        port=80,
        debug=False,
        # dev_tools_silence_routes_logging = False,
        # dev_tools_ui=True,
        # dev_tools_hot_reload=True,
        threaded=True,
    )

# inputs =  gr.inputs.Image(type='pil', label="Original Image")
# outputs = gr.outputs.Image(type="pil",label="Output Image")

# title = "MiDaS"
# description = "Gradio demo for MiDaS v2.1 which takes in a single image for computing relative depth. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
# article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1907.01341v3'>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer</a> | <a href='https://github.com/intel-isl/MiDaS'>Github Repo</a></p>"

# examples = [
#     ["turtle.jpg"],
#     ["lions.jpg"]
# ]

# app = gr.Interface(depth, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch(enable_queue=True)
