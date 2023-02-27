import torch
from sanic import Sanic, response
import subprocess
import traceback

import app # app.py

# Create the http server app
server = Sanic("my_app")

# Init handler is called once when the server starts
app.init()

# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

# Inference POST handler at '/' is called for every http call from Banana
@server.route('/', methods=["POST"]) 
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    try:
        output = app.inference(model_inputs)
    except Exception as err:
        output = {
            "error": {
                "code": "APP_INFERENCE_ERROR",
                "name": type(err).__name__,
                "message": str(err),
                "stack": traceback.format_exc(),
            }
        }
        print(output)

    return response.json(output)

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=3000, workers=torch.cuda.device_count())