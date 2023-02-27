from diffusers import StableDiffusionPipeline
import torch
from sanic import Sanic, response
import subprocess

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
# user_src.init()

# Create the http server app
server = Sanic("my_app")

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

    # output = user_src.inference(model_inputs)

    # --- Move to user_src.inference() ---
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]  
        
    image.save("output.png")
    # ---

    return response.json(model_inputs)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=3000, workers=1)