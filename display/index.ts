const express = require("express");
const request = require("request");
const bodyParser = require("body-parser");
const fs = require("fs");

const app = express();
const port = 3000;

const API_URL = "https://o8bwd0d4xqff9p-3000.proxy.runpod.net";

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/public/index.html");
});

app.post("/download", (apiReq, apiRes) => {
  // prompt, negative_prompt, width, height, steps, count, seed, model_id, pipeline, controlnet_type, init_image, guidance_scale, safe_mode
  const prompt = apiReq.body.prompt;
  const negative_prompt = apiReq.body.negative_prompt;
  const width = apiReq.body.width;
  const height = apiReq.body.height;
  const steps = apiReq.body.steps;
  const count = apiReq.body.count;
  const seed = apiReq.body.seed;
  const model_id = apiReq.body.model_id;
  const pipeline = apiReq.body.pipeline;
  const controlnet_type = apiReq.body.controlnet_type;
  const scheduler = apiReq.body.scheduler;
  const init_image = apiReq.body.init_image;
  const guidance_scale = apiReq.body.guidance_scale;
  const strength = apiReq.body.strength;
  const safe_mode = apiReq.body.safe_mode;

  const options = {
    method: "POST",
    url: API_URL,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: prompt,
      negative_prompt: negative_prompt,
      width: parseInt(width),
      height: parseInt(height),
      steps: parseInt(steps),
      count: parseInt(count),
      seed: parseInt(seed),
      model_id: model_id,
      pipeline: pipeline,
      controlnet_type: controlnet_type,
      scheduler: scheduler,
      init_image: init_image,
      guidance_scale: parseFloat(guidance_scale),
      strength: parseFloat(strength),
      safe_mode: !!safe_mode,
    }),
  };

  request(options, (err, res, data) => {
    const parsedBody = JSON.parse(data);

    console.log(parsedBody);

    const datestring = new Date().toISOString().replace(/:/g, "-");
    const filename = `${datestring}.json`;

    // save the parsed body to a json file
    fs.writeFile(
      `./public/images/${filename}`,
      JSON.stringify(parsedBody),
      (err) => {
        if (err) {
          console.log(err);
        } else {
          console.log(`Saved converted.json`);
        }
      }
    );

    apiRes.send("Images converted and saved to current directory");
  });
});

app.get("/load", (req, res) => {
  fs.readdir("./public/images", (err: any, files: any) => {
    if (err) {
      console.log(err);
      res.status(500).send("Error fetching converted images");
      return;
    }

    const jsonFiles = files.filter((file: any) => file.endsWith(".json"));

    const images = [];
    // take the content of all json
    for (let i = 0; i < jsonFiles.length; i++) {
      const jsonFile = jsonFiles[i];
      const jsonContent = fs.readFileSync(
        `./public/images/${jsonFile}`,
        "utf8"
      );
      const parsedJson = JSON.parse(jsonContent);
      images.push(parsedJson.images);
    }

    res.json(images);
  });
});

app.post("/convert", (apiReq, apiRes) => {
  const convert_to = apiReq.body.convert_to;
  const width = apiReq.body.width;
  const height = apiReq.body.height;
  const init_image = apiReq.body.init_image;

  const options = {
    method: "POST",
    url: `${API_URL}/convert`,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      convert_to: convert_to,
      width: parseInt(width),
      height: parseInt(height),
      init_image: init_image,
    }),
  };

  request(options, (err, res, data) => {
    const parsedBody = JSON.parse(data);

    console.log(parsedBody);

    const datestring = new Date().toISOString().replace(/:/g, "-");
    const filename = `${datestring}-${convert_to}.json`;

    // save the parsed body to a json file
    fs.writeFile(
      `./public/converted/${filename}`,
      JSON.stringify(parsedBody),
      (err) => {
        if (err) {
          console.log(err);
        } else {
          console.log(`Saved converted.json`);
        }
      }
    );

    apiRes.send("Images converted and saved to current directory");
  });
});

app.get("/load-converted", (req, res) => {
  fs.readdir("./public/converted", (err: any, files: any) => {
    if (err) {
      console.log(err);
      res.status(500).send("Error fetching converted images");
      return;
    }

    const jsonFiles = files.filter((file: any) => file.endsWith(".json"));

    const images = [];
    // take the content of all json
    for (let i = 0; i < jsonFiles.length; i++) {
      const jsonFile = jsonFiles[i];
      const jsonContent = fs.readFileSync(
        `./public/converted/${jsonFile}`,
        "utf8"
      );
      const parsedJson = JSON.parse(jsonContent);
      images.push(parsedJson.images);
    }

    res.json(images);
  });
});

app.listen(port, () => {
  console.log(`Server listening on port http://localhost:${port}`);
});
