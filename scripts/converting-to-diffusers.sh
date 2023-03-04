
mkdir diff && cd diff
wget https://civitai.com/api/download/models/11925

mkdir ned && mkdir out
mv 11925 ./ned/NeverEnding-Dream.safetensors

git clone https://github.com/danbrown/ckpt-to-diffusers.git

cd ckpt-to-diffusers

python ./convert_2/convert_diffusers20_original_sd.py ../ned/NeverEnding-Dream.safetensors ../out/ --v1 --reference_model runwayml/stable-diffusion-v1-5

cd ../out && mkdir feature_extractor && cd feature_extractor
wget https://huggingface.co/danbrown/A-to-Zovya-RPG-v1-5/resolve/main/feature_extractor/preprocessor_config.json

cd .. && mkdir safety_checker && cd safety_checker
wget https://huggingface.co/danbrown/A-to-Zovya-RPG-v1-5/resolve/main/safety_checker/pytorch_model.bin
wget https://huggingface.co/danbrown/A-to-Zovya-RPG-v1-5/resolve/main/safety_checker/config.json

cd ..

cp ../ned/NeverEnding-Dream.safetensors .

git init && git checkout -b main
git config --global user.email "" && git config --global user.name ""
git remote add origin https://huggingface.co/danbrown/NeverEnding-Dream
git pull origin main

git add NeverEnding-Dream.safetensors && git commit -m "added model"
git add . && git commit -m "add diffusers weights"

huggingface-cli lfs-enable-largefiles .
git push origin main
