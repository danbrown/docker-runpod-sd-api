#!/bin/bash
echo "Container Started"
echo "conda activate ldm" >> ~/.bashrc
source ~/.bashrc

# if has a run.sh file, chmod it and run it in the background
if [ -f /workspace/run.sh ]
then
    chmod +x /workspace/run.sh
    echo "run.sh file found, running it in the background in the ldm environment"
    conda run -n ldm bash -c "/workspace/run.sh &" &
fi

# Start SSH Service
if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
    echo "SSH Service Started"
fi

# Start Jupyter Lab
if [[ $JUPYTER_PASSWORD ]]
then
    cd /
    conda run -n ldm bash -c "jupyter lab --ip=* --port=8888 --no-browser --allow-root --ServerApp.token='$JUPYTER_PASSWORD' --ServerApp.allow_origin=* --FileContentsManager.preferred_dir=/workspace --NotebookApp.terminado_settings='{\"shell_command\": [\"/bin/bash\"]}' &" &
    echo "Jupyter Lab Started"
fi

echo "Container Started Successfully"

sleep infinity

