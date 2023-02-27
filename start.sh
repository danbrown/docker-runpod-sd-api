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
    ln -sf /examples /workspace
    ln -sf /root/welcome.ipynb /workspace

    pip install jupyterlab
    pip install jupyterlab-git
    pip install jupyter

    cd /
    jupyter lab --allow-root --no-browser --port=8888 --ip=* \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace
    echo "Jupyter Lab Started"
fi

echo "Container Started Successfully"

sleep infinity