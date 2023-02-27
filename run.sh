#!/bin/bash

# clone the updated repo
git clone https://github.com/danbrown/docker-runpod-sd-api.git api

# install the requirements, it is already installed in the docker image, but just in case
pip install -r api/requirements.txt

# runs the server in the background, if the SERVER_AUTOSTART is set
if [ $SERVER_AUTOSTART ] ; then
    python api/server.py &
fi
