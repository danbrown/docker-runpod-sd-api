
ARG FROM_IMAGE="odanielbrown/sd-base:v10"
FROM ${FROM_IMAGE} as base
ENV FROM_IMAGE=${FROM_IMAGE}

# use bash as default shell
WORKDIR /workspace

SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

# get the api
RUN git clone https://github.com/danbrown/docker-runpod-sd-api.git api

# install requirements
WORKDIR /workspace/api
RUN pip install -r requirements.txt
WORKDIR /workspace

# copy over run script, it will run the server in ldm environment
ADD run.sh .

# expose ports, 8888 for jupyter, 3000 for the server
EXPOSE 8888 3000

# Start
ADD start.sh .
RUN chmod a+x start.sh
CMD ["./start.sh"]