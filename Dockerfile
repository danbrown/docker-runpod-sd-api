
ARG FROM_IMAGE="odanielbrown/runpod-base:v7"
FROM ${FROM_IMAGE} as base
ENV FROM_IMAGE=${FROM_IMAGE}

# use bash as default shell
WORKDIR /workspace

SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

# install requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

# install diffusers
RUN git clone https://github.com/huggingface/diffusers && cd diffusers && \
  git checkout 39a3c77e0d4a22de189b02398cf2d003d299b4ae && cd ..
RUN pip install -e diffusers

# copy over the code

ADD app.py .
ADD server.py .
ADD run.sh .

# expose ports
EXPOSE 8888 3000

# Start
ADD start.sh .
RUN chmod a+x start.sh
CMD ["./start.sh"]