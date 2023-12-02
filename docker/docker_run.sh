FROM ubuntu:latest
COPY . /digits
WORKDIR /digits
VOLUME /digits/models
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN pip3 install -r requirements.txt
CMD ["python3", "exp.py"]