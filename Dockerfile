#Python's Alpine Base Image
FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 8080
#Start Flask Server

CMD ["python","uicode.py"]
