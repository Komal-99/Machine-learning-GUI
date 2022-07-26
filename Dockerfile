#Python's Alpine Base Image
FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
#Start Flask Server
CMD [ "python","/codes/uicode.py"]
#Expose server port