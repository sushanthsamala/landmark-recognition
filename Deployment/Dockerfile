FROM python:3.6
WORKDIR /app



COPY main.py /app
COPY index_to_label.txt /app
COPY requirements.txt /app
COPY inception_50.pth /app
ADD templates /app/templates



RUN pip install -r ./requirements.txt
	



ENTRYPOINT ["python", "main.py"]~