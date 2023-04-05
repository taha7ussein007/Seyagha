FROM python:3.9.7

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y && \
    apt-get install build-essential -y \
    supervisor wget unzip

WORKDIR /user/src/Seyagha/

COPY paraphrasingModule.py /user/src/Seyagha/paraphrasingModule.py
COPY main.py /user/src/Seyagha/main.py
COPY requirements.txt /user/src/Seyagha/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]