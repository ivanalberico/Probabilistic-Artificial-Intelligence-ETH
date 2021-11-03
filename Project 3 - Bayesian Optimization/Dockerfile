FROM python:3.8.5-slim
ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
WORKDIR /code
CMD python -u checker_client.py
