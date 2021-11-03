FROM python:3.8.5-slim
RUN apt-get update
RUN apt-get install -y xvfb python-opengl ffmpeg
ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
WORKDIR /code
CMD xvfb-run -s "-screen 0 1400x900x24" python -u checker_client.py
