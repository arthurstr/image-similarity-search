#
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

#
COPY ./requirements-docker.txt /requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

#
COPY ./src src

ENTRYPOINT ["uvicorn", "src.controller.controller:app", "--host", "0.0.0.0", "--port", "5000"]

