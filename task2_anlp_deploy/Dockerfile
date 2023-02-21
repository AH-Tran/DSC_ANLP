FROM python:3.7.12-slim-buster

RUN pip3 install pandas jupyterlab docker
RUN pip3 install farm==0.7.1
RUN pip3 install protobuf==3.20.*


COPY . .
RUN chmod +x /run_task_2.py

ENTRYPOINT [ "python", "/run_task_2.py" ]
