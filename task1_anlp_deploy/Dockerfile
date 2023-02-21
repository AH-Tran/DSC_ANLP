FROM python:3.7.12-slim-buster

#WORKDIR /usr/app/src

RUN pip3 install pandas jupyterlab docker simpletransformers
#RUN pip3 install transformers
RUN pip3 install torch
RUN pip3 install -U pip setuptools wheel
RUN pip3 install spacy
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.5.0/en_core_web_md-3.5.0-py3-none-any.whl 
# RUN pip3 install mlflow==1.13.1
RUN pip3 install onnxruntime

COPY . .
RUN chmod +x /run_task_1.py
#RUN chmod +x /run_task_1_nner.py

ENTRYPOINT [ "python", "/run_task_1.py" ]
#ENTRYPOINT [ "python", "/run_task_1_nner.py" ]


