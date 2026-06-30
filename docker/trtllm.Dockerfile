FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
RUN useradd -m -s /bin/bash benchuser
VOLUME ["/engine", "/model_repo"]
USER benchuser
EXPOSE 8000 8001
ENTRYPOINT ["tritonserver", "--model-repository=/model_repo"]
