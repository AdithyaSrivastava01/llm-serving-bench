FROM vllm/vllm-openai:latest
RUN useradd -m -s /bin/bash benchuser
USER benchuser
EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
