FROM lmsysorg/sglang:latest
RUN useradd -m -s /bin/bash benchuser
USER benchuser
EXPOSE 30000
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
