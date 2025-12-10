# FROM python:3.10.16

# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# RUN apt-get update && apt-get install -y curl && \
# curl -fsSL https://ollama.com/install.sh | sh

# COPY src/rule_handler/ .
# COPY DandD/ .
# COPY entrypoint.sh .

# EXPOSE 11434

# # Start everything via entrypoint
# ENTRYPOINT ["./entrypoint.sh"]



FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime


WORKDIR /

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


RUN apt-get update && apt-get install -y curl && \
curl -fsSL https://ollama.com/install.sh | sh


COPY src/rule_handler/ .
COPY DandD/ ./DandD
COPY entrypoint.sh .



EXPOSE 11434

# Start everything via entrypoint
ENTRYPOINT ["./entrypoint.sh"]