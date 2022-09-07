# multi-stage
# base image
ARG PY_VER=3.8-slim
FROM python:$PY_VER AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY . .

FROM python:$PY_VER
COPY --from=compile-image /root/.local /root/.local
COPY . .
# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "src/train.py"]


