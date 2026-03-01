FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . . 

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_NO_SYNC=1

CMD ["uv", "run", "python3", "main.py"]
