FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    curl \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libx11-xcb1 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libglx0 \
    libglvnd0 \
    libopengl0 \
    libdbus-1-3 \
    libxkbcommon0 \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . . 

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_NO_SYNC=1

CMD ["uv", "run", "python3", "src/AI_dio/UI/sound_app.py"]
