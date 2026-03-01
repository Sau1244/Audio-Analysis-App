# Notes for Developers

## 1. Setting up the project

### A. Running inside container (preferred)

Build the docker image:

1. Development container:

```bash
docker build -f docker/development.dockerfile . -t pite-dev
```

2. Training container:

```bash
docker build -f docker/training.dockerfile . -t pite-train
```

Run the image:

1. Development container:

```bash
docker run -it --env DISPLAY=$DISPLAY --env XAUTHORITY=$XAUTHORITY -v $XAUTHORITY:$XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all pite-dev:latest
```

2. Training container:

```bash
docker run -it --gpus all pite-train:latest
```

### B. Running locally

Install dev depenendencies:

```bash
uv sync --group dev
```

Run project:

```bash
uv run python3 main.py
```

## 2. Pre-commit hooks

```bash
uv run pre-commit install
```

After that, linters will run on each commit. You can also run them manually:

```bash
uv run pre-commit run --all-files
```

## 3. Pull requests

#### Please refrain from pushing onto `main`!

Every change should be submitted as a **Pull Request**. This approach enables two things:

1. Other developers can review the code
2. Automatic workflows will be ran to test and check the code

**NOTE:** Consider using understandable commit titles, for example:

```
add: saving output to file
fix: out-of-bounds bug
```

## 4. Miscellaneous

1. All application source files should be inside `/src/AI-dio/` in adequate subdirectories (f.e. `gui`, `audio`, `model`)
2. Avoid pushing images/audio files/other auxiliary files into the repository
3. Keep correct naming conventions (PEP8)

## 5. Using the AI for inference

For files:

```python
from AI_dio.inference import predict_file

result = predict_file(
    checkpoint="checkpoints/your_checkpoint.pt",
    wav="path/to/audio.wav",
)
print(result.score, result.label)
```

For in-memory tensors:

```python
import torch
from AI_dio.inference import predict_audio

audio = torch.randn(1, 16000)
result = predict_audio(
    checkpoint="checkpoints/your_checkpoint.pt",
    audio=audio,
    sample_rate=16000,
)
print(result.score, result.label)
```

Output fields:

- `score`: mean probability for a fake across windows (0.0–1.0)
- `scores`: per-window probabilities in order
- `label`: `"fake"` if `score >= threshold`, else `"real"`
- `threshold`: threshold used for the decision
- `window_sec`/`stride_sec`: windowing parameters used
- `wav`: resolved path when using `predict_file`
