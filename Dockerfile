# nbody-sim runtime image (SPEC §10).
# Pure-python + numpy/matplotlib -> a single slim stage is optimal; no build tools needed.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Layer-cache friendly order: metadata + package first (changes rarely),
# experiment/benchmark scripts second (change often).
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

COPY experiments ./experiments
COPY benchmarks ./benchmarks

# Non-root runtime; output dirs pre-created so bind mounts stay writable.
RUN useradd --create-home runner \
    && mkdir -p experiments/out figures benchmarks/out \
    && chown -R runner:runner /app
USER runner

# Default: the two-Plummer collision (JSON for the player + montage figure).
# Any other script works as an override, e.g.:
#   docker run --rm nbody-sim benchmarks/bench_scaling.py
ENTRYPOINT ["python"]
CMD ["experiments/exp_cluster.py"]
