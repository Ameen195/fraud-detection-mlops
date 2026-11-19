FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src src
COPY config config
COPY docker docker
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
# data, registry and results are mounted by docker-compose, not baked into the image
ENTRYPOINT ["sh", "/app/docker/entrypoint.sh"]
CMD ["uvicorn", "fraudguard.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
