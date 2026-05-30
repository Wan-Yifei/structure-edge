# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer / runner)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies first (cached layer if requirements don't change)
COPY docker/requirements-backtest.txt /tmp/requirements-backtest.txt
RUN uv pip install --system -r /tmp/requirements-backtest.txt

# Copy project source
COPY . .

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# AWS CLI v2 (official thin installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    && apt-get remove -y curl unzip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project
COPY --from=builder /app /app

WORKDIR /app

# Non-root user for HPC environments that disallow root containers
RUN useradd -m -u 1000 backtest \
    && chown -R backtest:backtest /app \
    && mkdir -p /home/backtest/.aws \
    && chown backtest:backtest /home/backtest/.aws
USER backtest

# Runtime dirs (may be bind-mounted from host)
RUN mkdir -p /app/db /app/backtest/results

# Fix Windows CRLF line endings and set executable bit
RUN sed -i 's/\r//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
