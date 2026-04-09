# Cerememory Dockerfile — multi-stage build
# Stage 1: Build
FROM rust:1.88-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY adapters/ adapters/
COPY bindings/python-native/ bindings/python-native/
COPY bindings/typescript-native/ bindings/typescript-native/
COPY tests/ tests/

RUN cargo build --locked --release --bin cerememory \
    && mkdir -p /tmp/cerememory-data

# Stage 2: Runtime (distroless for minimal attack surface)
FROM gcr.io/distroless/cc-debian12:nonroot

LABEL org.opencontainers.image.source="https://github.com/co-r-e/cerememory"

COPY --from=builder --chown=nonroot:nonroot /app/target/release/cerememory /usr/local/bin/cerememory
COPY --from=builder --chown=nonroot:nonroot /tmp/cerememory-data /data

ENV CEREMEMORY_DATA_DIR=/data
ENV CEREMEMORY_HTTP__BIND_ADDRESS=0.0.0.0
ENV CEREMEMORY_HTTP__PORT=8420

EXPOSE 8420 8421

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["cerememory", "healthcheck"]

USER nonroot:nonroot

ENTRYPOINT ["cerememory"]
CMD ["serve"]
