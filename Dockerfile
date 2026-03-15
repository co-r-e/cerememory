# Cerememory Dockerfile — multi-stage build
# Stage 1: Build
FROM rust:1.77-bookworm AS builder

WORKDIR /app

# Copy workspace manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY adapters/ adapters/
COPY tests/ tests/
COPY proto/ proto/

# Build in release mode
RUN cargo build --release --bin cerememory

# Stage 2: Runtime (distroless for minimal attack surface)
FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder /app/target/release/cerememory /usr/local/bin/cerememory

# Default config
ENV CEREMEMORY_DATA_DIR=/data
ENV CEREMEMORY_HTTP__PORT=8420

EXPOSE 8420 8421

VOLUME ["/data"]

# Use the built-in healthcheck subcommand (no curl/wget needed in distroless)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["cerememory", "healthcheck"]

ENTRYPOINT ["cerememory"]
CMD ["serve"]
