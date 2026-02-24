# Opik — Self-Hosted Infrastructure

This directory contains the Docker Compose stack for running
[Opik](https://github.com/comet-ml/opik) locally.  
No Comet-ML cloud account or API key is required.

---

## Stack Components

| Service | Image | Purpose |
|:--------|:------|:--------|
| `frontend` | `opik-frontend` | React UI — http://localhost:5173 |
| `backend` | `opik-backend` | REST API — http://localhost:5173/api |
| `python-backend` | `opik-python-backend` | Python evaluator / optimizer |
| `mysql` | `mysql:8.4.2` | Relational state store |
| `clickhouse` | `clickhouse-server:25.3` | Analytics / trace store |
| `zookeeper` | `zookeeper:3.9.4` | ClickHouse coordination |
| `redis` | `redis:7.2.4-alpine` | Job queue & caching |
| `minio` | `minio/minio` | S3-compatible object storage |
| `mc` | `minio/mc` | MinIO bucket initialiser (one-shot) |

---

## Quick Start

### Prerequisites
- Docker ≥ 24 and Docker Compose v2 (`docker compose version`)
- ~4 GB free RAM for the full stack

### 1 — Start the stack

```bash
./infra/opik/opik.sh up
```

First run pulls all images (~2–3 GB). Subsequent starts are fast.  
The UI becomes available at **http://localhost:5173** once the `frontend`
healthcheck passes (typically 60–90 s on first boot).

### 2 — Configure the opik Python SDK

```bash
./infra/opik/opik.sh configure
```

This writes `~/.opik.config`:

```ini
[opik]
url_override = http://localhost:5173/api/
workspace = default
```

The `opik` Python package reads this file automatically — no environment
variables or code changes needed.

### 3 — Upload the dataset

```bash
python src/upload_to_opik.py
```

---

## Management Commands

```bash
./infra/opik/opik.sh up        # Start stack (detached)
./infra/opik/opik.sh down      # Stop containers (data preserved)
./infra/opik/opik.sh destroy   # Stop + delete all volumes (full reset)
./infra/opik/opik.sh logs      # Tail all service logs
./infra/opik/opik.sh status    # Show container health
./infra/opik/opik.sh configure # Write ~/.opik.conf for local SDK
```

Or use `docker compose` directly from this directory:

```bash
cd infra/opik
docker compose --env-file .env up -d
docker compose --env-file .env ps
docker compose --env-file .env down
```

---

## Configuration

Edit [`infra/opik/.env`](.env) to customise:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OPIK_VERSION` | `latest` | Image tag to pull |
| `NGINX_PORT` | `5173` | UI / API port |
| `MINIO_ROOT_USER` | `THAAIOSFODNN7EXAMPLE` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `LESlrXUtnFEMI/...` | MinIO secret key |
| `OPIK_USAGE_REPORT_ENABLED` | `false` | Anonymous telemetry to Comet-ML |

---

## Data Persistence

Named Docker volumes store all persistent data:

| Volume | Contents |
|:-------|:---------|
| `opik_mysql` | MySQL database |
| `opik_clickhouse` | ClickHouse data |
| `opik_clickhouse-server` | ClickHouse logs |
| `opik_zookeeper` | ZooKeeper state |
| `opik_redis-data` | Redis AOF |
| `opik_minio-data` | Object storage blobs |

`./infra/opik/opik.sh down` stops containers but **preserves** volumes.  
`./infra/opik/opik.sh destroy` removes containers **and** volumes (irreversible).

---

## Troubleshooting

**Backend never becomes healthy**  
Check MySQL and ClickHouse first — the backend waits for both:
```bash
./infra/opik/opik.sh logs | grep -E "(mysql|clickhouse|backend)"
```

**Port 5173 already in use**  
Change `NGINX_PORT` in `.env` and re-run `opik.sh configure` to update `~/.opik.conf`.

**Full reset**  
```bash
./infra/opik/opik.sh destroy
./infra/opik/opik.sh up
```
