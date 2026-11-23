
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Building EGT309 Docker image..."
docker build -t egt309-ml:latest .

echo "Starting EGT309 JupyterLab..."
docker run --rm -it \
  -p 127.0.0.1:8888:8888 \
  -v "$(pwd):/app" \
  --name egt309-container \
  egt309-ml:latest
