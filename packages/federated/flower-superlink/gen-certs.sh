#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Generate a self-signed CA + server certificate for the Flower SuperLink.
#
# Usage:   SUPERLINK_IP=192.168.2.33 ./gen-certs.sh [output-dir]
# Default: SUPERLINK_IP=127.0.0.1, output-dir=$PWD/workspace/flower/superlink-certificates
#
# Outputs (matching the upstream tutorial's expected layout):
#   <out>/ca.crt        — root cert; copy to each client machine
#   <out>/server.pem    — server cert (SAN includes SUPERLINK_IP)
#   <out>/server.key    — server private key
#
# After generation, the flower-superlink Ryzer will pick these up via its
# volume mount (config.yaml maps the output dir to /app/certificates:ro).

set -euo pipefail

SUPERLINK_IP="${SUPERLINK_IP:-127.0.0.1}"
OUT_DIR="${1:-$PWD/workspace/flower/superlink-certificates}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "Generating Flower TLS certs for SUPERLINK_IP=${SUPERLINK_IP} in ${OUT_DIR}"

# 1. CA
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 \
  -subj "/CN=Flower SuperLink CA" \
  -out ca.crt

# 2. Server key + CSR
openssl genrsa -out server.key 4096
openssl req -new -key server.key \
  -subj "/CN=${SUPERLINK_IP}" \
  -out server.csr

# 3. Server cert signed by the CA, with SAN
cat >server.ext <<EOF
subjectAltName = IP:${SUPERLINK_IP},IP:127.0.0.1,DNS:localhost
extendedKeyUsage = serverAuth
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.pem -days 365 -sha256 -extfile server.ext

rm -f server.csr server.ext ca.srl
chmod 644 ca.crt server.pem
chmod 600 ca.key server.key

echo
echo "Done. Files:"
ls -l "$OUT_DIR"
echo
echo "Next:"
echo "  - Copy ca.crt to each client machine."
echo "  - Start the SuperLink with:"
echo "      ryzers run flower-superlink \\"
echo "        --ssl-ca-certfile=/app/certificates/ca.crt \\"
echo "        --ssl-certfile=/app/certificates/server.pem \\"
echo "        --ssl-keyfile=/app/certificates/server.key"
