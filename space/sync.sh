#!/usr/bin/env bash
# Copy the nes2net package next to app.py before pushing the Space.
set -e; cd "$(dirname "$0")"; rm -rf nes2net; mkdir nes2net
cp ../nes2net/{__init__,backend,frontend,model,data,metrics,rawboost}.py nes2net/
