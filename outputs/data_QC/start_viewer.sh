#!/bin/bash

# Start local HTTP server for data QC viewer
# Usage: ./start_viewer.sh [port]

PORT=${1:-8000}

echo "=========================================="
echo "Starting Data QC Viewer Local Server"
echo "=========================================="
echo ""
echo "Server will start on port $PORT"
echo ""
echo "Once started, open your browser and go to:"
echo ""
echo "    http://localhost:$PORT/viewer.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Start Python HTTP server
python -m http.server $PORT
