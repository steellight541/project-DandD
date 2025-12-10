#!/bin/bash
set -e

OLLAMA_LOG_FILE="/ollama.log" # Define a log file location

ollama serve > "$OLLAMA_LOG_FILE" 2>&1 & 

echo "Waiting for Ollama service to start..."

# --- Robust Ollama Readiness Check ---
OLLAMA_URL="http://127.0.0.1:11434"
TIMEOUT=60
START_TIME=$(date +%s)

while ! curl -s $OLLAMA_URL > /dev/null; do
    ELAPSED_TIME=$(( $(date +%s) - START_TIME ))
    if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
        echo "Error: Ollama failed to start within $TIMEOUT seconds."
        # Print the last few lines of the log for debugging
        echo "--- Ollama Log Tail ---"
        tail -n 10 "$OLLAMA_LOG_FILE"
        exit 1
    fi
    echo "Ollama not ready yet, waiting..."
    sleep 5
done
echo "Ollama is running."
# -------------------------------------

# Pull the model. The output of the pull command will still show on your console.
if ! ollama list | grep -q $EMBEDDED_OLLAMA_MODEL; then
    echo "Pulling $EMBEDDED_OLLAMA_MODEL model..."
    ollama pull $EMBEDDED_OLLAMA_MODEL
fi

# Give it an additional short wait after the pull/load to ensure it's in memory
sleep 5 

export OLLAMA_HOST=127.0.0.1:11434
# OLLAMA_DEBUG is now irrelevant for console output, but harmless to leave.
export OLLAMA_DEBUG="ERROR" 

# Now run your main application. Only its output (and the pull output) will be visible.
python -u main.py

# Optional: Stop Ollama when finished
# kill %1