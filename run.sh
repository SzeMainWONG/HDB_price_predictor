#!/usr/bin/env bash
# Simplified WSL Script for VS Code Ubuntu
# With Virtual Environment Support

set -e

echo "========================================="
echo "  Running Python Model & Streamlit App   "
echo "  WSL Environment Detected               "
echo "========================================="

# Check and activate virtual environment (only if not already active)
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "Setting up virtual environment..."
    [ ! -d "venv" ] && python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment activated!"
else
    echo ""
    echo "✅ Already in virtual environment: $(basename $VIRTUAL_ENV)"
fi

# Install dependencies only if needed
echo ""
echo "Checking dependencies..."
if [ -f "requirements.txt" ]; then
    # Check if any package is missing
    MISSING=false
    while IFS= read -r package; do
        # Skip empty lines and comments
        [[ -z "$package" || "$package" =~ ^# ]] && continue
        # Extract package name (remove version specifiers)
        pkg_name=$(echo "$package" | sed -E 's/([a-zA-Z0-9_-]+).*/\1/')
        if ! pip show "$pkg_name" &>/dev/null; then
            MISSING=true
            break
        fi
    done < requirements.txt
    
    if [ "$MISSING" = true ]; then
        echo "📦 Installing missing packages from requirements.txt..."
        pip install -r requirements.txt
        echo "✅ Packages installed!"
    else
        echo "✅ All requirements already installed!"
    fi
else
    # Install common packages if missing
    if ! pip show streamlit &>/dev/null; then
        echo "📦 Installing common packages (streamlit, pandas, numpy)..."
        pip install streamlit pandas numpy
        echo "✅ Packages installed!"
    else
        echo "✅ Common packages already installed!"
    fi
fi

# Show installed packages (minimal)
echo ""
echo "Key packages installed:"
pip list | grep -E "streamlit|pandas|numpy|torch|tensorflow|scikit|transformers" || pip list | head -10

# Run the model
echo ""
echo "[1/2] Running my-model.py..."
if python my-model.py; then
    echo "✅ my-model.py completed successfully!"
else
    echo "❌ my-model.py failed!"
    exit 1
fi

# Set browser for WSL
export BROWSER="powershell.exe -c Start-Process"

# Run Streamlit
echo ""
echo "[2/2] Starting Streamlit app..."
echo "The app will open in your Windows browser."
echo "Press Ctrl+C to stop the server."
echo ""
streamlit run my-app.py