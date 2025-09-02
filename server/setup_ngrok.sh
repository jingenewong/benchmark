#!/bin/bash

# Set install dir
INSTALL_DIR="$HOME/bin"

# Download ngrok v3
echo "Downloading ngrok v3..."
wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -O ngrok.tgz

# Unpack and move to ~/bin
echo "Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
tar -xzf ngrok.tgz || { echo "❌ Failed to extract ngrok.tgz"; exit 1; }

mv ngrok "$INSTALL_DIR/ngrok"
rm ngrok.tgz

# Add to PATH if not already there
if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc"; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
  echo "Added ngrok to PATH in .bashrc"
fi

# Source the new path for this session
export PATH="$HOME/bin:$PATH"

# Verify install
echo "Installed ngrok version:"
ngrok version
