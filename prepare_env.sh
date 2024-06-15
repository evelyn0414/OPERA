#!/bin/bash
# Define your variables
GIT_REPO_URL="https://github.com/evelyn0414/OPERA.git"
PROJECT_DIR="$(pwd)"
echo "Project path $PROJECT_DIR"
BASHRC_FILE="$HOME/.bashrc"
 
# Check if the project path is already in .bashrc
if grep -q "$PROJECT_DIR" "$BASHRC_FILE"; then
    echo "Project path already exists in $BASHRC_FILE"
else
    # Add the project path to .bashrc
    echo "export PYTHONPATH=\$PYTHONPATH:$PROJECT_DIR" >> "$BASHRC_FILE"
    echo "Project path added to $BASHRC_FILE"
fi

