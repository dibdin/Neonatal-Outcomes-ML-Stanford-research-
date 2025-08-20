#!/bin/bash

# Configuration
USER="dibadin"
REMOTE_HOST="login.sherlock.stanford.edu"
REMOTE_DIR="/home/users/dibadin/serum_biomarkers"
LOCAL_DIR="."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔄 Syncing files to Sherlock...${NC}"

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo -e "${RED}❌ Error: rsync is not installed${NC}"
    exit 1
fi

# Test SSH connection first
echo -e "${YELLOW}🔍 Testing SSH connection...${NC}"
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$USER@$REMOTE_HOST" exit 2>/dev/null; then
    echo -e "${RED}❌ Error: Cannot connect to $REMOTE_HOST${NC}"
    echo -e "${YELLOW}💡 Make sure you have SSH access set up with key-based authentication${NC}"
    exit 1
fi

# Create remote directory if it doesn't exist
echo -e "${YELLOW}📁 Ensuring remote directory exists...${NC}"
ssh "$USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Sync files with better exclusions and error handling
echo -e "${YELLOW}📤 Starting file sync...${NC}"
if rsync -avz --progress \
    --exclude='.git/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='*.log' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.tmp' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --timeout=60 \
    "$LOCAL_DIR/" \
    "$USER@$REMOTE_HOST:$REMOTE_DIR/"; then
    
    echo -e "${GREEN}✅ Sync completed successfully!${NC}"
    echo -e "${YELLOW}💡 To run your updated code on Sherlock:${NC}"
    echo -e "   ssh $USER@$REMOTE_HOST"
    echo -e "   cd $REMOTE_DIR"
    echo -e "   sbatch job.slurm"
else
    echo -e "${RED}❌ Sync failed!${NC}"
    echo -e "${YELLOW}💡 Common issues:${NC}"
    echo -e "   - Check your SSH connection"
    echo -e "   - Verify you have write permissions on $REMOTE_DIR"
    echo -e "   - Check if the remote directory exists"
    exit 1
fi
