#!/bin/bash

set -e

echo "=========================================="
echo "Python 3.12 and pip 25 Installation Script"
echo "=========================================="

echo "Updating package list..."
apt-get update

echo "Installing dependencies..."
apt-get install -y build-essential libssl-dev zlib1g-dev libncurses5-dev \
libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev wget curl

echo "Downloading Python 3.12..."
cd /tmp
PYTHON_VERSION="3.12.7"
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz

echo "Extracting Python source..."
tar -xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

echo "Configuring Python build..."
./configure --enable-optimizations --with-ensurepip=install

echo "Building Python (this may take several minutes)..."
make -j$(nproc)

echo "Installing Python 3.12..."
make altinstall

echo "Cleaning up..."
cd /tmp
rm -rf Python-${PYTHON_VERSION}*

echo "Verifying Python 3.12 installation..."
python3.12 --version

echo "Upgrading pip to version 25..."
python3.12 -m pip install --upgrade pip==25.0

echo "Verifying pip installation..."
python3.12 -m pip --version

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
