export PYTHON_MAJOR=3
export PYTHON_MINOR=10
export PYTHON_MICRO=15
export PYTHON_VERSION=${PYTHON_MAJOR}.${PYTHON_MINOR}.${PYTHON_MICRO}

echo "Installing Python${PYTHON_VERSION}..."

echo "Installing dependencies..."
sudo apt-get update
sudo apt-get -y install \
    curl \
    gcc \
    libbz2-dev \
    libev-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    make \
    tk-dev \
    wget \
    zlib1g-dev


echo "Downloading Python-${PYTHON_VERSION}.tgz..."
curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz

echo "Extracting Python-${PYTHON_VERSION}.tgz..."
tar -xvzf Python-${PYTHON_VERSION}.tgz

cd Python-${PYTHON_VERSION}

echo "Configuring Python-${PYTHON_VERSION}..."
./configure --enable-optimizations

echo "Building Python-${PYTHON_VERSION}..."
make -j $(nproc)
sudo make altinstall

echo "Python version:"
python${PYTHON_MAJOR}.${PYTHON_MINOR} -V

cd ..

echo "Cleaning up..."
sudo rm -Rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz
