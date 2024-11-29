export GCC_VERSION=11
echo "Upgrading GCC to version ${GCC_VERSION}..."

sudo apt-get update


echo "Add repository ppa:ubuntu-toolchain-r/test with GCC_VERSION=11" 
sudo apt-get -y install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test

echo "Install gcc-${GCC_VERSION} and g++-${GCC_VERSION}"
sudo apt -y install gcc-${GCC_VERSION} g++-${GCC_VERSION}

echo "Set gcc-${GCC_VERSION} and g++-${GCC_VERSION} as default"

sudo update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} ${GCC_VERSION}0 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION}

echo "\ngcc version:"
gcc --version

echo "You can switch between versions using the command:"
echo "sudo update-alternatives --config gcc"
