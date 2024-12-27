# check python version

set -e

PYTHON_VERSION_SATISFIED=$(python -c 'import sys; print(sys.version_info[:2] >= (3, 10))')
if [ "$PYTHON_VERSION_SATISFIED" != 'True' ]; then
    echo "Please, use Python 3.10"
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

mkdir -p libs
cd libs
REPO_URL="https://github.com/TeUm3048/Open3D.git"
BRANCH_NAME="v0.18.0-patch-avx-off"

if [ -d "Open3D" ]; then
    echo "Directory Open3D already exists. Checking branch..."
    cd Open3D
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
        echo "Switching to branch $BRANCH_NAME..."
        git fetch origin
        git checkout $BRANCH_NAME
        git pull origin $BRANCH_NAME
    else
        echo "Repository is already on the correct branch."
    fi
else
    echo "Cloning forked Open3D repository..."
    git clone --single-branch --branch $BRANCH_NAME $REPO_URL
    cd Open3D
fi
echo "Installing Open3D..."

# Install dependencies
echo "Installing dependencies..."
bash ./util/install_deps_ubuntu.sh assume-yes

# Build Open3D
echo "Building Open3D..."
mkdir -p build
cd build
cmake -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/open3d_install ..
make -j$(nproc)

# Install Open3D
echo "Installing Open3D..."
make install

# Add Open3D to PYTHONPATH
echo "Adding Open3D to PYTHONPATH..."
make install-pip-package
cd lib/python_package
python -m pip install .
echo "Checking Open3D installation..."
OPEN3D=$(python -c 'import open3d; print(1)')
if [ "$OPEN3D" != '1' ]; then
    echo "Open3D is not installed :("
    exit 1
fi
