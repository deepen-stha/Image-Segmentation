#!/bin/bash

# isntalling dependencies for linux system
install_linux_dependencies() {
    sudo apt update
    sudo apt install -y cmake libopencv-dev
}

# installing dependencies for mac
install_mac_dependencies() {
    brew update
    brew install cmake opencv
}

# for windows we have to install manually
install_windows_dependencies() {
    echo "For Windows, please install dependencies manually or use MSYS2/WSL."
}

# on the basis of OS type run the command
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    install_linux_dependencies
elif [[ "$OSTYPE" == "darwin"* ]]; then
    install_mac_dependencies
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    install_windows_dependencies
else
    echo "Unsupported OS."
fi
