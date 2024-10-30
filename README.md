# Image Segmentation
## Install Dependencies
1. change the mode of the file `chmod +x dependencies.sh`.
2. install the dependencies `./dependencies.sh`.
## Requirments
1. Install openCV for reading image pixels.
2. Install CMake
3. Make changes in the CMakeLists.txt file according to the package installation.
## Build and Running
1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `cmake --build . --config Release`
5. `cd ..` come out of build directory.
5. `.\build\Release\imagesegmentation.exe` run the file.
5. `cmake --build . --config Release --target clean` if you want to clean the build directory.