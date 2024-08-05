FROM isabek/opencv-3.4:latest

# Set the working directory
WORKDIR /app

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
    gcc-7 \
    g++-7 \
    cmake \
    make \
    wget \
    zlib1g-dev && \  
    rm -rf /var/lib/apt/lists/*

# Set GCC 7 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 60

# Install a newer version of CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.sh && \
    chmod +x cmake-3.18.4-Linux-x86_64.sh && \
    ./cmake-3.18.4-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.18.4-Linux-x86_64.sh

# Copy the current directory contents into the container at /app
COPY . .

# Create build directory
RUN mkdir build

# Set the working directory to /app/build
WORKDIR /app/build

ENV CXXFLAGS="-std=c++17"

ENV LDFLAGS="-lstdc++fs"

# Configure the project using CMake
RUN cmake -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 ..

# Build the project
RUN make

# Run the compiled binary
CMD ["./Osiris"]
