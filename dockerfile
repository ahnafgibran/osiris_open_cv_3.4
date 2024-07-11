# Use the custom OpenCV base image
FROM open_cv_3.4:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory to the working directory
COPY . .

# Install any additional dependencies if necessary
RUN apt-get update && \
    apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

# Create a build directory
RUN mkdir build

# Set the working directory to the build directory
WORKDIR /app/build

# Configure the project using CMake
RUN cmake ..

# Add a temporary long-running command to pause the build
# RUN sleep infinity

# Build the project using CMake
RUN make

# # Set the entry point to run the built executable
CMD ["./Osiris"]
