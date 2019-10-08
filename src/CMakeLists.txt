cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(Torch_DIR "/opt/pytorch/pytorch/torch/share/cmake/Torch/")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)
add_executable(example-app example-app.cpp pytorch_cpp_wrapper.cpp)
target_link_libraries(example-app ${TORCH_LIBRARIES} ${Open_CV_LIBS}
	                          opencv_core opencv_highgui opencv_imgcodecs)
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
