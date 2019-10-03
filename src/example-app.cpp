#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <memory>

std::shared_ptr<torch::jit::script::Module> import_module(const char* filename)
{
    std::shared_ptr<torch::jit::script::Module> module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(filename);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return nullptr;
    }

    return module;
}

int main(int argc, const char* argv[]) {
    // Check the number of arguments
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    
    // Import trained model
    std::shared_ptr<torch::jit::script::Module> module = import_module(argv[1]);

    // Import an image
    cv::Mat input_img = cv::imread(argv[2]);

    // Get the size of the input image
    int height = input_img.size().height;
    int width  = input_img.size().width;

    // Create a vector of inputs.
    at::Tensor input_tensor_ones  = torch::ones({1, 3, 264, 480}).to(torch::kCUDA);
    std::vector<int64_t>shape = {1, height, width, 3};
    at::Tensor input_tensor = torch::from_blob(input_img.data, at::IntList(shape), at::ScalarType::Byte).to(torch::kFloat).to(torch::kCUDA);
    input_tensor = at::transpose(input_tensor, 1, 2);
    input_tensor = at::transpose(input_tensor, 1, 3);

//    std::cout << input_tensor << std::endl;

    std::cout << input_tensor << std::endl;

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module->forward({input_tensor}).toTensor();
    // Calculate argmax to get a label on each pixel
    at::Tensor output_args = at::argmax(output, 1).to(torch::kCPU).to(at::kByte);

    // Convert to OpenCV
    cv::Mat mat(height, width, CV_8U, output_args[0]. template data<uint8_t>());

//    std::cout << mat.size() << std::endl;

//    mat.convertTo(mat, CV_8U, 255);
    cv::imwrite("hoge.jpg", mat);
    std::cout << mat << std::endl;
}
