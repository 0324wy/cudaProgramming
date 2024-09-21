#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define channel 3

__global__ void colorToGray(unsigned char *pin, unsigned char *pout, int width, int height)
{

    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < width && col < height)
    {
        int grayOffset = row * width + col;

        int colorOffset = grayOffset * channel;

        unsigned char r = pin[colorOffset];
        unsigned char g = pin[colorOffset + 1];
        unsigned char b = pin[colorOffset + 2];

        pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main() {
    string filePath = "./image.jpeg";  // Replace with your image file path

    // Load image using OpenCV
    Mat img = imread(filePath, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Error: Cannot load image!" << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int imgSize = width * height * 3;  // Size of RGB image (3 channels)
    int graySize = width * height;     // Size of grayscale image (1 channel)

    unsigned char *h_gray = (unsigned char *)malloc(graySize);

    unsigned char *d_gray;
    cudaMalloc((void **)&d_gray, graySize * sizeof(unsigned char));

    unsigned char *h_color = img.data;

    unsigned char *d_color;
    cudaMalloc((void **)&d_color, imgSize * sizeof(unsigned char));


    cudaMemcpy(d_color, h_color, imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadPerBlock(64, 64);
    dim3 BlockPerGrid(ceil(width / 64.0), ceil(height / 64.0));

    colorToGray<<<BlockPerGrid, threadPerBlock>>>(d_color, d_gray, width, height);

    cudaMemcpy(h_gray, d_gray, graySize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    Mat grayImg(height, width, CV_8UC1, h_gray);

    imwrite("grayscale_image.png", grayImg);

    cudaFree(d_color);
    cudaFree(h_gray);

    free(h_color);
    free(h_gray);
}