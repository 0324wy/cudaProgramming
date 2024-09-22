#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BlurSize 4

__global__ void colorBlur(unsigned char *pin, unsigned char *pout, int width, int height)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < height && col < width)
    {
        for (int channel = 0; channel < 3; ++channel) {  // Loop over RGB channels
            int pixVal = 0;
            int pixels = 0;

            for (int blurRow = -BlurSize; blurRow <= BlurSize; ++blurRow)
            {
                for (int blurCol = -BlurSize; blurCol <= BlurSize; ++blurCol)
                {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // Ensure the neighboring pixel is within the image boundaries
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                    {
                        // Access the correct channel (R, G, or B)
                        pixVal += pin[(curRow * width + curCol) * 3 + channel];  
                        pixels++;
                    }
                }
            }

            // Write the averaged pixel value to the output for the current channel
            pout[(row * width + col) * 3 + channel] = (unsigned char)(pixVal / pixels);
        }
    }
}

int main()
{
    // Load the image using OpenCV
    Mat image = imread("./image.jpeg");
    if (image.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // Calculate the size of the image in bytes (3 channels)
    int img_size = width * height * 3 * sizeof(unsigned char);

    // Host input and output pointers
    unsigned char *h_pin = image.data;
    unsigned char *h_pout = (unsigned char *)malloc(img_size);

    // Device input and output pointers
    unsigned char *d_pin, *d_pout;
    cudaMalloc((void **)&d_pin, img_size);
    cudaMalloc((void **)&d_pout, img_size);

    // Copy the input image to the GPU
    cudaMemcpy(d_pin, h_pin, img_size, cudaMemcpyHostToDevice);

    // Configure the kernel launch with 16x16 threads per block
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid((width + threadPerBlock.x - 1) / threadPerBlock.x,
                      (height + threadPerBlock.y - 1) / threadPerBlock.y);

    // Launch the kernel
    colorBlur<<<blockPerGrid, threadPerBlock>>>(d_pin, d_pout, width, height);

    // Copy the result back to the CPU
    cudaMemcpy(h_pout, d_pout, img_size, cudaMemcpyDeviceToHost);

    // Create the output image
    Mat blurImg(height, width, CV_8UC3, h_pout);

    // Save the result
    imwrite("blurred_image.png", blurImg);

    // Free device and host memory
    cudaFree(d_pin);
    cudaFree(d_pout);
    free(h_pout);

    return 0;
}