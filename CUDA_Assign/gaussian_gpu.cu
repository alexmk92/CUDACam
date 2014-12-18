#include <stdio.h>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

__global__ void gaussian_kernel(      
                                    const unsigned char* const inputChannel,   // Return the blurred channel (R, G or B)
                                    unsigned char*       const outputChannel,  // Modifying either R, G or B channel from orig
                                    int numRows, 
                                    int numCols,                 
                                    const float* const filter,                 // The weight of sigma 
                                    const int filterWidth                      // how many pixels our stencil is sampling -- normally 9
                              )
{
    // Create a vector to hold the location of the thread (row and col)
    const int2 pixel_2D = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                     blockIdx.y * blockDim.x + threadIdx.y );

    const int  pixel_1D = pixel_2D.y * numCols + pixel_2D.x;

    // Are we outside the bounds of the image? If so quit
    if( pixel_2D.x >= numCols || pixel_2D.y >= numRows ) { return; }

    // Local variables
    float color = 0.0f;

    // Loop through the rows and columns, getting the average - do rows before cols to void cache hits
    for ( int img_y = 0; img_y < filterWidth; img_y++ )  
    {
        // Loop through the columns
        for ( int img_x = 0; img_x < filterWidth; img_x++ )
        {
            // Clamp the filter to the image border - ensuring we don't go out of bounds
            int clamp_x = __min(__max(pixel_2D.x + img_x - filterWidth / 2, 0), numCols - 1);
            int clamp_y = __min(__max(pixel_2D.y + img_y - filterWidth / 2, 0), numRows - 1);

            // Calculate the average value of the pixel
            const float avg = filter[img_y * filterWidth + img_x];
            
            // Calculate the new blur value using the algorithm outlined in documentation
            color += avg*static_cast<float>(inputChannel[clamp_y * numCols + clamp_x]);
        }
    }

    // Write the new pixel value to the output channel
    outputChannel[pixel_1D] = color;
}

//This kernel takes in an image represented as a uchar4 and creates three images, one for each channel
__global__ void separateChannels(     const uchar4* const inputImageRGBA,
                                      int numRows,
                                      int numCols,
                                      unsigned char* const redChannel,
                                      unsigned char* const greenChannel,
                                      unsigned char* const blueChannel
                                )
{
    // Create a vector holding x and y position of the thread, named pixel as 
    const int2 pixel_2D = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                     blockIdx.y * blockDim.x + threadIdx.y );

    const int  pixel_1D = pixel_2D.y * numCols + pixel_2D.x;

    // Are we outside the bounds of the image? If so quit
    if(pixel_2D.x >= numCols || pixel_2D.y >= numRows) { return; }

    // Separate the channels so we have one image per channel
    redChannel[pixel_1D]   = inputImageRGBA[pixel_1D].x;
    greenChannel[pixel_1D] = inputImageRGBA[pixel_1D].y;
    blueChannel[pixel_1D]  = inputImageRGBA[pixel_1D].z; 
}

//This kernel takes in three color channels and recomines them to form a single image
__global__ void recombineChannels(    
									  const unsigned char* const redChannel,
                                      const unsigned char* const greenChannel,
                                      const unsigned char* const blueChannel,
                                      uchar4* const outputImageRGBA,
                                      int numRows,
                                      int numCols
                                 )
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image - if we do, then return
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
      return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);

  //Allocate memory for the filter on the GPU
  const float memSize = sizeof(float) * filterWidth * filterWidth;
  cudaMalloc(&d_filter, memSize);

  //Copy the filter on the host to GPU memory
  cudaMemcpy(d_filter, h_filter, memSize, cudaMemcpyHostToDevice);

}

//Free all allocated memory for this kernel
void cleanDevice() {
  cudaFree(d_red);
  cudaFree(d_green);
  cudaFree(d_blue);
  cudaFree(d_filter);
}

void gaussian_gpu(
                            const uchar4 * const h_inputImageRGBA, 
                            uchar4 * const d_inputImageRGBA,
                            uchar4* const d_outputImageRGBA, 
                            const size_t numRows, 
                            const size_t numCols,
                            unsigned char *d_redBlurred, 
                            unsigned char *d_greenBlurred, 
                            unsigned char *d_blueBlurred,
                            const int filterWidth
                  )
{
  //Set reasonable block size to process the image (1024 threads per block here)
  const dim3 blockSize(8, 8);
  //Compute correct grid size from the image size and and block size.
  const dim3 gridSize(numCols/16, numRows/16);

  //Launch a kernel to separate the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(
                                              d_inputImageRGBA,
                                              numRows,
                                              numCols,
                                              d_red,
                                              d_green,
                                              d_blue
                                           );

  // Wait for device computation to finish and check for any errors
  cudaDeviceSynchronize(); cudaGetLastError();

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); cudaGetLastError();

  // Call the convolution kernel 3 times, for R, G and B
  gaussian_kernel<<<gridSize, blockSize>>>(
                                            d_red,
                                            d_redBlurred,
                                            numRows,
                                            numCols,
                                            d_filter, 
                                            filterWidth
                                        );

  // synchronize before calling next kernel
  cudaDeviceSynchronize(); cudaGetLastError();

  gaussian_kernel<<<gridSize, blockSize>>>(
                                            d_green,
                                            d_greenBlurred,
                                            numRows,
                                            numCols,
                                            d_filter, 
                                            filterWidth
                                        );

  // synchronize before calling next kernel
  cudaDeviceSynchronize(); cudaGetLastError();

  gaussian_kernel<<<gridSize, blockSize>>>(
                                            d_blue,
                                            d_blueBlurred,
                                            numRows,
                                            numCols,
                                            d_filter, 
                                            filterWidth
                                        );

  // synchronize before calling next kernel
  cudaDeviceSynchronize(); cudaGetLastError();


  // Recombine channels
  recombineChannels<<<gridSize, blockSize>>>(
                                                d_redBlurred,
                                                d_greenBlurred,
                                                d_blueBlurred,
                                                d_outputImageRGBA,
                                                numRows,
                                                numCols
                                            );

  cudaDeviceSynchronize(); cudaGetLastError();
  cleanDevice();
}


