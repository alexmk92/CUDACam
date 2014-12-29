#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "gpu_timer.h"
#include "gaussian_cpu.h"

// Method prototypes

// convolution method which alters the channel data for the host
void convoluteChannel_cpu(
						const unsigned char* const channel,			// Input channel
						unsigned char* const channelBlurred,		// Output channel
						const size_t numRows, const size_t numCols,	// Channel width/height (rows, cols)
						const float *filter,						// The weight of sigma, to convulge
						const int filterWidth						// This is normally a sample of 9
					 );

// Apply gaussian blur on the host
void gaussian_cpu(
					const uchar4* const rgbaImage,	     // Our input image from the camera
					uchar4* const outputImage,		     // The image we are writing back for display
					size_t numRows, size_t numCols,      // Width and Height of the input image (rows/cols)
					const float* const filter,	         // The value of sigma
					const int filterWidth			     // The size of the stencil (3x3) 9
				 );


// Creates a call to the gaussian_gpu method located in gaussian_gpu.cu
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
                        );

// Set the memory on the host and move to the GPU for processing
void allocateMemoryAndCopyToGPU(
									const size_t numRowsImage, 
									const size_t numColsImage,
									const float* const h_filter, 
									const size_t filterWidth
								);

// Initializes our stream object which sets up pointers to the device and host variables, this shall allow
// us to set allocate memory and pass objects to the CUDA func
void beginStream(
					uchar4 **h_inputFrame,						// Pointer to host input frame
					uchar4 **h_outputFrame,						// Pointer to host output frame
					uchar4 **d_inputFrame,						// Pointer to device input frame
					uchar4 **d_outputFrame,						// Pointer to device output frame
					unsigned char **d_redBlurred,				// Device red channel blur
					unsigned char **d_greenBlurred,				// Device green channel blur 
					unsigned char **d_blueBlurred,				// Device blue channel blur
					float **h_filter, int *filterWidth,			// The width we want our filter to be
					cv::Mat src,								// The source frame we just captured
					const bool runningGPU                       // Are we allocating memory on the device/host?
				);

// Clean up any resources on the host
void clean_mem(void);
