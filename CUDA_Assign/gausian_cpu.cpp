#include <algorithm>
#include <cassert>
// for uchar4 struct
#include <cuda_runtime.h>

// Convolution kernel - this manipulates the given channel and writes out a new blurred channel.
void convoluteChannel_cpu(
						const unsigned char* const channel,			// Input channel
						unsigned char* const channelBlurred,		// Output channel
						const size_t numRows, const size_t numCols,	// Channel width/height (rows, cols)
						const float *filter,						// The weight of sigma, to convulge
						const int filterWidth						// This is normally a sample of 9
					 )
{
	// Handle an uneven width
	assert(filterWidth %2 == 1);

	// Loop through the images given R, G or B channel
	for(int rows = 0; rows < (int)numRows; rows++)
	{
		for(int cols = 0; cols < (int)numCols; cols++)
		{
			// Declare new pixel colour value
			float newColor = 0.f;

			// Loop for every row along the stencil size (3x3 matrix)
			for(int filter_x = -filterWidth/2; filter_x <= filterWidth/2; filter_x++)
			{
				// Loop for every col along the stencil size (3x3 matrix)
				for(int filter_y = -filterWidth/2; filter_y <= filterWidth/2; filter_y++)
				{
					// Clamp to the boundary of the image to ensure we don't access a null index.
					int image_x = __min(__max(rows + filter_x, 0), static_cast<int>(numRows -1));
					int image_y = __min(__max(cols + filter_y, 0), static_cast<int>(numCols -1));

					// Assign the new pixel value to the current pixel, numCols and numRows are both 3, so we only 
					// need to use one to find the current pixel index (similar to how we find the thread in a block)
					float pixel = static_cast<float>(channel[image_x * numCols + image_y]);

					// Sigma is the new weight to apply to the image, we perform the equation to get a radnom weighting,
					// if we don't do this the image will become choppy.
					float sigma = filter[(filter_x + filterWidth / 2) * filterWidth + filter_y + filterWidth/2];
					//float sigma = 1 / 81.f;

					// Set the new pixel value
					newColor += pixel * sigma;
				}
			}

			// Set the value of the next pixel at the current image index with the newly declared color
			channelBlurred[rows * numCols + cols] = newColor;
		}
	}
}

void gaussian_cpu(
					const uchar4* const rgbaImage,	     // Our input image from the camera
					uchar4* const outputImage,		     // The image we are writing back for display
					size_t numRows, size_t numCols,      // Width and Height of the input image (rows/cols)
					const float* const filter,	         // The value of sigma
					const int filterWidth			     // The size of the stencil (3x3) 9
				 )
{
	// Build an array to hold each channel for the given image
	unsigned char *r_c = new unsigned char[numRows * numCols];
	unsigned char *g_c = new unsigned char[numRows * numCols];
	unsigned char *b_c = new unsigned char[numRows * numCols];

	// Build arrays for each of the output (blurred) channels
	unsigned char *r_bc = new unsigned char[numRows * numCols]; 
	unsigned char *g_bc = new unsigned char[numRows * numCols]; 
	unsigned char *b_bc = new unsigned char[numRows * numCols]; 

	// Separate the image into R,G,B channels
	for(size_t i = 0; i < numRows * numCols; i++) 
	{
		uchar4 rgba = rgbaImage[i];
		r_c[i] = rgba.x;
		g_c[i] = rgba.y;
		b_c[i] = rgba.z;
	}

	// Convolute each of the channels using our array
	convoluteChannel_cpu(r_c, r_bc, numRows, numCols, filter, filterWidth);
	convoluteChannel_cpu(g_c, g_bc, numRows, numCols, filter, filterWidth);
	convoluteChannel_cpu(b_c, b_bc, numRows, numCols, filter, filterWidth);

	// Recombine the channels to build the output image - 255 for alpha as we want 0 transparency
	for(size_t i = 0; i < numRows * numCols; i++) 
	{
		uchar4 rgba = make_uchar4(r_bc[i], g_bc[i], b_bc[i], 255);
		outputImage[i] = rgba;
	}

	// Free the memory, as we no longer need it - throwing strange error for now
	/*
	delete[] r_c; 
	delete[] g_c; 
	delete[] b_c;
	delete[] r_bc; 
	delete[] g_bc; 
	delete[] b_bc;
	*/
}