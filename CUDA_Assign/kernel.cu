#include "main.h"

using namespace cv;
using namespace std;

// Global vars - I know these are dangerous, but this is needing to be hacked together as for reasons explained in video/documentation
cv::Mat frameIn;
cv::Mat frameOut;
cv::Mat frameRGBA;

uchar4 *d_frameIn;
uchar4 *d_frameOut;

GpuTimer g_timer;
GpuTimer c_timer;

// Timer var references
float cpuTime;
float gpuTime;

// Filter vars, these will be used to apply a weighting matrix to the image
float *_h_filter;
int   filterWidth;
int i = 0;

// Channels and frame initializers
uchar4 *h_inputFrame;
uchar4 *d_inputFrame;
uchar4 *h_outputFrame;
uchar4 *d_outputFrame;
unsigned char *d_redBlurred;
unsigned char *d_greenBlurred; 
unsigned char *d_blueBlurred;

// Helper methods for frame
size_t numRows() { return frameIn.rows; }
size_t numCols() { return frameIn.cols; }
size_t numPixels() { return (numRows() * numCols()); }

// Main entry point for the program, we fire off our Camera instanciation as well
// as the call to the GPU method here
int main() 
{
	// Local variables
	int cpu_frames;  
	int gpu_frames;
	int total_frames;

	// Open the first camera we find (at index 0)
	VideoCapture camera(0);

	// Check we have a valid camera, if not break out the program
	if(!camera.isOpened()) return -1;
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Ask user how many frames they wish to grab
	cout << "How many frames do you wish to process? ";
	scanf("%i", &cpu_frames);
	gpu_frames = cpu_frames;
	total_frames = cpu_frames;


	namedWindow("Source", WINDOW_AUTOSIZE);
	namedWindow("Dest", WINDOW_AUTOSIZE);

	cout << "Calculating gaussian kernel on GPU";

	// Keep processing frames - Do CPU First
	while(gpu_frames > 0)
	{
		//cout << gpu_frames << "\n";
		camera >> frameIn;

		// I/O Pointers
		beginStream(&h_inputFrame, &h_outputFrame, &d_inputFrame, &d_outputFrame, &d_redBlurred, &d_greenBlurred, &d_blueBlurred, &_h_filter, &filterWidth, frameIn);
	
		// Show the source image
		imshow("Source", frameIn);

		g_timer.Start();
		// Allocate mem to GPU
		allocateMemoryAndCopyToGPU(numRows(), numCols(), _h_filter, filterWidth);

		// Apply the gaussian kernel filter and then free any memory ready for the next iteration
		gaussian_gpu(h_inputFrame, d_inputFrame, d_outputFrame, numRows(), numCols(), d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
		
		// Output the blurred image
		cudaMemcpy(h_outputFrame, d_frameOut, sizeof(uchar4) * numPixels(), cudaMemcpyDeviceToHost);
		g_timer.Stop();
		cudaDeviceSynchronize();
		gpuTime += g_timer.Elapsed();
		cout << "Time for this kernel " << g_timer.Elapsed() << "\n";

		Mat outputFrame(Size(numCols(), numRows()), CV_8UC1, h_outputFrame, Mat::AUTO_STEP);

		clean_mem();

		imshow("Dest", outputFrame);

		// 1ms delay to prevent system from being interrupted whilst drawing the new frame
		waitKey(1);
		gpu_frames--;
	}
	cout << "Computed " << total_frames << " frames in " << g_timer.Elapsed() << " msecs.\n";
	
	
	cout << "Calculating gaussian kernel on CPU";
	// Keep processing frames - Do CPU now
	while(cpu_frames > 0)
	{
		camera >> frameIn;

		imshow("Source", frameIn);

		// I/O Pointers
		c_timer.Start();
		beginStream(&h_inputFrame, &h_outputFrame, &d_inputFrame, &d_outputFrame, &d_redBlurred, &d_greenBlurred, &d_blueBlurred, &_h_filter, &filterWidth, frameIn);

		gaussian_cpu(h_inputFrame, h_outputFrame, numRows(), numCols(), _h_filter, 9);
		c_timer.Stop();
		cudaDeviceSynchronize();
		cpuTime += c_timer.Elapsed();
		cout << "Time for this kernel " << c_timer.Elapsed() << "\n";

		// Create the output frame and alloc mem
		Mat outputFrame(Size(numCols(), numRows()), CV_8UC1, h_outputFrame, Mat::AUTO_STEP);

		imshow("Dest", outputFrame);

		// Clean up any memory we've allocated
		clean_mem();

		// 1ms delay to prevent system from being interrupted whilst drawing the new frame
		waitKey(1);
		cpu_frames--;
	}
	cout << "Computed " << total_frames << " frames in " << c_timer.Elapsed() << " msecs.\n";

	cout << "The GPU is " << (cpuTime / gpuTime) << " times faster than the CPU.\n";
	cout << "\n\nResult set processed, terminating...";
	waitKey(50000);
}

// Stream the image
void beginStream(
					uchar4 **h_inputFrame,						// Pointer to host input frame
					uchar4 **h_outputFrame,						// Pointer to host output frame
					uchar4 **d_inputFrame,						// Pointer to device input frame
					uchar4 **d_outputFrame,						// Pointer to device output frame
					unsigned char **d_redBlurred,				// Device red channel blur
					unsigned char **d_greenBlurred,				// Device green channel blur 
					unsigned char **d_blueBlurred,				// Device blue channel blur
					float **h_filter, int *filterWidth,			// The width we want our filter to be
					cv::Mat src									// The source frame we just captured
				)
{
		// Check we are okay
		cudaFree(0);

		// Move source data into the input frame, ensuring RGBA format
		cv::cvtColor(src, frameIn, CV_BGR2RGBA); 

		// Allocate memory for the output frame
		frameOut.create(frameIn.rows, frameIn.cols, CV_8UC4);

		// Allocate host variables, casting the frameIn and frameOut vars to uchar4 elements, these will
		// later be processed by the kernel
		*h_inputFrame = (uchar4 *)frameIn.ptr<unsigned char>(0);
		*h_outputFrame = (uchar4 *)frameOut.ptr<unsigned char>(0);

		// The image has been created, now we can find out how many pixels we are going to be working with
		const size_t numPixels = numRows() * numCols();

		// Allocate memory on the device for I/O
		cudaMalloc(d_inputFrame, sizeof(uchar4) * numPixels);
		cudaMalloc(d_outputFrame, sizeof(uchar4) * numPixels);
		cudaMemset(*d_outputFrame, 0, numPixels * sizeof(uchar4));

		// Copy the input frame array to the CPU for processing
		cudaMemcpy(*d_inputFrame, *h_inputFrame, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

		// Set the global references of the current working image
		d_frameIn  = *d_inputFrame;
		d_frameOut = *d_outputFrame;

		// Create blur kernel
		const int stencil = 9;
		const float sigma = 2.f;

		*filterWidth = stencil;

		// Fill the filter for convulution
		*h_filter = new float[stencil * stencil];
		_h_filter = *h_filter;

		float filterSum = 0.f;

		// Create the weightings for the filter
		for (int r = -stencil/2; r <= stencil/2; ++r) {
			for (int c = -stencil/2; c <= stencil/2; ++c) 
			{
				float filterValue = expf( -(float)(c * c + r * r) / (2.f * sigma * sigma));
				(*h_filter)[(r + stencil/2) * stencil + c + stencil/2] = filterValue;
				filterSum += filterValue;
			}
	   }
		
		float normalise = 1.f / filterSum;

		for (int r = -stencil/2; r <= stencil/2; ++r) {
			for (int c = -stencil/2; c <= stencil/2; ++c) {
				(*h_filter)[(r + stencil/2) * stencil + c + stencil/2] *= normalise;
		    }
		}

		// Alloacate memory for the channels
		cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels);
		cudaMalloc(d_greenBlurred, sizeof(unsigned char) * numPixels);
		cudaMalloc(d_blueBlurred, sizeof(unsigned char) * numPixels);
		cudaMemset(*d_redBlurred, 0, sizeof(unsigned char) * numPixels);
		cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
		cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char) * numPixels);

}

// Free any memory
void clean_mem(void)
{
	cudaFree(d_frameIn);
	cudaFree(d_frameOut);
	delete[] _h_filter;
}