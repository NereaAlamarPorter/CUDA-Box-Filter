/* Code completed by Antonio Marco Rodrigo Jimenez */

//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

//Optimizado para imagenes con relacion de aspecto 5:4
// x*y = N*32
// x/y = 5/4
// N=10 -> x*y = 320
#define BLOCK_SIZE_X 32 
#define BLOCK_SIZE_Y 16


#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "device_launch_parameters.h"

//////////////////////////////////////////////////////////////////////////
//					Usamos memoria de constantes						//
//////////////////////////////////////////////////////////////////////////
//Subimos las dimensiones (cuadradas) del filtro a memoria de constantes. Cambiar aquí cuando se cambie la dimensión del filtro a utilizar
__constant__ const int constantKernelWidth = 5; //MODICAR SEGUN EL FILTRO
//Queremos subir el propio filtro a memoria de constantes, lo cual haremos más adelante con cudaMemcpyToSymbol
__constant__ float constantFilter[constantKernelWidth * constantKernelWidth];
//////////////////////////////////////////////////////////////////////////

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

__global__
void box_filter(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	/*const float* const filter, */const int filterWidth) //Descomentar este parametro del metodo si se quiere usar sin memoria de constantes
{
	// TODO: 
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
	//
	// if ( absolute_image_position_x >= numCols ||
	//      absolute_image_position_y >= numRows )
	// {
	//     return;
	// }
	// NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
	// los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//Comprobamos que no nos estamos saliendo de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float result = 0.0f;
	for (int y = 0; y < filterWidth; y++)
	{
		for (int x = 0; x < filterWidth; x++)
		{
			//Con este doble for realizamos la convolución. Guardamos aqui el valor de fila
			int row = (int)(thread_2D_pos.y + (y - filterWidth / 2));
			//Nos aseguramos que el valor calculado no se sale de los limites
			if (row < 0)
				row = 0;
			if (row > numRows - 1)
				row = numRows - 1;

			//Guardamos aqui el valor de columna
			int column = (int)(thread_2D_pos.x + (x - filterWidth / 2));
			//Nos aseguramos que el valor calculado no se sale de los limites
			if (column < 0)
				column = 0;
			if (column > numCols - 1)
				column = numCols - 1;

			//Devolvemos el valor de la multiplicacion final de la convolucion:

			//Comentar si se quiere usar sin memoria de constantes
			result += (float)constantFilter[y*filterWidth + x] * (float)(inputChannel[row*numCols + column]);

			//Descomentar si se quiere usar sin memoria de constantes
		    //result += (float)filter	   [y*filterWidth + x] * (float)(inputChannel[row*numCols + column]);
		}
	}
	//Nos aseguramos de que el color final se encuentra entre 0 y 255
	if (result < 0.0f)
		result = 0.0f;
	if (result > 255.0f)
		result = 255.0f;
	outputChannel[thread_1D_pos] = result;

	//Descomentar si se quiere devolver la misma imagen de entrada (for testing)
	//outputChannel[thread_1D_pos] = inputChannel[thread_1D_pos];
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	// TODO: 
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
	//
	// if ( absolute_image_position_x >= numCols ||
	//      absolute_image_position_y >= numRows )
	// {
	//     return;
	// }

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Dividimos la imagen de entrada en sus 3 canales de color RGB
	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
//Descomentar si se quiere usar sin memoria de constantes, y comentamos el filtro constant del define del principio
//float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
	const float* const h_filter, const size_t filterWidth)
{
	//allocate memory for the three different channels
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

	//TODO: 
	//Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
	//Descomentar si se quiere usar sin memoria de constantes
	//checkCudaErrors(cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float)));
	
	//Copiar el filtro de la CPU host (h_filter) a memoria global de la GPU device (d_filter)
	//Descomentar si se quiere usar sin memoria de constantes
	//checkCudaErrors(cudaMemcpy(d_filter, h_filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
}

void create_filter(float **h_filter, int *filterWidth) {

	const int KernelWidth = constantKernelWidth; //OJO CON EL TAMAÑO DEL FILTRO (Introducir a mano el numero en caso de no usar memoria de constantes//	
	*filterWidth = constantKernelWidth;

	//create and fill the filter we will convolve with
	*h_filter = new float[KernelWidth * KernelWidth];

	/*
	//Filtro gaussiano: blur
	const float KernelSigma = 2.;

	float filterSum = 0.f; //for normalization

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
	  for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
		float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
		(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
		filterSum += filterValue;
	  }
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
	  for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
		(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
	  }
	}
	*/
	
	//Laplaciano 5x5
	(*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
	(*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
	(*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
	(*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
	(*h_filter)[20] = 1.;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;

	//TODO: crear los filtros segun necesidad
	//NOTA: cuidado al establecer el tamaño del filtro a utilizar

	//Creamos diversos filtros para distintos propositos, comentar y descomentar el que se quiera utilizar

	//Aumentar nitidez 3x3
	/*
	(*h_filter)[0] = 0;   (*h_filter)[1] = -0.25;    (*h_filter)[2] = 0;
	(*h_filter)[3] = -0.25;  (*h_filter)[4] = 2.;  (*h_filter)[5] = -0.25;
	(*h_filter)[6] = 0; (*h_filter)[7] = -0.25; (*h_filter)[8] = 0;
	*/

	/*
	//Detección de línea horizontal - Line Detection Horizontal
	(*h_filter)[0] = -1.;   (*h_filter)[1] = -1.;    (*h_filter)[2] = -1.;
	(*h_filter)[3] = 2.;  (*h_filter)[4] = 2.;  (*h_filter)[5] = 2.;
	(*h_filter)[6] = -1.; (*h_filter)[7] = -1.; (*h_filter)[8] = -1.;
	*/

	/*
	//Suavizado - Smooth Arithmetic Mean
	(*h_filter)[0] = 0.111;   (*h_filter)[1] = 0.111;    (*h_filter)[2] = 0.111;
	(*h_filter)[3] = 0.111;  (*h_filter)[4] = 0.111;  (*h_filter)[5] = 0.111;
	(*h_filter)[6] = 0.111; (*h_filter)[7] = 0.111; (*h_filter)[8] = 0.111;
	*/

	//Subimos el filtro h_filter a memoria de constantes, como definimos al principio en constantFilter
	//Comentar en caso de que no queramos usar memoria de constantes
	cudaMemcpyToSymbol(constantFilter, *h_filter, sizeof(float) * KernelWidth * KernelWidth);
}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
	unsigned char *d_redFiltered,
	unsigned char *d_greenFiltered,
	unsigned char *d_blueFiltered,
	const int filterWidth)
{
	//TODO: Calcular tamaños de bloque
	const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	//Redondeamos hacia arriba con ceil en caso de quedarnos cortos
	const dim3 gridSize(ceil(1.0f*numCols / blockSize.x), ceil(1.0f*numRows / blockSize.y));

	//TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
	separateChannels << <gridSize, blockSize, 0/*Shared Memory Bytes = 0*/ >> > (d_inputImageRGBA, 
		numRows, 
		numCols, 
		d_red, 
		d_green, 
		d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//TODO: Ejecutar convolución. Una por canal
	//Descomentar el parametro d_filter si no se quiere usar memoria de constantes
	box_filter << <gridSize, blockSize, 0 >> > (d_red, d_redFiltered, numRows, numCols, /*d_filter,*/ filterWidth);
	box_filter << <gridSize, blockSize, 0 >> > (d_green, d_greenFiltered, numRows, numCols, /*d_filter,*/ filterWidth);
	box_filter << <gridSize, blockSize, 0 >> > (d_blue, d_blueFiltered, numRows, numCols, /*d_filter,*/ filterWidth);
	//Ponemos 0 en el kernel porque no usamos memoria compratida
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Recombining the results. 
	recombineChannels << <gridSize, blockSize >> > (d_redFiltered,
		d_greenFiltered,
		d_blueFiltered,
		d_outputImageRGBA,
		numRows,
		numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));

	//Descomentar si no se quiere usar memoria de constantes
	//checkCudaErrors(cudaFree(d_filter));
}