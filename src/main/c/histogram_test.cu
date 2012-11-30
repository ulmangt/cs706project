#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

int width = 1000;
int height = 1000;
int numBins = 10;

cudaArray *cuArray;
float* imageData;
int* dBins;
int* hBins;

// a reference to a 2D texture where each texture element contains a 1D float value
// cudaReadModeElementType specifies that the returned data value should not be normalized
texture<float,  2, cudaReadModeElementType> texture_float_2D;

// clamp
inline __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// bins     global memory vector to be filled with bin counts
// nbins    size of bins vector
// minX     the minimum x texture coordinate
// stepX    step size in x in texture coordinates
// minY     the minimum y texture coordinate
// stepY    step size in y in texture coordinates
// minZ     data value of the left edge of the left-most bin
// maxZ     data value of the right edge of the right-most bin
extern "C" __global__ void calculateHistogram1( int *bins, int nbins,
                                                float minX, float stepX,
                                                float minY, float stepY,
                                                float minZ, float maxZ )
{
    // use block and thread ids to get texture coordinates for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // convert block/thread ids into texture coordinates
    float x = minX + stepX * i;
    float y = minY + stepY * j;

    // don't over count if texture coordinates are out of bounds
    if ( x < 1.0 && y < 1.0 )
    {
        // perform texture lookup
        float result = tex2D(texture_float_2D, x, y);
    
        // calculate bin index
        float stepZ = ( maxZ - minZ ) / nbins;
        float fbinIndex = floor( ( result - minZ ) / stepZ );
        int binIndex = (int) clamp( fbinIndex, 0, nbins-1 );
    
        // atomically add one to the bin corresponding to the data value
        atomicAdd( bins+binIndex, 1 );
    }
}

void initImageData( float* data )
{
    int w,h;

    float pi = atan(1) * 4;

    for ( w = 0; w < width; w++ )
    {
        for ( h = 0; h < height; h++ )
        {
            float x = w / ( float ) width;
            float y = h / ( float ) height;

            float r = rand() / (float) RAND_MAX;
            data[h+w*height] = ( y * y + sin( 2 * pi * x * x ) + r ) * 100;
        }
    }
}

void init(int argc, char **argv)
{
    // size of texture data
    unsigned int size = width * height * sizeof(float);

    // allocate space for texture data and initialize with interesting function
    imageData = (float*) malloc( size );
    initImageData( imageData );

    // set up CUDA texture description (32 bit float)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // create a CUDA array for accessing texture data
    cudaMallocArray(&cuArray,&channelDesc,width,height);

    // copy image data from the host into the CUDA array
    cudaMemcpyToArray(cuArray, 0, 0, imageData, size, cudaMemcpyHostToDevice);

    // set texture access modes for the CUDA texture variable
    // (clamp access for texture coordinates outside 0 to 1)
    texture_float_2D.addressMode[0] = cudaAddressModeClamp;
    texture_float_2D.addressMode[1] = cudaAddressModeClamp;
    texture_float_2D.filterMode = cudaFilterModeLinear;
    texture_float_2D.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTextureToArray(texture_float_2D, cuArray, channelDesc);

    // Allocate space for histogram bin results
    int sizeBins = sizeof( int ) * numBins;
    hBins = (int*) malloc( sizeBins );
    cudaMalloc( &dBins, sizeBins );
}

void calculateHistogram(void)
{
    int sizeBins = sizeof( int ) * numBins;

    cudaMemset( dBins, 0, sizeBins );

    // calculate block and grid dimensions
    dim3 dimBlock( 16, 16, 1);
    int gridX = ceil( width / (float) dimBlock.x );
    int gridY = ceil( height / (float) dimBlock.y );
    dim3 dimGrid( gridX, gridY, 1);

    // run the kernel over the whole texture
    float stepX = 1.0 / width;
    float stepY = 1.0 / height;
    float minZ = -50.0;
    float maxZ = 200.0;
    calculateHistogram1<<<dimGrid, dimBlock, 0>>>( dBins, numBins, 0, stepX, 0, stepY, minZ, maxZ );

    // copy results back to host
    cudaMemcpy( hBins, dBins, sizeBins, cudaMemcpyDeviceToHost );

    // print results
    int sum = 0;
    int i;
    for ( i = 0 ; i < numBins ; i++ )
    {
        sum += hBins[i];
        printf( "%d\n", hBins[i] );
    }

    printf( "sum %d\n", sum );
}

//Main program
int main(int argc, char **argv)
{
  printf("CUDA Histogram Calculator\n");

  init( argc, argv );

  calculateHistogram( );

  free( hBins );
  free( imageData );

  cudaFree(dBins);
  cudaFreeArray(cuArray);

  return 0;
}
