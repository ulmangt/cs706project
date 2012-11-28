#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h> 
#include <helper_cuda.h> 

int width = 1000;
int height = 1000;
int numBins = 10;

GLuint* textureHandles;
cudaGraphicsResource_t* graphicsResource;
cudaArray* array;

bool initialized = false;

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

extern "C"
__global__ void test_float_2D( int *bins, int nbins,
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
    
        float stepZ = ( maxZ - minZ ) / nbins;
        float fbinIndex = floor( ( result - minZ ) / stepZ );
        int binIndex = (int) clamp( fbinIndex, 0, nbins-1 );
    
        // atomically add one to the bin corresponding to the texture value
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
            data[h+w*height] = 0;//( y * y + sin( 2 * pi * x * x ) + r ) * 100;
        }
    }
}

void init(void)
{
    // size of texture data
    unsigned int size = width * height * sizeof(float);

    // allocate space for texture data and initialize with interesting function
    imageData = (float*) malloc( size );
    initImageData( imageData );

    // set up CUDA texture description (32 bit float)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // create a CUDA array for accessing texture data
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));

    // copy image data from the host into the CUDA array
    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      imageData,
                                      size,
                                      cudaMemcpyHostToDevice));

    // set texture access modes for the CUDA texture variable
    // (clamp access for texture coordinates outside 0 to 1)
    texture_float_2D.addressMode[0] = cudaAddressModeClamp;
    texture_float_2D.addressMode[1] = cudaAddressModeClamp;
    texture_float_2D.filterMode = cudaFilterModeLinear;
    texture_float_2D.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(texture_float_2D, cuArray, channelDesc));

    // Allocate space for histogram bin results
    int sizeBins = sizeof( int ) * numBins;
    hBins = (int*) malloc( sizeBins );
    cudaMalloc( &dBins, sizeBins );
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
    test_float_2D<<<dimGrid, dimBlock, 0>>>( dBins, numBins, 0, stepX, 0, stepY, minZ, maxZ );

    // copy results back to host
    cudaMemcpy( hBins, dBins, sizeBins, cudaMemcpyDeviceToHost );

    // print results
    int i;
    for ( i = 0 ; i < numBins ; i++ )
    {
        printf( "%d\n", hBins[i] );
    }
}

//Drawing funciton
void draw(void)
{
  if ( !initialized )
  {
    init( );
    initialized = true;
  }

  //Background color
  glClearColor( 0,1,0,1 );
  glClear( GL_COLOR_BUFFER_BIT );


  //Draw order
  glFlush();
}

//Main program
int main(int argc, char **argv)
{
  printf("CUDA Histogram Calculator Starting...\n");

  //cutilDeviceInit(argc, argv);
  glutInit(&argc, argv);

  //Simple buffer
  glutInitDisplayMode( GLUT_SINGLE | GLUT_RGB );
  glutInitWindowPosition(50,25);
  glutInitWindowSize(500,250);
  glutCreateWindow("Green window");

  //Call to the drawing function
  glutDisplayFunc(draw);
  glutMainLoop();

  return 0;
}
