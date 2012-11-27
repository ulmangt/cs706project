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

int width = 4000;
int height = 4000;

GLuint* textureHandles;
cudaGraphicsResource_t* graphicsResource;
cudaArray* array;

bool initialized = false;

float* imageData;


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

    // perform texture lookup
    float result = tex2D(texture_float_2D, x, y);
    
    float stepZ = ( maxZ - minZ ) / nbins;
    float fbinIndex = floor( ( result - minZ ) / stepZ );
    int binIndex = (int) clamp( fbinIndex, 0, nbins-1 );
    
    // atomically add one to the bin corresponding to the texture value
    atomicAdd( bins+binIndex, 1 );
}




void initImageData( float* data )
{
    int w,h;

    double pi = atan(1) * 4;

    for ( w = 0; w < width; w++ )
    {
        for ( h = 0; h < height; h++ )
        {
            double x = w / ( double ) width;
            double y = h / ( double ) height;

            double r = rand() / (double) RAND_MAX;
            data[h+w*height] = ( y * y + sin( 2 * pi * x * x ) + r ) * 100;
        }
    }
}

void init(void)
{
/*
    glEnable( GL_TEXTURE_2D );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    textureHandles = (GLuint*) malloc( sizeof( GLuint ) );

    glGenTextures( 1, textureHandles );

    glBindTexture( GL_TEXTURE_2D, *textureHandles );

    printf("Texture %d\n", *textureHandles);

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

    imageData = (float*) malloc( sizeof( float ) * textureWidth * textureHeight );
    initImageData( imageData );

    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, textureWidth, textureHeight, 0, GL_LUMINANCE, GL_FLOAT, imageData );

    //graphicsResource = (cudaGraphicsResource_t*) malloc( sizeof( cudaGraphicsResource_t ) );
    cudaGraphicsGLRegisterImage( graphicsResource, *textureHandles, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly );

    cudaGraphicsMapResources( 1, graphicsResource, 0 );

    cudaGraphicsSubResourceGetMappedArray( &array, *graphicsResource, 0, 0 );

    texture_float_2D.addressMode[0] = cudaAddressModeClamp;
    texture_float_2D.addressMode[1] = cudaAddressModeClamp;
    texture_float_2D.filterMode = cudaFilterModeLinear;
    texture_float_2D.normalized = true;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaBindTextureToArray( texture_float_2D, array, channelDesc );
*/

    unsigned int size = width * height * sizeof(float);

    imageData = (float*) malloc( size );
    initImageData( imageData );

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));

    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      imageData,
                                      size,
                                      cudaMemcpyHostToDevice));

    texture_float_2D.addressMode[0] = cudaAddressModeClamp;
    texture_float_2D.addressMode[1] = cudaAddressModeClamp;
    texture_float_2D.filterMode = cudaFilterModeLinear;
    texture_float_2D.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(texture_float_2D, cuArray, channelDesc));

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
