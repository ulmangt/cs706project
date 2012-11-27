#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <GL/glut.h>
#include <cuda_gl_interop.h>

int textureWidth = 4000;
int textureHeight = 4000;

GLuint* textureHandles;
cudaGraphicsResource_t* graphicsResource;

bool initialized = false;

float* imageData;

void initImageData( float* data )
{
    int w,h;

    double pi = atan(1) * 4;

    for ( w = 0; w < textureWidth; w++ )
    {
        for ( h = 0; h < textureHeight; h++ )
        {
            double x = w / ( double ) textureWidth;
            double y = h / ( double ) textureHeight;

            double r = rand() / (double) RAND_MAX;
            data[h+w*textureHeight] = ( y * y + sin( 2 * pi * x * x ) + r ) * 100;
        }
    }
}

void init(void)
{
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

    graphicsResource = (cudaGraphicsResource_t*) malloc( sizeof( cudaGraphicsResource_t ) );
    cudaGraphicsGLRegisterImage( graphicsResource, *textureHandles, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly );
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
