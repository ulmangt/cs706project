#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GL/glut.h>

int main( int argc, char** argv )
{
    printf("CUDA Histogram Calculator\n");

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );

    glEnable( GL_TEXTURE_2D );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );
}
