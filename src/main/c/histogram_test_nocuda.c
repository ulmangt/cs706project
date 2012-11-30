#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define numBins 10
#define width 1000
#define height 1000

inline float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
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

//Main program
int main(int argc, char **argv)
{
    // size of texture data
    unsigned int size = width * height * sizeof(float);

    // allocate space for texture data and initialize with interesting function
    float *imageData = (float*) malloc( size );
    initImageData( imageData );

    // allocate space for accumulated bin counts
    int bins[numBins];
    int i;
    for ( i = 0 ; i < numBins ; i++ )
    {
        bins[i] = 0;
    }

    float minZ = -50;
    float maxZ = 200;
    float stepZ = ( maxZ - minZ ) / numBins;

    clock_t start = clock();

    for ( i = 0 ; i < width * height ; i++ )
    {
        // retrieve a data value
        float data = imageData[i];

        // calculate histogram bin index for the data value
        float fbinIndex = floor( ( data - minZ ) / stepZ );
        int binIndex = (int) clamp( fbinIndex, 0, numBins-1 );

        // increment the histogram bin
        bins[binIndex] += 1;
    }

    // time the cpu step
    clock_t diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf( "%d millis\n", msec );

    // print results
    int sum = 0;
    for ( i = 0 ; i < numBins ; i++ )
    {
        sum += bins[i];
        printf( "%d\n", bins[i] );
    }

    printf( "sum %d\n", sum );

    free( imageData );

    return 0;
}
