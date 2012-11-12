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
