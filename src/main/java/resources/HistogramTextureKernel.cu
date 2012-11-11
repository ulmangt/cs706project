// a reference to a 2D texture where each texture element contains a 1D float value
// cudaReadModeElementType specifies that the returned data value should not be normalized
texture<float,  2, cudaReadModeElementType> texture_float_2D;

extern "C"
__global__ void test_float_2D( int *bins, int nbins,
                               float minX, float stepX,
                               float minY, float stepY,
                               float minZ, float maxZ )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float result = tex2D(texture_float_2D, posX, posY);
    bins[0] = (int) result;
}
