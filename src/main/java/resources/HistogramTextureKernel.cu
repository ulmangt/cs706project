// a reference to a 2D texture where each texture element contains a 1D float value
// cudaReadModeElementType specifies that the returned data value should not be normalized
texture<float,  2, cudaReadModeElementType> texture_float_2D;

extern "C"
__global__ void test_float_2D(float *output, float posX, float posY)
{
    float result = tex2D(texture_float_2D, posX, posY);
    output[0] = result;
}
