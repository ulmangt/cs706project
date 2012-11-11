package edu.gmu.ulman.histogram;

import static jcuda.driver.CUaddress_mode.*;
import static jcuda.driver.CUarray_format.*;
import static jcuda.driver.CUfilter_mode.*;
import static jcuda.driver.JCudaDriver.*;

import java.io.IOException;

import javax.media.opengl.GL;
import javax.media.opengl.GLContext;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUarray;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaGraphicsRegisterFlags;
import edu.gmu.ulman.histogram.util.PtxUtils;

public class JCudaHistogramCalculator
{
    private double minValue;
    private double maxValue;
    private int numBins;
    private int[] hHistogramBins;
    private CUdeviceptr dHistogramBins;

    private CUdevice device;
    private CUcontext context;
    private CUmodule module;
    private CUgraphicsResource gfxResource;
    private CUarray hostArray;
    private CUtexref textureReference;
    private CUfunction functionTest;
    
    public JCudaHistogramCalculator( int numBins, double minValue, double maxValue )
    {
        this.numBins = numBins;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    /**
     * Initialize the CUDA driver.
     * 
     * @param texHandle OpenGL texture handle
     * @throws IOException 
     */
    public void initialize( int texHandle ) throws IOException
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled( true );

        // initialize the CUDA driver API, passing in no argument flags
        cuInit( 0 );

        // get a handle to the first GPU device
        // (arguments of >1 allow for the possibility of multi-gpu systems)
        device = new CUdevice( );
        cuDeviceGet( device, 0 );

        // create a CUDA context and associate it with the current thread
        // no argument flags means the default strategy will be used for
        // scheduling the OS thread that the CUDA context is current on while
        // waiting for results from the GPU. The default strategy is CU_CTX_SCHED_AUTO.
        //
        // From the JCUDA Javadocs, CU_CTX_SCHED_AUTO indicates:
        //
        // a heuristic based on the number of active CUDA contexts in the process C and
        // the number of logical processors in the system P. If C > P, then CUDA will
        // yield to other OS threads when waiting for the GPU, otherwise CUDA will not
        // yield while waiting for results and actively spin on the processor.
        context = new CUcontext( );
        cuCtxCreate( context, 0, device );

        // Load the file containing kernels for calculating histogram values on the GPU into a CUmodule
        String ptxFileName = PtxUtils.preparePtxFile( "src/main/java/resources/HistogramTextureKernel.cu" );
        module = new CUmodule( );
        cuModuleLoad( module, ptxFileName );

        // create a cudaGraphicsResource which allows CUDA kernels to access OpenGL textures
        // http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__TYPES_gc0c4e1704647178d9c5ba3be46517dcd.html
        gfxResource = new CUgraphicsResource( );
        // JCuda.cudaTextureType2D = GL_TEXTURE_2D and indicates the texture is a 2D texture
        // cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly indicates CUDA will only read from the texture
        cuGraphicsGLRegisterImage( gfxResource, texHandle, GL.GL_TEXTURE_2D, cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly );

        // create a cuArray object which will be used to access OpenGL texture data via CUDA
        hostArray = new CUarray( );

        // map the cuGraphicsResource, which makes it accessible to CUDA
        // (OpenGL should not access the texture until we unmap)
        cuGraphicsMapResources( 1, new CUgraphicsResource[] { gfxResource }, null );

        // associate the cuArray with the cuGraphicsResource
        cuGraphicsSubResourceGetMappedArray( hostArray, gfxResource, 0, 0 );

        // create a texture reference and bind it to the global variable "texture_float_2D" in the CUDA source code
        textureReference = new CUtexref( );
        cuModuleGetTexRef( textureReference, module, "texture_float_2D" );
        
        // set how to interpolate between texture values (no linear interpolation)
        // and what to do for texture coordinates outside of texture bounds (clamp)
        cuTexRefSetFilterMode( textureReference, CU_TR_FILTER_MODE_POINT );
        cuTexRefSetAddressMode( textureReference, 0, CU_TR_ADDRESS_MODE_CLAMP );
        cuTexRefSetAddressMode( textureReference, 1, CU_TR_ADDRESS_MODE_CLAMP );
        
        // access the texture using 0.0 to 1.0 normalized texture coordinates
        cuTexRefSetFlags( textureReference, CU_TRSF_NORMALIZED_COORDINATES );
        cuTexRefSetFormat( textureReference, CU_AD_FORMAT_FLOAT, 1 );

        // bind the cudaTextureReference to the cudaArray
        cuTexRefSetArray( textureReference, hostArray, CU_TRSA_OVERRIDE_FORMAT );

        // unbind cuGraphicsResource so that it can be accessed by OpenGL again
        //cuGraphicsUnmapResources( 1, new CUgraphicsResource[] { gfxResource }, null );

        
        // obtain the test function
        functionTest = new CUfunction( );
        cuModuleGetFunction( functionTest, module, "test_float_2D" );
        
        // allocate host memory for histogram bins
        hHistogramBins = new int[numBins];

        // allocate device pointer with space for all bin values
        dHistogramBins = new CUdeviceptr( );
        cuMemAlloc( dHistogramBins, numBins * Sizeof.INT );

        // zero out device memory array
        cuMemsetD32( dHistogramBins, 0, numBins );
    }
    
    public void calculateHistogram( float texPosX, float texPosY )
    {
        // map the cuGraphicsResource, which makes it accessible to CUDA
        // (OpenGL should not access the texture until we unmap)
        //cuGraphicsMapResources( 1, new CUgraphicsResource[] { gfxResource }, null );
        
        // set up the function parameters 
        Pointer pHistogramBins = Pointer.to( dHistogramBins );
        Pointer pPosX = Pointer.to( new float[] { texPosX } );
        Pointer pPosY = Pointer.to( new float[] { texPosY } );
        
        Pointer kernelParameters = Pointer.to( pHistogramBins, pPosX, pPosY );
        
        int numElements = 1;
        int blockSizeX = 256;
        int gridSizeX = ( int ) Math.ceil( ( double ) numElements / blockSizeX );
        cuLaunchKernel( functionTest, gridSizeX, 1, 1, // Grid dimension
                blockSizeX, 1, 1, // Block dimension
                0, null, // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        
        cuCtxSynchronize( );
        
        // unbind cuGraphicsResource so that it can be accessed by OpenGL again
        //cuGraphicsUnmapResources( 1, new CUgraphicsResource[] { gfxResource }, null );
        
        // copy the kernel output back to the host
        cuMemcpyDtoH( Pointer.to( hHistogramBins ), dHistogramBins, Sizeof.INT * numBins );
        
        System.out.println( "Kernel Output: ");
        for ( int i = 0 ; i < numBins ; i++ )
        {
            System.out.println( hHistogramBins[i] );
        }
    }

    public void dispose( GLContext glContext )
    {
        cuArrayDestroy( hostArray );
    }
}