package edu.gmu.ulman.histogram;

import static com.metsci.glimpse.gl.util.GLPBufferUtils.*;

import javax.media.opengl.GLContext;

import com.metsci.glimpse.canvas.FrameBufferGlimpseCanvas;
import com.metsci.glimpse.context.GlimpseContext;
import com.metsci.glimpse.gl.GLSimpleFrameBufferObject;
import com.metsci.glimpse.gl.Jogular;
import com.metsci.glimpse.plot.ColorAxisPlot2D;

public class HeatMapHistogramTest
{
    public static void main( String[] args )
    {
        // add OpenGL native libraries to java.library.path
        Jogular.initJogl( );
        
        // build the GlimpseLayout which contains the heat map and histogram painters
        HeatMapHistogramViewer viewer = new HeatMapHistogramViewer( );
        ColorAxisPlot2D layout = viewer.getLayout( );
        
        // set the selected region of the heat map
        layout.getAxisX( ).setSelectionCenter( 0.5 );
        layout.getAxisX( ).setSelectionSize( 0.1 );
        layout.getAxisY( ).setSelectionCenter( 0.5 );
        layout.getAxisY( ).setSelectionSize( 0.1 );
        
        // create a GlimpseCanvas which renders onto an offscreen OpenGL renderbuffer
        // backed by an OpenGL texture
        GLContext parentContext = createPixelBuffer( 1, 1 ).getContext( );
        FrameBufferGlimpseCanvas frameBuffer = new FrameBufferGlimpseCanvas( 800, 800, parentContext );

        // get a GlimpseContext and GLContext for the FrameBufferGlimpseCanvas
        GlimpseContext context = frameBuffer.getGlimpseContext( );
        GLSimpleFrameBufferObject fbo = frameBuffer.getFrameBuffer( );
        GLContext glContext = context.getGLContext( );

        // make the FrameBufferGlimpseCanvas current and bind it to the GLContext
        // so that OpenGL drawing operations will draw into the frame buffer
        glContext.makeCurrent( );
        fbo.bind( glContext );

        // paint the provided layout into the frame buffer canvas
        // this also triggers the CUDA histogram calculation
        layout.paintTo( context );

        // dispose of OpenGL resources
        layout.dispose( context );
        frameBuffer.dispose( );
        
        // unbind the frame buffer and release the gl context
        fbo.unbind( glContext );
    }
}
