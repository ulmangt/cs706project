package edu.gmu.ulman.histogram;

import static com.metsci.glimpse.util.logging.LoggerUtils.*;

import java.util.logging.Logger;

import javax.media.opengl.GL;
import javax.media.opengl.GLContext;

import com.metsci.glimpse.axis.Axis2D;
import com.metsci.glimpse.context.GlimpseBounds;
import com.metsci.glimpse.painter.base.GlimpseDataPainter2D;
import com.metsci.glimpse.support.projection.InvertibleProjection;
import com.metsci.glimpse.support.projection.Projection;

public class HistogramPainter extends GlimpseDataPainter2D
{
    private static final Logger logger = Logger.getLogger( HistogramPainter.class.getName( ) );

    AccessibleFloatTexture2D texture;
    JCudaHistogramCalculator calculator;
    
    private double minValue;
    private double maxValue;
    private int numBins;

    public HistogramPainter( AccessibleFloatTexture2D texture, int numBins, double minValue, double maxValue )
    {
        this.texture = texture;
        
        this.numBins = numBins;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    @Override
    public void paintTo( GL gl, GlimpseBounds bounds, Axis2D axis )
    {
        // check if the opengl texture handle has been allocated
        // if not, do nothing until it is
        if ( calculator == null )
        {
            int[] handles = texture.getTextureHandles( );
            if ( handles == null || handles.length == 0 || handles[0] <= 0 ) return;

            calculator = new JCudaHistogramCalculator( numBins, minValue, maxValue );

            try
            {
                calculator.initialize( handles[0] );
            }
            catch ( Exception e )
            {
                logWarning( logger, "Trouble initializing JCudaHistogramCalculator (CUDA)", e );
                calculator = null;
            }
        }

        if ( calculator != null )
        {
            // get the position of the mouse selection in axis coordinates
            float centerX = (float) axis.getAxisX( ).getSelectionCenter( );
            float centerY = (float) axis.getAxisY( ).getSelectionCenter( );
            
            // get the projection which maps between axis coordinates and texture coordinates
            Projection projection = texture.getProjection( );
            if ( projection instanceof InvertibleProjection )
            {
                // get the texture coordinates corresponding to the mouse selection
                InvertibleProjection invProjection = (InvertibleProjection) projection;
                float texFracX = (float) invProjection.getTextureFractionX( centerX, centerY );
                float texFracY = (float) invProjection.getTextureFractionY( centerX, centerY );
            
                // run the cuda kernel
                calculator.calculateHistogram( texFracX, texFracY );
            }
        }
    }

    @Override
    protected void dispose( GLContext context )
    {
        calculator.dispose( context );
    }
}
