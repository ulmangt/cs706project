package edu.gmu.ulman.histogram;

import static com.metsci.glimpse.util.logging.LoggerUtils.*;

import it.unimi.dsi.fastutil.floats.Float2IntMap;
import it.unimi.dsi.fastutil.floats.Float2IntOpenHashMap;

import java.util.logging.Logger;

import javax.media.opengl.GL;
import javax.media.opengl.GLContext;

import com.metsci.glimpse.axis.Axis2D;
import com.metsci.glimpse.context.GlimpseBounds;
import com.metsci.glimpse.painter.base.GlimpseDataPainter2D;
import com.metsci.glimpse.painter.plot.HistogramPainter;
import com.metsci.glimpse.support.projection.InvertibleProjection;
import com.metsci.glimpse.support.projection.Projection;

public class CUDAHistogramPainter extends GlimpseDataPainter2D
{
    private static final Logger logger = Logger.getLogger( CUDAHistogramPainter.class.getName( ) );

    AccessibleFloatTexture2D texture;
    JCudaHistogramCalculator calculator;

    private HistogramPainter delegate;
    
    private double minValue;
    private double maxValue;
    private int numBins;
    
    private Axis2D textureAxis;

    public CUDAHistogramPainter( AccessibleFloatTexture2D texture, Axis2D textureAxis, int numBins, double minValue, double maxValue )
    {
        this.texture = texture;
        
        this.numBins = numBins;
        this.minValue = minValue;
        this.maxValue = maxValue;
        
        this.textureAxis = textureAxis;
        
        this.delegate = new HistogramPainter( );
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

            double stepX = 1.0 / texture.getDimensionSize( 0 );
            double stepY = 1.0 / texture.getDimensionSize( 1 );
            
            calculator = new JCudaHistogramCalculator( numBins, minValue, maxValue, stepX, stepY );

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
            float centerX = (float) textureAxis.getAxisX( ).getSelectionCenter( );
            float centerY = (float) textureAxis.getAxisY( ).getSelectionCenter( );
            
            float sizeX = (float) textureAxis.getAxisX( ).getSelectionSize( ) / 2;
            float sizeY = (float) textureAxis.getAxisY( ).getSelectionSize( ) / 2;
            
            // get the position of the mouse selection in axis coordinates
            float centerMinX = centerX - sizeX;
            float centerMaxX = centerX + sizeX;
            
            float centerMinY = centerY - sizeY;
            float centerMaxY = centerY + sizeY;
            
            // get the projection which maps between axis coordinates and texture coordinates
            Projection projection = texture.getProjection( );
            if ( projection instanceof InvertibleProjection )
            {
                // get the texture coordinates corresponding to the mouse selection
                InvertibleProjection invProjection = (InvertibleProjection) projection;
                float texFracMinX = (float) invProjection.getTextureFractionX( centerMinX, centerMinY );
                float texFracMaxX = (float) invProjection.getTextureFractionX( centerMaxX, centerMinY );
                
                float texFracMinY = (float) invProjection.getTextureFractionY( centerMinX, centerMinY );
                float texFracMaxY = (float) invProjection.getTextureFractionY( centerMinX, centerMaxY );
            
                // run the cuda kernel
                int[] bins = calculator.calculateHistogram( texFracMinX, texFracMaxX, texFracMinY, texFracMaxY );
                
                Float2IntMap map = new Float2IntOpenHashMap( );
                
                float binStep = (float) ( ( maxValue - minValue ) / numBins );
                int totalSize = 0;
                
                for ( int i = 0 ; i <  numBins ; i++ )
                {
                    float key = (float) ( minValue + binStep * i );
                    int value = bins[i];
                    totalSize += bins[i];
                    map.put( key, value );
                }
                
                delegate.setData( map, totalSize, binStep );
                
                delegate.paintTo( gl, bounds, axis );
            }
        }
    }

    @Override
    protected void dispose( GLContext context )
    {
        calculator.dispose( context );
    }
}
