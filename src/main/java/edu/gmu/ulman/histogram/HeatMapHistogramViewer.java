/*
 * Based on TaggedHeatMapExample.java, which is released under the following license:
 * 
 * Copyright (c) 2012, Metron, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Metron, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL METRON, INC. BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package edu.gmu.ulman.histogram;

import static com.metsci.glimpse.axis.tagged.Tag.*;

import com.metsci.glimpse.axis.Axis2D;
import com.metsci.glimpse.axis.tagged.NamedConstraint;
import com.metsci.glimpse.axis.tagged.Tag;
import com.metsci.glimpse.axis.tagged.TaggedAxis1D;
import com.metsci.glimpse.examples.Example;
import com.metsci.glimpse.gl.texture.ColorTexture1D;
import com.metsci.glimpse.layout.GlimpseAxisLayout2D;
import com.metsci.glimpse.layout.GlimpseLayoutProvider;
import com.metsci.glimpse.painter.decoration.BorderPainter;
import com.metsci.glimpse.painter.info.CursorTextZPainter;
import com.metsci.glimpse.painter.texture.TaggedHeatMapPainter;
import com.metsci.glimpse.plot.ColorAxisPlot2D;
import com.metsci.glimpse.plot.TaggedColorAxisPlot2D;
import com.metsci.glimpse.support.colormap.ColorGradients;
import com.metsci.glimpse.support.projection.FlatProjection;
import com.metsci.glimpse.support.projection.Projection;

/**
 * A 2D heatmap visualization which uses OpenGL Shader Language to apply a dynamic
 * color scale to the data and CUDA to dynamically calculate a histogram based
 * on the region of the overall heatmap selected by the mouse.
 *
 * @author ulman
 */
public class HeatMapHistogramViewer implements GlimpseLayoutProvider
{
    public static int TEXTURE_WIDTH = 500;
    public static int TEXTURE_HEIGHT = 500;
    
    public static float MIN_Z = -50.0f;
    public static float MAX_Z = 200.0f;
    
    public static void main( String[] args ) throws Exception
    {
        Example.showWithSwing( new HeatMapHistogramViewer( ) );
    }

    TaggedHeatMapPainter heatmap;

    @Override
    public ColorAxisPlot2D getLayout( )
    {
        // create a heat map plot
        TaggedColorAxisPlot2D plot = new TaggedColorAxisPlot2D( );

        // get the tagged z axis
        TaggedAxis1D axisZ = plot.getAxisZ( );

        // add some named tags at specific points along the axis
        // also add a custom "attribute" to each tag which specifies the relative (0 to 1)
        // point along the color scale which the tag is attached to
        final Tag t1 = axisZ.addTag( "T1", -200.0 ).setAttribute( TEX_COORD_ATTR, 0.0f );
        final Tag t5 = axisZ.addTag( "T5", 200.0 ).setAttribute( TEX_COORD_ATTR, 1.0f );

        // add a constraint which prevents dragging the tags past one another
        axisZ.addConstraint( new NamedConstraint( "C1" )
        {
            final static double buffer = 1.0;

            @Override
            public void applyConstraint( TaggedAxis1D axis )
            {
                if ( t1.getValue( ) > t5.getValue( ) - buffer ) t1.setValue( t5.getValue( ) - buffer );
            }
        } );

        // set border and offset sizes in pixels
        plot.setBorderSize( 15 );
        plot.setAxisSizeX( 50 );
        plot.setAxisSizeY( 60 );
        plot.setTitleHeight( 0 );

        // set the x, y, and z initial axis bounds
        plot.setMinX( 0.0f );
        plot.setMaxX( 1.0f );

        plot.setMinY( 0.0f );
        plot.setMaxY( 1.0f );

        plot.setMinZ( -300.0f );
        plot.setMaxZ( 300.0f );

        // lock the aspect ratio of the x and y axis to 1 to 1
        plot.lockAspectRatioXY( 1.0f );

        // set the size of the selection box to 0.1 units
        plot.setSelectionSize( 0.05f );
        plot.getAxisX( ).setSelectionCenter( 0.5 );
        plot.getAxisY( ).setSelectionCenter( 0.5 );

        // generate some data to display
        double[][] data = generateData( TEXTURE_WIDTH, TEXTURE_HEIGHT );

        // generate a projection indicating how the data should be mapped to plot coordinates
        Projection projection = new FlatProjection( 0, 1, 0, 1 );

        // create an OpenGL texture wrapper object
        AccessibleFloatTexture2D texture = new AccessibleFloatTexture2D( TEXTURE_WIDTH, TEXTURE_HEIGHT );

        // load the data and projection into the texture
        texture.setProjection( projection );
        texture.setData( data );

        // setup the color map for the painter
        ColorTexture1D colors = new ColorTexture1D( 1024 );
        colors.setColorGradient( ColorGradients.jet );

        // create a painter to display the heatmap data
        // this heatmap painter knows about axis tags
        heatmap = new TaggedHeatMapPainter( axisZ );
        heatmap.setDiscardAbove( true );
        heatmap.setDiscardBelow( true );

        // add the heatmap data and color scale to the painter
        heatmap.setData( texture );
        heatmap.setColorScale( colors );

        // add the painter to the plot
        plot.addPainter( heatmap );

        // load the color map into the plot (so the color scale is displayed on the z axis)
        plot.setColorScale( colors );

        // create a painter which displays the cursor position and data value under the cursor
        CursorTextZPainter cursorPainter = new CursorTextZPainter( );
        plot.addPainter( cursorPainter );
        cursorPainter.setOffsetBySelectionSize( true );

        // tell the cursor painter what texture to report data values from
        cursorPainter.setTexture( texture );

        // add a painter to calculate the histogram values using CUDA and display in a subplot
        Axis2D histogramAxis = new Axis2D( );
        histogramAxis.set( MIN_Z, MAX_Z, 0.0f, 1.0f );
        GlimpseAxisLayout2D histogramLayout = new GlimpseAxisLayout2D( histogramAxis );
        histogramLayout.setLayoutData( "pos 20 20 220 220" );
        
        histogramLayout.addPainter( new BorderPainter( ) );
        histogramLayout.addPainter( new CUDAHistogramPainter( texture, plot.getAxis( ), 10, MIN_Z, MAX_Z ) );
        
        plot.getLayoutCenter( ).addLayout( histogramLayout );
        
        return plot;
    }

    /*
     * Generate fake data for the heat map.
     */
    public static double[][] generateData( int width, int height )
    {
        double[][] data = new double[width][height];

        for ( int w = 0; w < width; w++ )
        {
            for ( int h = 0; h < height; h++ )
            {
                double x = w / ( double ) width;
                double y = h / ( double ) height;

                data[w][h] = ( y * y + Math.sin( 2 * Math.PI * x * x ) /*+ Math.random( )*/ ) * 100;
            }
        }

        return data;
    }
}