package edu.gmu.ulman.histogram;

import com.metsci.glimpse.support.texture.FloatTextureProjected2D;

public class AccessibleFloatTexture2D extends FloatTextureProjected2D
{
    public AccessibleFloatTexture2D( int dataSizeX, int dataSizeY )
    {
        super( dataSizeX, dataSizeY );
    }

    public int[] getTextureHandles( )
    {
        return textureHandles;
    }
}
