package neural_net;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import linearAlgebra.Matrix;
import linearAlgebra.Vector;

@SuppressWarnings("unused")
public class sandbox {
	
	public static void main(String[] args) throws IOException {
		
		BufferedImage image = ImageIO.read(new File("C:\\Users\\Clayton\\Desktop\\mnist_data\\mnist_jpgfiles\\train\\mnist_2_1670.jpg"));
		Raster x = image.getData();
		
//		System.out.println(trainingImages[0]);
		
		for(int i = 0; i < x.getWidth(); i++) {
			System.out.print("\n");
			for(int j = 0; j < x.getHeight(); j++) {
				System.out.print((x.getSampleDouble(i, j, 0) / 255) + " ");
			}
		}


	}
	
}
