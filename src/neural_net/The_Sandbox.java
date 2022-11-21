package neural_net;

import java.io.File;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class The_Sandbox 
{
	public static void main(String[] args) throws IOException //this class is just for testing the code. all the machinery is in Network.java
	{	
		//train program
		trainNewNetworkAndSaveState();
		
		//test program
		testCurrentNetworkFromState();
	}
	
	//randomizes the order of the test data, 10k images, which is entirely separate from the training data
	public static Object[] fetchTestData()
	{
		File dir = new File("C:\\Users\\Clayton\\Desktop\\mnist_data\\mnist_jpgfiles\\test");
		File[] fileArray = dir.listFiles();
		int[] labels = new int[fileArray.length];

		Pattern p = Pattern.compile("\\d+(?=_)");
		Matcher m;
		int num = 0;
		Object[] testingData = new Object[2];
		
		// parsing files for labels
		for(int i = 0; i < fileArray.length; i++) {
			String pathName = fileArray[i].toString();
			m = p.matcher(pathName);
			if(m.find()) {
				num = Integer.parseInt(m.group());
			}
			labels[i] = num;
		}
		testingData[0] = fileArray;
		testingData[1] = labels;
		return testingData;
	}
	

	public static void trainNewNetworkAndSaveState() throws IOException 
	{	
		int[] netParams = {784, 16, 16, 10};
		Network net = new Network(netParams);
		
		long start = System.nanoTime();
		net = new Network(netParams);
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to build the network");
		start = System.nanoTime();
		net.trainNetwork(20, 3000, 20);
		System.out.println("Took " + (System.nanoTime() - start)/1e9 + " to train.");
		net.saveState();
	}
	
	public static void testCurrentNetworkFromState() throws IOException 
	{	
		Object[] testData = fetchTestData();
		Network net = Network.restoreState(new File("network-state.bin"));
		File[] images = (File[]) testData[0];
		int[] expected = (int[]) testData[1];
		double accuracy = 0.0d;
		
		for(int i = 0; i < 10000; i++) 
		{
			net.getInputAndMakePrediction(images[i]);
			double[] guess = net.output().getComponents();
			
			double max = 0;
			int num = 0;
			for(int j = 0; j < guess.length; j++) {
				if(guess[j] > max) {
					max = guess[j];
					num = j;
				}
			}
			System.out.println("Image " + i + ": Expected " + expected[i] + ", predicted " + num);
			if (num == expected[i]) {
				accuracy += 1.0d;
			}
		}
		accuracy /= (double) 10000;
		System.out.println("Network has accuracy of " + accuracy + " on mnist test set of 10,000 images.");
	}

}
