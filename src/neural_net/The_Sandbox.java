package neural_net;

import java.io.IOException;

public class The_Sandbox {
	
	public static void main(String[] args) throws IOException 
	{	
		//variables
		int[] netParams = {784, 16, 16, 10};
		Network net;
		Object[] trainingData;
		
		//program
		long start = System.nanoTime();
		net = new Network(netParams);
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to build the network");
		start = System.nanoTime();
		trainingData = Network.makeTrainingData();
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to make training data");
		start = System.nanoTime();
		net.trainNetwork(1200, 50, trainingData);
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to train.");
		
		net.saveState();
	}

}
