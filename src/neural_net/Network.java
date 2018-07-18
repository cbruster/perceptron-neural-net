package neural_net;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.imageio.ImageIO;
import linearAlgebra.Matrix;
import linearAlgebra.Vector;

//TODO make this shit serializeable 
public class Network {
	
	public static Neuron[][] layers;
	public static File[] trainingImages;
	public static int[] trainingLabels;
	
	public static void main(String[] args) throws IOException {
		
		//variables
		int[] netParams = {784, 16, 16, 10};
		@SuppressWarnings("unused")
		Object[] trainingData;
		
		//program
		long start = System.nanoTime();
		buildNetwork(netParams);
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to build the network");
		start = System.nanoTime();
		makeTrainingData();
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to make training data");
		start = System.nanoTime();
		trainNetwork(1200, 50);
		System.out.println("Took "  + (System.nanoTime() - start)/1e9 + " to train.");
	}
	
	private static void buildNetwork(int[] networkParams) 
	{
		int numLayers = networkParams.length;
		layers = new Neuron[numLayers][];
		
		for(int i = 0; i < numLayers; i++) {
			int layerSize = networkParams[i];
			layers[i] = new Neuron[layerSize];
			
			for(int j = 0; j < layerSize; j++) {
				layers[i][j] = new Neuron();
//				System.out.println("Placing neuron in layer " + i + " at index " + j);
				if(i == 0) {
					layers[i][j].isInputLayerNeuron = true;
				}
				if(i >= 1){
					layers[i][j].isInputLayerNeuron = false;
					layers[i][j].makeConnections(layers[i-1]); // sets weights of connections random
					layers[i][j].setBias(Math.random());       // randomize neuron bias
				}
			}
		}
//		System.out.println("Neural net is built!");
	}
	
	private static void makeTrainingData() { //TODO take params, and return Object[] = {File[], int[]} files and labels respectively
		
		File dir = new File("PATH");
		File[] fileArray = dir.listFiles();
		int[] labels = new int[fileArray.length];
		List<File> fileList = new ArrayList<File>();
		Pattern p = Pattern.compile("\\d+(?=_)");
		Matcher m;
		int num = 0;
		
		// add files to collection for shuffling
		for(int i = 0; i < fileArray.length; i++) {
			fileList.add(fileArray[i]);
		}
		Collections.shuffle(fileList);
		
		// parsing files for labels
		for(int i = 0; i < fileArray.length; i++) {
			fileArray[i] = fileList.get(i); //put files back in array
			String pathName = fileArray[i].toString();
			m = p.matcher(pathName);
			if(m.find()) {
				num = Integer.parseInt(m.group());
			}
			labels[i] = num;
		}
		
		trainingImages = fileArray;
		trainingLabels = labels;
	}
	
	private static void getInputAndMakePrediction(int arg) throws IOException
	{
		BufferedImage image = ImageIO.read(trainingImages[arg]);
		Raster x = image.getData();
		
		//setting activation layer
		do {
			int i = 0, j = 0;
			for(Neuron n : layers[0]) {
				n.setActivation(x.getSampleDouble(i, j, 0) / 255.0d);
				j++;
				if(j == x.getWidth()) {
					j = 0;
					i++;
				}
			} 
		} 
		while(false);
		
		//feeding the data forward in the network
		for(int i = 1; i < layers.length; i++) {
			for(int j = 0; j < layers[i].length; j++) {
				layers[i][j].activate();
			}
		}
	}
	
	private static void trainNetwork(int epochs, int batchSize) throws IOException
	{
		int e = 0; //counting epochs
		int g = 0; //indexing gradient array
		int x = 0; //indexing training images
		int l = 0; //indexing labels
		double[] gradient;
		Vector gradientVector = null;
		Vector[] outcomes = new Vector[batchSize];
		Matrix expectedOutcomes;
		
		while (e < epochs) {
			e++;
			//making the matrix for storing expected values for backpropagation
			for(int i = 0; i < batchSize; i++) {
				int num = trainingLabels[l];
				l++;
				double[] column = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				column[num] = 1.0d;
				outcomes[i] = new Vector(column);
			}
			expectedOutcomes = Matrix.vectorsToMatrix(outcomes);
			
			//forming gradient stochastically via backpropagation 
			for(int i = 0; i < batchSize; i++) {
				
				getInputAndMakePrediction(x);
				x++;
				
				// showing outputs while training
				Vector[] exampleOutput = {getActivationVectorForLayer(layers.length - 1), expectedOutcomes.getColumnAsVector(i)};
				System.out.println("Prediction vs Expected");
				Matrix.vectorsToMatrix(exampleOutput).putMatrix();
				
				// backpropagation is done here
				if(i == 0) {
					gradientVector = backprop(expectedOutcomes.getColumnAsVector(0));
				} else {
					gradientVector = Vector.Add(
							gradientVector, 
							backprop(expectedOutcomes.getColumnAsVector(i)));
				}
			}
			gradientVector.scale(1.0d / (double) batchSize);
			gradient = gradientVector.getComponents();
			
			//modify network params based on gradient
			g = 0;
			for(int i = 1; i < Network.layers.length; i++) {
				//go through neurons in layer setting its synapses to new values
				for(int j = 0; j < layers[i].length; j++) {
					for(Synapse s : layers[i][j].synapses) {
						s.setWeight(
								s.getWeight() + 
								gradient[g]);
						g++;
					}
				}
				//go through the neurons in the layer again to set biases
				for(int j = 0; j < layers[i].length; j++) {
					layers[i][j].setBias(
							layers[i][j].getBias() + 
							gradient[g]);
					g++;
				}
			}
		}
	}
	
	private static Matrix getWeightMatrixForLayer(int layer)
	{
		if(layer == 0) {
			return null;
		} else {
			int N = layers[layer].length;
			double[][] weightMatrixData = new double[N][];
			for(int i = 0; i < N; i++) {
				weightMatrixData[i] = layers[layer][i].getWeightsActingOnNeuron();
			}
			return new Matrix(weightMatrixData);
		}
	}

	private static Vector getActivationVectorForLayer(int layer) 
	{
		Neuron[] neurons = getNeuronsInLayer(layer);
		double[] activations = new double[neurons.length];
		
		for(int i = 0; i < neurons.length; i++) {
			activations[i] = neurons[i].getActivation();
		}
		return new Vector(activations);
	}
	
	private static Neuron[] getNeuronsInLayer(int layer) 
	{
		return layers[layer];
	}

	private static Vector backprop(Vector expected)
	{
		//variables
		int lastLayer = layers.length - 1; //starting point for algorithm
		double[] errorComponents; // temp storage for vector components to index directly
		double[] activComponents; // temp storage for activation components to index directly
		double[] gradientArray;
		ArrayList<Double> gradient = new ArrayList<Double>();
		Vector activationError = null;
		Matrix layerWeights; // actual data not actually used, this matrix is called for its dimensions
		
		for(int layer = lastLayer; layer > 0; layer--) {
	
			//calculation for output layer is different than the rest
			if(layer == lastLayer) {
				activationError = Vector.hadamard(
						Vector.Sub(expected, getActivationVectorForLayer(lastLayer)),
						sigPrimeVector(lastLayer));
				errorComponents = activationError.getComponents(); 
				///error components are exactly equal to derivative of cost with respect to biases, so they are added
				for(int i = errorComponents.length - 1; i >= 0; i--) {
					gradient.add(errorComponents[i]);
				}
				layerWeights = getWeightMatrixForLayer(layer); 
				activComponents = getActivationVectorForLayer(layer-1).getComponents(); //previous layer normal activations
				for(int j = layerWeights.rows - 1; j >= 0; j--) {
					for(int k = layerWeights.cols - 1; k >= 0; k--) {
						gradient.add(activComponents[k] * errorComponents[j]);
					}
				}
				
			} else {
				//activationError is reassigned using itself, because itself is the previous layer's activation error.
				Vector weightErrorProd = getWeightMatrixForLayer(layer + 1)
						.transpose()
						.mMultiply(activationError)
						.getColumnAsVector(0);
				activationError = Vector.hadamard(
						weightErrorProd, 
						sigPrimeVector(layer));
				errorComponents = activationError.getComponents();
				for(int i = errorComponents.length - 1 ; i >= 0; i--) {
					gradient.add(errorComponents[i]);
				}
				layerWeights = getWeightMatrixForLayer(layer); 
				activComponents = getActivationVectorForLayer(layer-1).getComponents();
				for(int j = layerWeights.rows - 1; j >= 0; j--) {
					for(int k = layerWeights.cols - 1; k >= 0; k--) {
						gradient.add(activComponents[k] * errorComponents[j]);
					}
				}
			}
		} 
		gradientArray = new double[gradient.size()];
		Collections.reverse(gradient); // remember, elements were filled into the gradient vector in reverse order
		for(int i = 0; i < gradientArray.length; i++) {
			gradientArray[i] = gradient.get(i);
		}
		return new Vector(gradientArray);
	}
	
	/* Misc methods that are useful */
	
	// the sigmoid function
	public static double sigmoid(double z) 
	{
		return 1.0d / (1.0d + Math.exp(-z));
	}
	// derivative of sigmoid function
	public static double sigPrime(double z) 
	{
		return Math.exp(z) / Math.pow((Math.exp(z) + 1.0d), 2);
	}
	// derivative of a layer's activation
	private static Vector sigPrimeVector(int layer) 
	{
		Neuron[] neurons = getNeuronsInLayer(layer);
		double[] sigPrimeZ = new double[neurons.length];
		
		for(int i = 0; i < neurons.length; i++) {
			sigPrimeZ[i] = sigPrime(neurons[i].zValue());
		}
		return new Vector(sigPrimeZ);
	}
}

class Neuron {
	
	protected boolean isInputLayerNeuron;
	private double activation;   // range is 0 to 1 as double
	private double bias;        // if this is a bias neuron, sigmoid will add a bias in calc for activation
	protected Synapse[] synapses;
	
	Neuron() 
	{
		activation = 0.0d;
		bias = 0.0d;
	}
	
	void setActivation(double arg) 
	{
		if (isInputLayerNeuron == true) {
			this.activation = arg;
		} else throw new RuntimeException("Cannot set activation of a neuron that is not in the input layer.");
	}
	
	void setBias(double arg) 
	{
		if(isInputLayerNeuron == false) {
			this.bias = arg;
		} else throw new RuntimeException("Cannot set bias of a neuron that is in the input layer.");
	}
	
	double getActivation() {
		return this.activation;
	}
	
	double getBias() {
		return this.bias;
	}
	
	double[] getWeightsActingOnNeuron() 
	{
		int L = this.synapses.length;
		double[] weightArray = new double[L];
		for (int i = 0; i < L; i++) {
			weightArray[i] = this.synapses[i].getWeight();
		}
		return weightArray;
	}

	//build connections between a layer and a node in the next layer
	void makeConnections(Neuron[] layer) 
	{
		this.synapses = new Synapse[layer.length];
		for(int i = 0; i < layer.length; i++) {
			this.synapses[i] = new Synapse(layer[i], this);
			this.synapses[i].setWeight(Math.random() * 2 - 1); //weight is set randomly when connections are made
		}
	}
	
	void activate() {	
		this.activation = Network.sigmoid(this.zValue());
	}
	
	double zValue() 
	{	
		Synapse[] s = this.synapses;
		Vector weightVector;
		Vector activationVector;
		double[] activationArray = new double[s.length];
		double[] weightArray = new double[s.length];
		double z;
		
		for(int i = 0; i < s.length; i++) {
			weightArray[i] = s[i].getWeight();
			activationArray[i] = s[i].getFrom().activation;
		}
		weightVector = new Vector(weightArray);
		activationVector = new Vector(activationArray);
		
		z = Matrix.dotProduct(
				activationVector.getComponents(), 
				weightVector.getComponents()
				) + this.bias;
		
		return z;
	}
}

class Synapse {
	
	private double weight;
	private final Neuron from;
	private final Neuron to;
	
	Synapse(Neuron from, Neuron to) 
	{
		this.from = from;
		this.to = to;
	}
	double getWeight() {
		return this.weight;
	}
	void setWeight(double arg) {
		this.weight = arg;
	}
	Neuron getFrom() {
		return this.from;
	}
	Neuron getTo() {
		return this.to;
	}
}
