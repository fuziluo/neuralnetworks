package com.github.neuralnetworks.samples.test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Ignore;
import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.ScalingInputFunction;
import com.github.neuralnetworks.samples.mnist.MnistInputProvider;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationTrainer;
import com.github.neuralnetworks.training.events.EarlyStoppingListener;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;

/**
 * MNIST test
 */
public class MnistTest {

    @Test
    public void testSigmoidBP() {
	Environment.getInstance().setUseDataSharedMemory(false);
	Environment.getInstance().setUseWeightsSharedMemory(false);
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 784, 10 }, true);

	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 0f, 1, 1000, 1);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
//	bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 5000, 0.015f));

//	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);
	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

//    @Ignore
    @Test
    public void testSigmoidHiddenBP() {
	NeuralNetworkImpl mlp = NNFactory.mlpSigmoid(new int[] { 784, 300, 100, 10 }, true);

	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(mlp, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 0f, 1, 1000, 1);

	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
	bpt.addEventListener(new EarlyStoppingListener(trainInputProvider, 5000, 0.015f));

	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
//	Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);
	bpt.train();
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testLeNetSmall() {
	// cpu execution mode
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
//    Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);

	// Convolutional network
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 20, 1 }, { 2, 2 }, { 5, 5, 50, 1 }, { 2, 2 }, {512}, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// Mnist dataset provider
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	// Backpropagation trainer that also works for convolutional and subsampling layers
//	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.01f, 0.5f, 0f, 0f, 0f, 1, 1000, 1);
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f), 0.5f), 0.1f, 0.0f, 0f, 0f, 0f, 500, 1, 200);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
	bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 100, 0.015f));

	// training
	bpt.train();

	// testing
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    @Test
    public void testLeNetTiny() {
	Environment.getInstance().setUseDataSharedMemory(false);
	Environment.getInstance().setUseWeightsSharedMemory(false);

	// very simple convolutional network with a single 2x2 max pooling layer
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 2, 2 }, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// MNIST dataset
//	System.out.println(System.getProperty("user.dir"));
	String cwd = System.getProperty("user.dir");
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
//	MnistInputProvider testInputProvider = new MnistInputProvider("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	// Backpropagation trainer that also works for convolutional and subsampling layers
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.02f, 0.5f, 0f, 0f, 0f, 1, 1, 1);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));

	// cpu execution
//	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
	// training
	bpt.train();

	// testing
	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }

    /**
     * MNIST small LeNet network
     */
    @Test
    public void testLeNetTiny2() {
	Environment.getInstance().setUseDataSharedMemory(false);
	Environment.getInstance().setUseWeightsSharedMemory(false);

	// very simple convolutional network with a single convolutional layer with 6 5x5 filters and a single 2x2 max pooling layer
	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 6, 1 }, {2, 2}, {10} }, true);
//	NeuralNetworkImpl nn = NNFactory.convNN(new int[][] { { 28, 28, 1 }, { 5, 5, 4, 1 }, {2, 2}, {10} }, true);
	nn.setLayerCalculator(NNFactory.lcSigmoid(nn, null));
	NNFactory.lcMaxPooling(nn);

	// MNIST dataset
	MnistInputProvider trainInputProvider = new MnistInputProvider("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	trainInputProvider.addInputModifier(new ScalingInputFunction(255));
	MnistInputProvider testInputProvider = new MnistInputProvider("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
	testInputProvider.addInputModifier(new ScalingInputFunction(255));

	// Backpropagation trainer that also works for convolutional and subsampling layers
	BackPropagationTrainer<?> bpt = TrainerFactory.backPropagation(nn, trainInputProvider, testInputProvider, new MultipleNeuronsOutputError(), new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 
			0.02f, 0.5f, 0f, 0f, 0f, 100, 1, 10);

	// log data
	bpt.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), false, true));
//	bpt.addEventListener(new EarlyStoppingListener(testInputProvider, 50, 0.015f));
	
	
	// cpu execution
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);
//	Environment.getInstance().setExecutionMode(EXECUTION_MODE.GPU);

	// training
	for (Connections ccs: nn.getInputLayer().getConnections()) {
		if (ccs instanceof Conv2DConnection) {
			System.out.println("Conv weights before training:\n"+Arrays.toString(((Conv2DConnection) ccs).getWeights().getElements()));
		}
	}
	for (Connections ccs: nn.getOutputLayer().getConnections()) {
		if (ccs instanceof FullyConnected) {
			System.out.println("Fully connected weights before training:\n"+Arrays.toString(((FullyConnected) ccs).getWeights().getElements()));
		}
	}
	bpt.train();
	for (Connections ccs: nn.getInputLayer().getConnections()) {
		if (ccs instanceof Conv2DConnection) {
			System.out.println("Conv weights after training:\n"+Arrays.toString(((Conv2DConnection) ccs).getWeights().getElements()));
		}
	}
	for (Connections ccs: nn.getOutputLayer().getConnections()) {
		if (ccs instanceof FullyConnected) {
			System.out.println("Fully connected weights after training:\n"+Arrays.toString(((FullyConnected) ccs).getWeights().getElements()));
		}
	}	// testing
//	bpt.test();

	assertEquals(0, bpt.getOutputError().getTotalNetworkError(), 0.1);
    }
}
