package com.github.neuralnetworks.calculation.neuronfunctions;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.IntStream;

import static org.jocl.CL.*;

import org.jocl.*;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.opencl.OpenCL;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Util;

public abstract class FullyConnectedOpenCL implements ConnectionCalculator {
    private static final long serialVersionUID = -8435155322138790083L;
    protected cl_device_id device;
    protected cl_platform_id platform;
    protected cl_program program;
    protected cl_context context;
    protected cl_command_queue commandQueue;

    /**
     * Number of input samples that will be calculated simultaneously
     */
    protected final int miniBatchSize;

    /**
     * Number of input connections that will be "combined" for simultaneous
     * calculation
     */
    protected final int series;

    public float[] input;
    protected final int[] inputStartPositions;
    protected final int[] inputRowSteps;
    protected final int[] inputColumnSteps;

    /**
     * output values
     */
    protected float[] output;
    protected final int outputStartPosition;
    protected final int outputRowStep;
    protected final int outputColumnStep;

    protected final float[] weights;
    protected final int[] weightStartPositions;
    protected final int[] weightsSize;
    protected final int[] weightsInitialStep;
    protected final int[] weightsStep;

    public FullyConnectedOpenCL(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
	super();
	this.miniBatchSize = TensorFactory.batchSize(valuesProvider);

	// input
	input = TensorFactory.tensor(Util.getOppositeLayer(inputConnections.get(0), targetLayer), inputConnections.get(0), valuesProvider).getElements();

	weights = ((FullyConnected) inputConnections.get(0)).getWeights().getElements();
	inputConnections.forEach(c -> {
	    Tensor t = TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider);
	    if (!(c instanceof FullyConnected)) {
		throw new IllegalArgumentException("Only FullyConnected connections are supported");
	    }

	    if (!(t instanceof Matrix)) {
		throw new IllegalArgumentException("Only matrices are supported as input");
	    }

	    if (input != t.getElements()) {
		throw new IllegalArgumentException("Only one input array is allowed");
	    }

	    if (weights != ((FullyConnected) c).getWeights().getElements()) {
		throw new IllegalArgumentException("Only one weight array is allowed");
	    }
	});

	this.series = inputConnections.size();
	this.inputStartPositions = new int[series];
	this.inputRowSteps = new int[series];
	this.inputColumnSteps = new int[series];
	IntStream.range(0, inputConnections.size()).forEach(i -> {
	    Matrix m = TensorFactory.tensor(Util.getOppositeLayer(inputConnections.get(i), targetLayer), inputConnections.get(i), valuesProvider);
	    inputStartPositions[i] = m.getStartIndex();
	    inputRowSteps[i] = m.getRowElementsDistance();
	    inputColumnSteps[i] = m.getColumnElementsDistance();
	});

	// output
	Matrix o = TensorFactory.tensor(targetLayer, inputConnections, valuesProvider);
	this.output = o.getElements();
	this.outputStartPosition = o.getStartIndex();
	this.outputRowStep = o.getRowElementsDistance();
	this.outputColumnStep = o.getColumnElementsDistance();

	// weights
	this.weightStartPositions = new int[series];
	this.weightsSize = new int[series];
	this.weightsInitialStep = new int[series];
	this.weightsStep = new int[series];

	IntStream.range(0, inputConnections.size()).forEach(i -> {
	    Matrix w = ((FullyConnected) inputConnections.get(i)).getWeights();
	    weightStartPositions[i] = w.getStartIndex();
	    if (inputConnections.get(0).getOutputLayer() == targetLayer) {
		weightsSize[i] = w.getColumns();
		weightsInitialStep[i] = w.getRowElementsDistance();
		weightsStep[i] = w.getColumnElementsDistance();
	    } else {
		weightsSize[i] = w.getRows();
		weightsInitialStep[i] = w.getColumnElementsDistance();
		weightsStep[i] = w.getRowElementsDistance();
	    }
	});
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (accept(connections, valuesProvider, targetLayer)) {
		//TODO
		calculateOpenCL();
	} else {
	    throw new IllegalArgumentException("A parameter does not match");
	}
    }

    private void calculateOpenCL() {
        setExceptionsEnabled(true);
//    	long t0 = System.currentTimeMillis();

        platform = OpenCL.getPlatform();
        device = OpenCL.getDevice();
        context = OpenCL.getContext();
        program = OpenCL.getProgram();
//        setPlatformAndDevice();
//        
//        //create context
//        cl_context_properties contextProperties = new cl_context_properties();
//        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);    
//        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device},null, null, null);
//
//    	long t1 = System.currentTimeMillis();
//    	System.out.println("create context"+ (t1-t0) +"ms");
	}


	public boolean accept(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (TensorFactory.batchSize(valuesProvider) != miniBatchSize) {
	    return false;
	}

	if (TensorFactory.tensor(targetLayer, connections, valuesProvider).getElements() != output) {
	    return false;
	}

	if (connections.size() != series || connections.size() == 0) {
	    return false;
	}
	if (connections.stream().filter(c -> TensorFactory.tensor(Util.getOppositeLayer(c, targetLayer), c, valuesProvider).getElements() != input).findAny().isPresent()) {
	    return false;
	}

	return true;
    }

    public float[] getInput() {
        return input;
    }

    public void setInput(float[] input) {
        this.input = input;
    }

    public float[] getOutput() {
        return output;
    }

    public void setOutput(float[] output) {
        this.output = output;
    }

}
