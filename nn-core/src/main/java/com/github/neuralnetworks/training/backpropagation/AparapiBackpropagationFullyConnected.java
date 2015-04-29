package com.github.neuralnetworks.training.backpropagation;

import java.util.Arrays;
import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.AparapiWeightedSum;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;

/**
 * Aparapi Backpropagation base weighted sum Supports learning rate, momentum
 * and weight decay
 */
public class AparapiBackpropagationFullyConnected extends AparapiWeightedSum implements BackPropagationConnectionCalculator {

    private static final long serialVersionUID = -5101971690861270462L;

    /**
     * Activation of the output layer from the feedforward phase
     */
    
    
//    @Constant 
//    should not be constant solved by cjx 20150320
	public float[] ffActivation;
    public final int activationStartPosition;
    public final int activationRowStep;
    public final int activationColumnStep;

    /**
     * Weight updates array
     */
    public final float[] weightUpdates;

    protected float learningRate;
    protected final float momentum;
    protected final float l1weightDecay;
    protected final float l2weightDecay;

    public AparapiBackpropagationFullyConnected(List<Connections> inputConnections, ValuesProvider valuesProvider, ValuesProvider activations, List<Tensor> weightUpdates, Layer targetLayer, float learningRate, float momentum, float l1weightDecay, float l2weightDecay) {
	super(inputConnections, valuesProvider, targetLayer);

	Matrix m = TensorFactory.tensor(targetLayer, inputConnections, activations);
	this.ffActivation = m.getElements();
//	System.out.println("!!!!ffActivation "+Arrays.toString(ffActivation));
	this.activationStartPosition = m.getStartIndex();
	this.activationRowStep = m.getRowElementsDistance();
	this.activationColumnStep = m.getColumnElementsDistance();

	this.learningRate = momentum;
	this.momentum = momentum;
	this.l1weightDecay = l1weightDecay;
	this.l2weightDecay = l2weightDecay;
	this.weightUpdates = weightUpdates.get(0).getElements();
    }

    @Override
    protected void after() {
	int id = getGlobalId();

	int inputStartPosition = 0, inputRowsStep = 0, inputColumnsStep = 0, weightStartPosition = 0, weightStep = 0, dim = 0, weightIndex = 0;
	float weight = 0, weightUpdate = 0, lr = learningRate;

	// each input example
	for (int k = 0; k < series; k++) {
	    // each element in the row/column
	    inputStartPosition = inputStartPositions[k];
	    inputRowsStep = inputRowSteps[k];
	    inputColumnsStep = inputColumnSteps[k];
	    weightStartPosition = weightStartPositions[k] + weightsInitialStep[k] * id;
	    weightStep = weightsStep[k];
	    dim = weightsSize[k];

	    for (int j = 0; j < dim; j++) {
			weightUpdate = 0;
			for (int i = 0; i < miniBatchSize; i++) {
//				float in = input[inputStartPosition + j * inputRowsStep + i * inputColumnsStep]/1000.0f;
//				float ffa = 1000.0f*ffActivation[activationStartPosition + id * activationRowStep + i * activationColumnStep];
//				weightUpdate += in*ffa;
			    weightUpdate += input[inputStartPosition + j * inputRowsStep + i * inputColumnsStep] * ffActivation[activationStartPosition + id * activationRowStep + i * activationColumnStep];
			    intermediumOut[activationStartPosition + id * activationRowStep + i * activationColumnStep] = ffActivation[activationStartPosition + id * activationRowStep + i * activationColumnStep];
			}
			weightIndex = weightStartPosition + j * weightStep;
			weight = weights[weightIndex];
			weightUpdate = lr * weightUpdate + momentum * weightUpdates[weightIndex] - l1weightDecay * abs(weight) - l2weightDecay * weight * weight / 2;
			weights[weightIndex] += weightUpdate;
			weightUpdates[weightIndex] = weightUpdate;
	    }
	}
	calcDerivative();
    }

    /**
     * calculate derivative after weights update
     */
    protected void calcDerivative() {
    }

    @Override
    public float getLearningRate() {
	return learningRate;
    }

    @Override
    public void setLearningRate(float learningRate) {
	this.learningRate = learningRate;
    }

    @Override
    public float getMomentum() {
	return momentum;
    }

    @Override
    public void setMomentum(float momentum) {
    }

    @Override
    public float getL1weightDecay() {
	return l1weightDecay;
    }

    @Override
    public void setL1weightDecay(float weightDecay) {
    }

    @Override
    public float getL2weightDecay() {
	return l2weightDecay;
    }

    @Override
    public void setL2weightDecay(float l2weightDecay) {
    }

    @Override
    public ValuesProvider getActivations() {
	return null;
    }

    @Override
    public void setActivations(ValuesProvider activations) {
    }
}
