package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

public class WeightedSumConnectionCalculatorOpenCL extends
		ConnectionCalculatorFullyConnected {

    /**
	 * 
	 */
	private static final long serialVersionUID = 5518880228392840943L;

	@Override
    protected ConnectionCalculator createInputFunction(List<Connections> inputConnections, ValuesProvider valuesProvider, Layer targetLayer) {
    	return new WeightedSumOpenCL(inputConnections, valuesProvider, targetLayer, WeightedSumOpenCL.NOACTIVATION);
    }

}
