package com.github.neuralnetworks.calculation.neuronfunctions;

import static org.jocl.CL.*;

import java.util.Arrays;
import java.util.List;

import org.jocl.*;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.training.backpropagation.BackPropagationConnectionCalculator;

public class BackpropagationFullyConnectedOpenCL extends FullyConnectedOpenCL implements BackPropagationConnectionCalculator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6751358756703136178L;
    /**
     * Activation of the output layer from the feedforward phase
     */
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

    private static final String KERNEL_SOURCE = "typedef struct This_s{\n" + 
    		"   int outputStartPosition;\n" + 
    		"   int outputRowStep;\n" + 
    		"   int miniBatchSize;\n" + 
    		"   int outputColumnStep;\n" + 
    		"   int activationStartPosition;\n" + 
    		"   int activationRowStep;\n" + 
    		"   __global float *ffActivation;\n" + 
    		"   __global float *output;\n" + 
    		"   int activationColumnStep;\n" + 
    		"   float learningRate;\n" + 
    		"   __constant int *inputStartPositions;\n" + 
    		"   __constant int *inputRowSteps;\n" + 
    		"   __constant int *inputColumnSteps;\n" + 
    		"   __constant int *weightStartPositions;\n" + 
    		"   __constant int *weightsInitialStep;\n" + 
    		"   __constant int *weightsStep;\n" + 
    		"   __constant int *weightsSize;\n" + 
    		"   __global float *input;\n" + 
    		"   __global float *weights;\n" + 
    		"   float momentum;\n" + 
    		"   __global float *weightUpdates;\n" + 
    		"   float l1weightDecay;\n" + 
    		"   float l2weightDecay;\n" + 
    		"   int series;\n" + 
    		"}This;\n" + 
    		"\n" + 
    		"void com_github_neuralnetworks_training_backpropagation_BackPropagationSigmoid$AparapiBackpropSigmoid__calcDerivative(This *this){\n" + 
    		"   float activation = 0.0f;\n" + 
    		"   int end = (this->outputStartPosition + (get_global_id(0) * this->outputRowStep)) + (this->miniBatchSize * this->outputColumnStep);\n" + 
    		"   int outputId = this->outputStartPosition + (get_global_id(0) * this->outputRowStep);\n" + 
    		"   int activationId = this->activationStartPosition + (get_global_id(0) * this->activationRowStep);\n" + 
    		"   for (; outputId<end; activationId = activationId + this->activationColumnStep){\n" + 
    		"      activation = this->ffActivation[activationId];\n" + 
    		"      this->output[outputId]  = (this->output[outputId] * activation) * (1.0f - activation);\n" + 
    		"      outputId = outputId + this->outputColumnStep;\n" + 
    		"   }\n" + 
    		"   return;\n" + 
    		"}\n" + 
    		"void com_github_neuralnetworks_training_backpropagation_AparapiBackpropagationFullyConnected__after(This *this){\n" + 
    		"   int id = get_global_id(0);\n" + 
    		"   int inputStartPosition = 0;\n" + 
    		"   int inputRowsStep = 0;\n" + 
    		"   int inputColumnsStep = 0;\n" + 
    		"   int weightStartPosition = 0;\n" + 
    		"   int weightStep = 0;\n" + 
    		"   int dim = 0;\n" + 
    		"   int weightIndex = 0;\n" + 
    		"   float weight = 0.0f;\n" + 
    		"   float weightUpdate = 0.0f;\n" + 
    		"   float lr = this->learningRate;\n" + 
    		"   for (int k = 0; k<this->series; k++){\n" + 
    		"      inputStartPosition = this->inputStartPositions[k];\n" + 
    		"      inputRowsStep = this->inputRowSteps[k];\n" + 
    		"      inputColumnsStep = this->inputColumnSteps[k];\n" + 
    		"      weightStartPosition = this->weightStartPositions[k] + (this->weightsInitialStep[k] * id);\n" + 
    		"      weightStep = this->weightsStep[k];\n" + 
    		"      dim = this->weightsSize[k];\n" + 
    		"      for (int j = 0; j<dim; j++){\n" + 
    		"         weightUpdate = 0.0f;\n" + 
    		"         for (int i = 0; i<this->miniBatchSize; i++){\n" + 
    		"            weightUpdate = weightUpdate + (this->input[((inputStartPosition + (j * inputRowsStep)) + (i * inputColumnsStep))] * this->ffActivation[((this->activationStartPosition + (id * this->activationRowStep)) + (i * this->activationColumnStep))]);\n" + 
    		"         }\n" + 
    		"         weightIndex = weightStartPosition + (j * weightStep);\n" + 
    		"         weight = this->weights[weightIndex];\n" + 
    		"         weightUpdate = (((lr * weightUpdate) + (this->momentum * this->weightUpdates[weightIndex])) - (this->l1weightDecay * fabs(weight))) - (((this->l2weightDecay * weight) * weight) / 2.0f);\n" + 
    		"         this->weights[weightIndex]  = this->weights[weightIndex] + weightUpdate;\n" + 
    		"         this->weightUpdates[weightIndex]  = weightUpdate;\n" + 
    		"      }\n" + 
    		"   }\n" + 
    		"   com_github_neuralnetworks_training_backpropagation_BackPropagationSigmoid$AparapiBackpropSigmoid__calcDerivative(this);\n" + 
    		"   return;\n" + 
    		"}\n" + 
    		"__kernel void BackpropagationFullyConnectedSigmoid(\n" + 
    		"   int outputStartPosition, \n" + 
    		"   int outputRowStep, \n" + 
    		"   int miniBatchSize, \n" + 
    		"   int outputColumnStep, \n" + 
    		"   int activationStartPosition, \n" + 
    		"   int activationRowStep, \n" + 
    		"   __global float *ffActivation, \n" + 
    		"   __global float *output, \n" + 
    		"   int activationColumnStep, \n" + 
    		"   float learningRate, \n" + 
    		"   __constant int *inputStartPositions, \n" + 
    		"   __constant int *inputRowSteps, \n" + 
    		"   __constant int *inputColumnSteps, \n" + 
    		"   __constant int *weightStartPositions, \n" + 
    		"   __constant int *weightsInitialStep, \n" + 
    		"   __constant int *weightsStep, \n" + 
    		"   __constant int *weightsSize, \n" + 
    		"   __global float *input, \n" + 
    		"   __global float *weights, \n" + 
    		"   float momentum, \n" + 
    		"   __global float *weightUpdates, \n" + 
    		"   float l1weightDecay, \n" + 
    		"   float l2weightDecay, \n" + 
    		"   int series\n" + 
    		"){\n" + 
    		"   This thisStruct;\n" + 
    		"   This* this=&thisStruct;\n" + 
    		"   this->outputStartPosition = outputStartPosition;\n" + 
    		"   this->outputRowStep = outputRowStep;\n" + 
    		"   this->miniBatchSize = miniBatchSize;\n" + 
    		"   this->outputColumnStep = outputColumnStep;\n" + 
    		"   this->activationStartPosition = activationStartPosition;\n" + 
    		"   this->activationRowStep = activationRowStep;\n" + 
    		"   this->ffActivation = ffActivation;\n" + 
    		"   this->output = output;\n" + 
    		"   this->activationColumnStep = activationColumnStep;\n" + 
    		"   this->learningRate = learningRate;\n" + 
    		"   this->inputStartPositions = inputStartPositions;\n" + 
    		"   this->inputRowSteps = inputRowSteps;\n" + 
    		"   this->inputColumnSteps = inputColumnSteps;\n" + 
    		"   this->weightStartPositions = weightStartPositions;\n" + 
    		"   this->weightsInitialStep = weightsInitialStep;\n" + 
    		"   this->weightsStep = weightsStep;\n" + 
    		"   this->weightsSize = weightsSize;\n" + 
    		"   this->input = input;\n" + 
    		"   this->weights = weights;\n" + 
    		"   this->momentum = momentum;\n" + 
    		"   this->weightUpdates = weightUpdates;\n" + 
    		"   this->l1weightDecay = l1weightDecay;\n" + 
    		"   this->l2weightDecay = l2weightDecay;\n" + 
    		"   this->series = series;\n" + 
    		"   {\n" + 
    		"      int id = get_global_id(0);\n" + 
    		"      int inputStartPosition = 0;\n" + 
    		"      int inputRowsStep = 0;\n" + 
    		"      int inputColumnsStep = 0;\n" + 
    		"      int weightStartPosition = 0;\n" + 
    		"      int weightStep = 0;\n" + 
    		"      int dim = 0;\n" + 
    		"      float value = 0.0f;\n" + 
    		"      for (int i = 0; i<this->miniBatchSize; i++){\n" + 
    		"         value = 0.0f;\n" + 
    		"         for (int k = 0; k<this->series; k++){\n" + 
    		"            inputStartPosition = this->inputStartPositions[k];\n" + 
    		"            inputRowsStep = this->inputRowSteps[k];\n" + 
    		"            inputColumnsStep = this->inputColumnSteps[k];\n" + 
    		"            weightStartPosition = this->weightStartPositions[k] + (this->weightsInitialStep[k] * id);\n" + 
    		"            weightStep = this->weightsStep[k];\n" + 
    		"            dim = this->weightsSize[k];\n" + 
    		"            for (int j = 0; j<dim; j++){\n" + 
    		"               {\n" + 
    		"                  float in = this->input[((inputStartPosition + (j * inputRowsStep)) + (i * inputColumnsStep))];\n" + 
    		"                  value = value + (in * this->weights[(weightStartPosition + (j * weightStep))]);\n" + 
    		"               }\n" + 
    		"            }\n" + 
    		"         }\n" + 
    		"         this->output[(this->outputStartPosition + (id * this->outputRowStep)) + (i * this->outputColumnStep)]  = this->output[(this->outputStartPosition + (id * this->outputRowStep)) + (i * this->outputColumnStep)] + value;\n" + 
    		"      }\n" + 
    		"      com_github_neuralnetworks_training_backpropagation_AparapiBackpropagationFullyConnected__after(this);\n" + 
    		"      return;\n" + 
    		"   }\n" + 
    		"}\n" + 
    		"";//TODO
    
	public BackpropagationFullyConnectedOpenCL(List<Connections> inputConnections, 
			ValuesProvider valuesProvider, 
			ValuesProvider activations, 
			List<Tensor> weightUpdates, 
			Layer targetLayer, 
			float learningRate, float momentum, float l1weightDecay, float l2weightDecay) {
		super(inputConnections, valuesProvider, targetLayer);
		Matrix m = TensorFactory.tensor(targetLayer, inputConnections, activations);
		this.ffActivation = m.getElements();
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
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
		super.calculate(connections, valuesProvider, targetLayer);
		long t0 = System.currentTimeMillis();

        //create command queue
        commandQueue = clCreateCommandQueue(context, device, 0, null);
//        String fileContent = "";
//        String path = ""; //TODO path of OpenCL kernel source file
//		try {
//			fileContent = new String(Files.readAllBytes(Paths.get(path)));
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
        
//        fileContent = KERNEL_SOURCE;
//        //Create Program With Source
//        program = clCreateProgramWithSource(context, 1, new String[]{ fileContent }, null, null);
//        //Build Program
//        clBuildProgram(program, 0, null, null, null, null);
        
		//create kernel
		cl_kernel kernel0 = clCreateKernel(program, "BackpropagationFullyConnectedSigmoid", null); //TODO arbitrary chosen. May change to switch
    	//create arguments
		int[] arg0 = new int[] {outputStartPosition};
		int[] arg1 = new int[] {outputRowStep};
		int[] arg2 = new int[] {miniBatchSize};
		int[] arg3 = new int[] {outputColumnStep};
		int[] arg4 = new int[] {activationStartPosition};
		int[] arg5 = new int[] {activationRowStep};
		//TODO all the read write flags may need to be optimized
		cl_mem arg6 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ffActivation.length* Sizeof.cl_float, Pointer.to(ffActivation), null);
		cl_mem arg7 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, output.length* Sizeof.cl_float, Pointer.to(output), null);
		int[] arg8 = new int[] {activationColumnStep};
		float[] arg9 = new float[] {learningRate};
		cl_mem arg10 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputStartPositions.length* Sizeof.cl_int, Pointer.to(inputStartPositions), null);
		cl_mem arg11 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputRowSteps.length* Sizeof.cl_int, Pointer.to(inputRowSteps), null);
		cl_mem arg12 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputColumnSteps.length* Sizeof.cl_int, Pointer.to(inputColumnSteps), null);
		cl_mem arg13 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightStartPositions.length* Sizeof.cl_int, Pointer.to(weightStartPositions), null);
		cl_mem arg14 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsInitialStep.length* Sizeof.cl_int, Pointer.to(weightsInitialStep), null);
		cl_mem arg15 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsStep.length* Sizeof.cl_int, Pointer.to(weightsStep), null);
		cl_mem arg16 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsSize.length* Sizeof.cl_int, Pointer.to(weightsSize), null);
		cl_mem arg17 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, input.length* Sizeof.cl_float, Pointer.to(input), null);
		cl_mem arg18 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
		float[] arg19 = new float[] {momentum};
		cl_mem arg20 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightUpdates.length* Sizeof.cl_float, Pointer.to(weightUpdates), null);
		float[] arg21 = new float[] {l1weightDecay};
		float[] arg22 = new float[] {l2weightDecay};
		int[] arg23 = new int[] {series};
		
        clSetKernelArg(kernel0, 0, Sizeof.cl_int, Pointer.to(arg0));
        clSetKernelArg(kernel0, 1, Sizeof.cl_int, Pointer.to(arg1));
        clSetKernelArg(kernel0, 2, Sizeof.cl_int, Pointer.to(arg2));
        clSetKernelArg(kernel0, 3, Sizeof.cl_int, Pointer.to(arg3));
        clSetKernelArg(kernel0, 4, Sizeof.cl_int, Pointer.to(arg4));
        clSetKernelArg(kernel0, 5, Sizeof.cl_int, Pointer.to(arg5));
        clSetKernelArg(kernel0, 6, Sizeof.cl_mem, Pointer.to(arg6));
        clSetKernelArg(kernel0, 7, Sizeof.cl_mem, Pointer.to(arg7));
        clSetKernelArg(kernel0, 8, Sizeof.cl_int, Pointer.to(arg8));
        clSetKernelArg(kernel0, 9, Sizeof.cl_float, Pointer.to(arg9));
        clSetKernelArg(kernel0, 10, Sizeof.cl_mem, Pointer.to(arg10));
        clSetKernelArg(kernel0, 11, Sizeof.cl_mem, Pointer.to(arg11));
        clSetKernelArg(kernel0, 12, Sizeof.cl_mem, Pointer.to(arg12));
        clSetKernelArg(kernel0, 13, Sizeof.cl_mem, Pointer.to(arg13));
        clSetKernelArg(kernel0, 14, Sizeof.cl_mem, Pointer.to(arg14));
        clSetKernelArg(kernel0, 15, Sizeof.cl_mem, Pointer.to(arg15));
        clSetKernelArg(kernel0, 16, Sizeof.cl_mem, Pointer.to(arg16));
        clSetKernelArg(kernel0, 17, Sizeof.cl_mem, Pointer.to(arg17));
        clSetKernelArg(kernel0, 18, Sizeof.cl_mem, Pointer.to(arg18));
        clSetKernelArg(kernel0, 19, Sizeof.cl_float, Pointer.to(arg19));
        clSetKernelArg(kernel0, 20, Sizeof.cl_mem, Pointer.to(arg20));
        clSetKernelArg(kernel0, 21, Sizeof.cl_float, Pointer.to(arg21));
        clSetKernelArg(kernel0, 22, Sizeof.cl_float, Pointer.to(arg22));
        clSetKernelArg(kernel0, 23, Sizeof.cl_int, Pointer.to(arg23));
        
//        System.out.println("weights bafore"+Arrays.toString(weights));
//		System.out.println("ffActivation "+Arrays.toString(ffActivation));
//		System.out.println("output "+Arrays.toString(output));
//		System.out.println("input "+Arrays.toString(input));


        //enqueues a command to execute a kernel on a device
        //local size null, leave to Nvidia implementation
        long[] global_work_size = {targetLayer.getUnitCount(connections)};
        clEnqueueNDRangeKernel(commandQueue, kernel0, 1, null, global_work_size, null, 0, null, null);
        //decrements the kernel reference count
        clReleaseKernel(kernel0);
        //wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
        clFinish(commandQueue);
        //read data from GPU
        clEnqueueReadBuffer(commandQueue, arg7, CL_TRUE, 0, output.length * Sizeof.cl_float, Pointer.to(output), 0, null, null);
        clEnqueueReadBuffer(commandQueue, arg18, CL_TRUE, 0, weights.length * Sizeof.cl_float, Pointer.to(weights), 0, null, null);
        clEnqueueReadBuffer(commandQueue, arg20, CL_TRUE, 0, weightUpdates.length * Sizeof.cl_float, Pointer.to(weightUpdates), 0, null, null);
       //cleanup work
        clReleaseMemObject(arg6);
        clReleaseMemObject(arg7);
        clReleaseMemObject(arg10);
        clReleaseMemObject(arg11);
        clReleaseMemObject(arg12);
        clReleaseMemObject(arg13);
        clReleaseMemObject(arg14);
        clReleaseMemObject(arg15);
        clReleaseMemObject(arg16);
        clReleaseMemObject(arg17);
        clReleaseMemObject(arg18);
        clReleaseMemObject(arg20);
        clReleaseCommandQueue(commandQueue);
//        clReleaseProgram(program);
//        clReleaseContext(context);
//        System.out.println("weights after"+Arrays.toString(weights));
//		System.out.println("weightUpdates "+Arrays.toString(weightUpdates));
//		System.out.println("output "+Arrays.toString(output));
//		System.out.println("ffActivation "+Arrays.toString(ffActivation));
//		long t1 = System.currentTimeMillis();
//		System.out.println("BP clean up "+(t1-t0) + "ms");

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
