package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.Arrays;
import java.util.List;

import org.jocl.*;

import static org.jocl.CL.*;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

public class WeightedSumOpenCL extends FullyConnectedOpenCL implements
		ConnectionCalculator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3553195103714539635L;
	public static final int SIGMOID = 0;
	public static final int TANH = 1;
	public static final int RELU = 2;
	public static final int SOFTMAX = 3;
	public static final int NOACTIVATION = 4;
	private final int activationType;
    //The source code of kernel
	//Tanh, ReLU and Softmax need to be added TODO
    private static final String KERNEL_SOURCE = "typedef struct This_s{\n" + 
    		"   int outputStartPosition;\n" + 
    		"   int outputRowStep;\n" + 
    		"   int miniBatchSize;\n" + 
    		"   int outputColumnStep;\n" + 
    		"   __global float *output;\n" + 
    		"   __constant int *inputStartPositions;\n" + 
    		"   __constant int *inputRowSteps;\n" + 
    		"   __constant int *inputColumnSteps;\n" + 
    		"   __constant int *weightStartPositions;\n" + 
    		"   __constant int *weightsInitialStep;\n" + 
    		"   __constant int *weightsStep;\n" + 
    		"   __constant int *weightsSize;\n" + 
    		"   __global float *input;\n" + 
    		"   __global float *weights;\n" + 
    		"   int series;\n" + 
    		"}This;\n" + 
    		"\n" + 
    		"void com_github_neuralnetworks_calculation_neuronfunctions_AparapiSigmoid$AparapiSigmoidFunction__after(This *this){\n" + 
    		"   int end = (this->outputStartPosition + (get_global_id(0) * this->outputRowStep)) + (this->miniBatchSize * this->outputColumnStep);\n" + 
    		"   for (int i = this->outputStartPosition + (get_global_id(0) * this->outputRowStep); i<end; i = i + this->outputColumnStep){\n" + 
    		"      this->output[i]  = 1.0f / (1.0f + exp(-this->output[i]));\n" + 
    		"   }\n" + 
    		"   return;\n" + 
    		"}\n" + 
    		"__kernel void weightedSumSigmoid(\n" + 
    		"   int outputStartPosition, \n" + 
    		"   int outputRowStep, \n" + 
    		"   int miniBatchSize, \n" + 
    		"   int outputColumnStep, \n" + 
    		"   __global float *output, \n" + 
    		"   __constant int *inputStartPositions, \n" + 
    		"   __constant int *inputRowSteps, \n" + 
    		"   __constant int *inputColumnSteps, \n" + 
    		"   __constant int *weightStartPositions, \n" + 
    		"   __constant int *weightsInitialStep, \n" + 
    		"   __constant int *weightsStep, \n" + 
    		"   __constant int *weightsSize, \n" + 
    		"   __global float *input, \n" + 
    		"   __global float *weights, \n" + 
    		"   int series\n" + 
    		"){\n" + 
    		"   This thisStruct;\n" + 
    		"   This* this=&thisStruct;\n" + 
    		"   this->outputStartPosition = outputStartPosition;\n" + 
    		"   this->outputRowStep = outputRowStep;\n" + 
    		"   this->miniBatchSize = miniBatchSize;\n" + 
    		"   this->outputColumnStep = outputColumnStep;\n" + 
    		"   this->output = output;\n" + 
    		"   this->inputStartPositions = inputStartPositions;\n" + 
    		"   this->inputRowSteps = inputRowSteps;\n" + 
    		"   this->inputColumnSteps = inputColumnSteps;\n" + 
    		"   this->weightStartPositions = weightStartPositions;\n" + 
    		"   this->weightsInitialStep = weightsInitialStep;\n" + 
    		"   this->weightsStep = weightsStep;\n" + 
    		"   this->weightsSize = weightsSize;\n" + 
    		"   this->input = input;\n" + 
    		"   this->weights = weights;\n" + 
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
    		"      com_github_neuralnetworks_calculation_neuronfunctions_AparapiSigmoid$AparapiSigmoidFunction__after(this);\n" + 
    		"      return;\n" + 
    		"   }\n" + 
    		"}\n" + 
    		"\n" + 
    		"__kernel void weightedSum(\n" + 
    		"   int outputStartPosition, \n" + 
    		"   int outputRowStep, \n" + 
    		"   int miniBatchSize, \n" + 
    		"   int outputColumnStep, \n" + 
    		"   __global float *output, \n" + 
    		"   __constant int *inputStartPositions, \n" + 
    		"   __constant int *inputRowSteps, \n" + 
    		"   __constant int *inputColumnSteps, \n" + 
    		"   __constant int *weightStartPositions, \n" + 
    		"   __constant int *weightsInitialStep, \n" + 
    		"   __constant int *weightsStep, \n" + 
    		"   __constant int *weightsSize, \n" + 
    		"   __global float *input, \n" + 
    		"   __global float *weights, \n" + 
    		"   int series\n" + 
    		"){\n" + 
    		"   This thisStruct;\n" + 
    		"   This* this=&thisStruct;\n" + 
    		"   this->outputStartPosition = outputStartPosition;\n" + 
    		"   this->outputRowStep = outputRowStep;\n" + 
    		"   this->miniBatchSize = miniBatchSize;\n" + 
    		"   this->outputColumnStep = outputColumnStep;\n" + 
    		"   this->output = output;\n" + 
    		"   this->inputStartPositions = inputStartPositions;\n" + 
    		"   this->inputRowSteps = inputRowSteps;\n" + 
    		"   this->inputColumnSteps = inputColumnSteps;\n" + 
    		"   this->weightStartPositions = weightStartPositions;\n" + 
    		"   this->weightsInitialStep = weightsInitialStep;\n" + 
    		"   this->weightsStep = weightsStep;\n" + 
    		"   this->weightsSize = weightsSize;\n" + 
    		"   this->input = input;\n" + 
    		"   this->weights = weights;\n" + 
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
    		"      return;\n" + 
    		"   }\n" + 
    		"}";
	public WeightedSumOpenCL(List<Connections> inputConnections,
			ValuesProvider valuesProvider, Layer targetLayer, int activationType) {
		super(inputConnections, valuesProvider, targetLayer);
		this.activationType = activationType;		
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
        cl_kernel kernel0;
        switch (activationType) {
    	case SIGMOID:
    		kernel0 = clCreateKernel(program, "weightedSumSigmoid", null); 
    		break;
    	case TANH:
    		kernel0 = clCreateKernel(program, "weightedSumTanh", null);
    		break;
    	case RELU:
    		kernel0 = clCreateKernel(program, "weightedSumReLU", null);
   		break;
    	case SOFTMAX:
    		kernel0 = clCreateKernel(program, "weightedSumSoftmax", null);
    		break;
    	case NOACTIVATION:
    		kernel0 = clCreateKernel(program, "weightedSum", null); 
    		break;
    	default:
    		kernel0 = clCreateKernel(program, "weightedSum", null); 
    		break;        	
        }
        
    	//create arguments
		int[] arg0 = new int[] {outputStartPosition};
		int[] arg1 = new int[] {outputRowStep};
		int[] arg2 = new int[] {miniBatchSize};
		int[] arg3 = new int[] {outputColumnStep};
		//TODO all the read write flags may need to be optimized
		cl_mem arg4 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, output.length* Sizeof.cl_float, Pointer.to(output), null);
		cl_mem arg5 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputStartPositions.length* Sizeof.cl_int, Pointer.to(inputStartPositions), null);
		cl_mem arg6 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputRowSteps.length* Sizeof.cl_int, Pointer.to(inputRowSteps), null);
		cl_mem arg7 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, inputColumnSteps.length* Sizeof.cl_int, Pointer.to(inputColumnSteps), null);
		cl_mem arg8 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightStartPositions.length* Sizeof.cl_int, Pointer.to(weightStartPositions), null);
		cl_mem arg9 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsInitialStep.length* Sizeof.cl_int, Pointer.to(weightsInitialStep), null);
		cl_mem arg10 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsStep.length* Sizeof.cl_int, Pointer.to(weightsStep), null);
		cl_mem arg11 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsSize.length* Sizeof.cl_int, Pointer.to(weightsSize), null);
		cl_mem arg12 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, input.length* Sizeof.cl_float, Pointer.to(input), null);
		cl_mem arg13 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
		int[] arg14 = new int[] {series};
		
        clSetKernelArg(kernel0, 0, Sizeof.cl_int, Pointer.to(arg0));
        clSetKernelArg(kernel0, 1, Sizeof.cl_int, Pointer.to(arg1));
        clSetKernelArg(kernel0, 2, Sizeof.cl_int, Pointer.to(arg2));
        clSetKernelArg(kernel0, 3, Sizeof.cl_int, Pointer.to(arg3));
        clSetKernelArg(kernel0, 4, Sizeof.cl_mem, Pointer.to(arg4));
        clSetKernelArg(kernel0, 5, Sizeof.cl_mem, Pointer.to(arg5));
        clSetKernelArg(kernel0, 6, Sizeof.cl_mem, Pointer.to(arg6));
        clSetKernelArg(kernel0, 7, Sizeof.cl_mem, Pointer.to(arg7));
        clSetKernelArg(kernel0, 8, Sizeof.cl_mem, Pointer.to(arg8));
        clSetKernelArg(kernel0, 9, Sizeof.cl_mem, Pointer.to(arg9));
        clSetKernelArg(kernel0, 10, Sizeof.cl_mem, Pointer.to(arg10));
        clSetKernelArg(kernel0, 11, Sizeof.cl_mem, Pointer.to(arg11));
        clSetKernelArg(kernel0, 12, Sizeof.cl_mem, Pointer.to(arg12));
        clSetKernelArg(kernel0, 13, Sizeof.cl_mem, Pointer.to(arg13));
        clSetKernelArg(kernel0, 14, Sizeof.cl_int, Pointer.to(arg14));
//        System.out.println("weights bafore "+Arrays.toString(weights));
//		System.out.println("output bafore "+Arrays.toString(output));
//		System.out.println("input bafore "+Arrays.toString(input));

        //enqueues a command to execute a kernel on a device
        //local size null, leave to Nvidia implementation
        long[] global_work_size = {targetLayer.getUnitCount(connections)};
        clEnqueueNDRangeKernel(commandQueue, kernel0, 1, null, global_work_size, null, 0, null, null);
        //decrements the kernel reference count
        clReleaseKernel(kernel0);
        //wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
        clFinish(commandQueue);
        //read data from GPU
        clEnqueueReadBuffer(commandQueue, arg4, CL_TRUE, 0, output.length * Sizeof.cl_float, Pointer.to(output), 0, null, null);
        //cleanup work
        clReleaseMemObject(arg4);
        clReleaseMemObject(arg5);
        clReleaseMemObject(arg6);
        clReleaseMemObject(arg7);
        clReleaseMemObject(arg8);
        clReleaseMemObject(arg9);
        clReleaseMemObject(arg10);
        clReleaseMemObject(arg11);
        clReleaseMemObject(arg12);
        clReleaseMemObject(arg13);
        clReleaseCommandQueue(commandQueue);
//        clReleaseProgram(program);
//        clReleaseContext(context);
        
        
//        System.out.println("weights after "+Arrays.toString(weights));
//		System.out.println("output after "+Arrays.toString(output));
//		System.out.println("input after "+Arrays.toString(input));

//		long t1 = System.currentTimeMillis();
//		System.out.println("FF clean up "+(t1-t0) + "ms");


	}
}
