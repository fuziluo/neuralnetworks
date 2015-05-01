typedef struct This_s{
   int outputStartPosition;
   int outputRowStep;
   int miniBatchSize;
   int outputColumnStep;
   __global float *output;
   __constant int *inputStartPositions;
   __constant int *inputRowSteps;
   __constant int *inputColumnSteps;
   __constant int *weightStartPositions;
   __constant int *weightsInitialStep;
   __constant int *weightsStep;
   __constant int *weightsSize;
   __global float *input;
   __global float *weights;
   int series;
}This;

void com_github_neuralnetworks_calculation_neuronfunctions_AparapiSigmoid$AparapiSigmoidFunction__after(This *this){
   int end = (this->outputStartPosition + (get_global_id(0) * this->outputRowStep)) + (this->miniBatchSize * this->outputColumnStep);
   for (int i = this->outputStartPosition + (get_global_id(0) * this->outputRowStep); i<end; i = i + this->outputColumnStep){
      this->output[i]  = 1.0f / (1.0f + exp(-this->output[i]));
   }
   return;
}
__kernel void weightedSumSigmoid(
   int outputStartPosition, 
   int outputRowStep, 
   int miniBatchSize, 
   int outputColumnStep, 
   __global float *output, 
   __constant int *inputStartPositions, 
   __constant int *inputRowSteps, 
   __constant int *inputColumnSteps, 
   __constant int *weightStartPositions, 
   __constant int *weightsInitialStep, 
   __constant int *weightsStep, 
   __constant int *weightsSize, 
   __global float *input, 
   __global float *weights, 
   int series
){
   This thisStruct;
   This* this=&thisStruct;
   this->outputStartPosition = outputStartPosition;
   this->outputRowStep = outputRowStep;
   this->miniBatchSize = miniBatchSize;
   this->outputColumnStep = outputColumnStep;
   this->output = output;
   this->inputStartPositions = inputStartPositions;
   this->inputRowSteps = inputRowSteps;
   this->inputColumnSteps = inputColumnSteps;
   this->weightStartPositions = weightStartPositions;
   this->weightsInitialStep = weightsInitialStep;
   this->weightsStep = weightsStep;
   this->weightsSize = weightsSize;
   this->input = input;
   this->weights = weights;
   this->series = series;
   {
      int id = get_global_id(0);
      int inputStartPosition = 0;
      int inputRowsStep = 0;
      int inputColumnsStep = 0;
      int weightStartPosition = 0;
      int weightStep = 0;
      int dim = 0;
      float value = 0.0f;
      for (int i = 0; i<this->miniBatchSize; i++){
         value = 0.0f;
         for (int k = 0; k<this->series; k++){
            inputStartPosition = this->inputStartPositions[k];
            inputRowsStep = this->inputRowSteps[k];
            inputColumnsStep = this->inputColumnSteps[k];
            weightStartPosition = this->weightStartPositions[k] + (this->weightsInitialStep[k] * id);
            weightStep = this->weightsStep[k];
            dim = this->weightsSize[k];
            for (int j = 0; j<dim; j++){
               {
                  float in = this->input[((inputStartPosition + (j * inputRowsStep)) + (i * inputColumnsStep))];
                  value = value + (in * this->weights[(weightStartPosition + (j * weightStep))]);
               }
            }
         }
         this->output[(this->outputStartPosition + (id * this->outputRowStep)) + (i * this->outputColumnStep)]  = this->output[(this->outputStartPosition + (id * this->outputRowStep)) + (i * this->outputColumnStep)] + value;
      }
      com_github_neuralnetworks_calculation_neuronfunctions_AparapiSigmoid$AparapiSigmoidFunction__after(this);
      return;
   }
}

__kernel void weightedSum(
   int outputStartPosition, 
   int outputRowStep, 
   int miniBatchSize, 
   int outputColumnStep, 
   __global float *output, 
   __constant int *inputStartPositions, 
   __constant int *inputRowSteps, 
   __constant int *inputColumnSteps, 
   __constant int *weightStartPositions, 
   __constant int *weightsInitialStep, 
   __constant int *weightsStep, 
   __constant int *weightsSize, 
   __global float *input, 
   __global float *weights, 
   int series
){
   This thisStruct;
   This* this=&thisStruct;
   this->outputStartPosition = outputStartPosition;
   this->outputRowStep = outputRowStep;
   this->miniBatchSize = miniBatchSize;
   this->outputColumnStep = outputColumnStep;
   this->output = output;
   this->inputStartPositions = inputStartPositions;
   this->inputRowSteps = inputRowSteps;
   this->inputColumnSteps = inputColumnSteps;
   this->weightStartPositions = weightStartPositions;
   this->weightsInitialStep = weightsInitialStep;
   this->weightsStep = weightsStep;
   this->weightsSize = weightsSize;
   this->input = input;
   this->weights = weights;
   this->series = series;
   {
      int id = get_global_id(0);
      int inputStartPosition = 0;
      int inputRowsStep = 0;
      int inputColumnsStep = 0;
      int weightStartPosition = 0;
      int weightStep = 0;
      int dim = 0;
      float value = 0.0f;
      for (int i = 0; i<this->miniBatchSize; i++){
         value = 0.0f;
         for (int k = 0; k<this->series; k++){
            inputStartPosition = this->inputStartPositions[k];
            inputRowsStep = this->inputRowSteps[k];
            inputColumnsStep = this->inputColumnSteps[k];
            weightStartPosition = this->weightStartPositions[k] + (this->weightsInitialStep[k] * id);
            weightStep = this->weightsStep[k];
            dim = this->weightsSize[k];
            for (int j = 0; j<dim; j++){
               {
                  float in = this->input[((inputStartPosition + (j * inputRowsStep)) + (i * inputColumnsStep))];
                  value = value + (in * this->weights[(weightStartPosition + (j * weightStep))]);
               }
            }
         }
         this->output[(this->outputStartPosition + (id * this->outputRowStep)) + (i * this->outputColumnStep)]  = this->output[(this->outputStartPosition + (id * this->outputRowStep)) + (i * this->outputColumnStep)] + value;
      }
      return;
   }
}
