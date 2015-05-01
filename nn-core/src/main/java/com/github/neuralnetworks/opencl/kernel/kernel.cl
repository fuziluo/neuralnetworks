typedef struct This_FullyConnected_FF_s{
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
}This_FullyConnected_FF;

void com_github_neuralnetworks_calculation_neuronfunctions_AparapiSigmoid$AparapiSigmoidFunction__after(This_FullyConnected_FF *this){
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
   This_FullyConnected_FF thisStruct;
   This_FullyConnected_FF* this=&thisStruct;
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
   This_FullyConnected_FF thisStruct;
   This_FullyConnected_FF* this=&thisStruct;
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

typedef struct This_FullyConnected_BP_s{
   int outputStartPosition;
   int outputRowStep;
   int miniBatchSize;
   int outputColumnStep;
   int activationStartPosition;
   int activationRowStep;
   __global float *ffActivation;
   __global float *output;
   int activationColumnStep;
   float learningRate;
   __constant int *inputStartPositions;
   __constant int *inputRowSteps;
   __constant int *inputColumnSteps;
   __constant int *weightStartPositions;
   __constant int *weightsInitialStep;
   __constant int *weightsStep;
   __constant int *weightsSize;
   __global float *input;
   __global float *weights;
   float momentum;
   __global float *weightUpdates;
   float l1weightDecay;
   float l2weightDecay;
   int series;
}This_FullyConnected_BP;

void com_github_neuralnetworks_training_backpropagation_BackPropagationSigmoid$AparapiBackpropSigmoid__calcDerivative(This_FullyConnected_BP *this){
   float activation = 0.0f;
   int end = (this->outputStartPosition + (get_global_id(0) * this->outputRowStep)) + (this->miniBatchSize * this->outputColumnStep);
   int outputId = this->outputStartPosition + (get_global_id(0) * this->outputRowStep);
   int activationId = this->activationStartPosition + (get_global_id(0) * this->activationRowStep);
   for (; outputId<end; activationId = activationId + this->activationColumnStep){
      activation = this->ffActivation[activationId];
      this->output[outputId]  = (this->output[outputId] * activation) * (1.0f - activation);
      outputId = outputId + this->outputColumnStep;
   }
   return;
}
void com_github_neuralnetworks_training_backpropagation_AparapiBackpropagationFullyConnected__after(This_FullyConnected_BP *this){
   int id = get_global_id(0);
   int inputStartPosition = 0;
   int inputRowsStep = 0;
   int inputColumnsStep = 0;
   int weightStartPosition = 0;
   int weightStep = 0;
   int dim = 0;
   int weightIndex = 0;
   float weight = 0.0f;
   float weightUpdate = 0.0f;
   float lr = this->learningRate;
   for (int k = 0; k<this->series; k++){
      inputStartPosition = this->inputStartPositions[k];
      inputRowsStep = this->inputRowSteps[k];
      inputColumnsStep = this->inputColumnSteps[k];
      weightStartPosition = this->weightStartPositions[k] + (this->weightsInitialStep[k] * id);
      weightStep = this->weightsStep[k];
      dim = this->weightsSize[k];
      for (int j = 0; j<dim; j++){
         weightUpdate = 0.0f;
         for (int i = 0; i<this->miniBatchSize; i++){
            weightUpdate = weightUpdate + (this->input[((inputStartPosition + (j * inputRowsStep)) + (i * inputColumnsStep))] * this->ffActivation[((this->activationStartPosition + (id * this->activationRowStep)) + (i * this->activationColumnStep))]);
         }
         weightIndex = weightStartPosition + (j * weightStep);
         weight = this->weights[weightIndex];
         weightUpdate = (((lr * weightUpdate) + (this->momentum * this->weightUpdates[weightIndex])) - (this->l1weightDecay * fabs(weight))) - (((this->l2weightDecay * weight) * weight) / 2.0f);
         this->weights[weightIndex]  = this->weights[weightIndex] + weightUpdate;
         this->weightUpdates[weightIndex]  = weightUpdate;
      }
   }
   com_github_neuralnetworks_training_backpropagation_BackPropagationSigmoid$AparapiBackpropSigmoid__calcDerivative(this);
   return;
}
__kernel void BackpropagationFullyConnectedSigmoid(
   int outputStartPosition, 
   int outputRowStep, 
   int miniBatchSize, 
   int outputColumnStep, 
   int activationStartPosition, 
   int activationRowStep, 
   __global float *ffActivation, 
   __global float *output, 
   int activationColumnStep, 
   float learningRate, 
   __constant int *inputStartPositions, 
   __constant int *inputRowSteps, 
   __constant int *inputColumnSteps, 
   __constant int *weightStartPositions, 
   __constant int *weightsInitialStep, 
   __constant int *weightsStep, 
   __constant int *weightsSize, 
   __global float *input, 
   __global float *weights, 
   float momentum, 
   __global float *weightUpdates, 
   float l1weightDecay, 
   float l2weightDecay, 
   int series
){
   This_FullyConnected_BP thisStruct;
   This_FullyConnected_BP* this=&thisStruct;
   this->outputStartPosition = outputStartPosition;
   this->outputRowStep = outputRowStep;
   this->miniBatchSize = miniBatchSize;
   this->outputColumnStep = outputColumnStep;
   this->activationStartPosition = activationStartPosition;
   this->activationRowStep = activationRowStep;
   this->ffActivation = ffActivation;
   this->output = output;
   this->activationColumnStep = activationColumnStep;
   this->learningRate = learningRate;
   this->inputStartPositions = inputStartPositions;
   this->inputRowSteps = inputRowSteps;
   this->inputColumnSteps = inputColumnSteps;
   this->weightStartPositions = weightStartPositions;
   this->weightsInitialStep = weightsInitialStep;
   this->weightsStep = weightsStep;
   this->weightsSize = weightsSize;
   this->input = input;
   this->weights = weights;
   this->momentum = momentum;
   this->weightUpdates = weightUpdates;
   this->l1weightDecay = l1weightDecay;
   this->l2weightDecay = l2weightDecay;
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
      com_github_neuralnetworks_training_backpropagation_AparapiBackpropagationFullyConnected__after(this);
      return;
   }
}
