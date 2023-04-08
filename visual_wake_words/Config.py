total_samples = 4448
BATCH_SIZE=32
IMAGE_SIZE = 96
epochs_0 = 20
epochs_1 = 10
epochs_2 = 20


# model names
model_name_ev = 'trained_models/model_with_ev_assist'
model_name_noev = 'trained_models/model_without_ev_assist'
model_name_base = 'trained_models/vww_96.h5'
#Golden REFERENCE - This is the flops and accuracy of the base model i.e. without any t-recx techniques
flops_noEE = 15.69# This is the flops count when we use the pretrainedResnet model from the tinyML repo w/o any changes to the structure
accuracy_noEE = 85.63


