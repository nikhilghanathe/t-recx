
# model names
model_name_ev = 'trained_models/model_with_ev_assist'
model_name_noev = 'trained_models/model_without_ev_assist'
model_name_sdn = 'trained_models/model_sdn'
model_name_branchynet = 'trained_models/model_branchynet'
model_name_base = 'trained_models/pretrainedResnet.h5'

#Golden REFERENCE - This is the flops and accuracy of the base model i.e. without any t-recx techniques
flops_noEE = (809)/32# This is the flops count when we use the pretrainedResnet model from the tinyML repo w/o any changes to the structure
accuracy_noEE = 87.2

#loss weights for sdn and branchynet
loss_weights_branchynet = [1,0.3,1]
loss_weights_sdn = [0,0,1]

#hyperparameters
epochs = 500
batch_size = 32
test_batch_size = 100
total_samples= 10000