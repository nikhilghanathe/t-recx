#hyperparameters
total_samples = 4800
BATCH_SIZE=100


# model names
# model_name_ev = 'trained_models/model_with_ev_assist'
model_name_ev = 'trained_models/model_with_ev_assist_ee_at_half_point3'
model_name_noev = 'trained_models/model_without_ev_assist'
model_name_ev_orig_endpoint = 'trained_models/model_with_ev_assist_orig_endpoint'
model_name_noev_orig_endpoint = 'trained_models/model_without_ev_assist_orig_endpoint'
model_name_eefmaps_concat = 'trained_models/model_eefmaps_concat'
model_name_sdn = 'trained_models/model_sdn'
model_name_branchynet = 'trained_models/model_branchynet'
model_name_base = 'trained_models/kws_ref_model'

#Golden REFERENCE - This is the flops and accuracy of the base model i.e. without any t-recx techniques
flops_noEE = 5.54# This is the flops count when we use the base dscnn model from the tinyML repo w/o any changes to the structure
accuracy_noEE = 92.2

#loss weights for sdn and branchynet
loss_weights_sdn = [0,0,0,1]
loss_weights_branchynet = [1,0.3,1]

