import os, sys, json


def analyze_stanalone_accuracy(model_name):
    fname = 'trace_data/trace_data_'+model_name + '_ee1_ref.json' 
    #load trace data
    with open(fname, 'r') as fp:
        predict_dict = json.load(fp)
    #load trace data of original
    # with open('trace_data/'+'trace_data_original.json', 'r') as fp:
    #     original_dict = json.load(fp)

    fname_eefinal = fname[:-10]+'final_ref.json'
    with open(fname_eefinal, 'r') as fp: 
        ee_final_dict = json.load(fp)

    EE_correct , correct = 0,  0
    for num, pred in predict_dict.items():
        truth = pred['truth']
        arg_max_1 = pred['arg_max_1']
        if truth == arg_max_1:
            EE_correct +=1

    for num, pred in ee_final_dict.items():
        truth = pred['truth']
        arg_max_1 = pred['arg_max_1']
        if truth == arg_max_1:
            correct +=1
    print(' EE correct is ', EE_correct/4448)
    print('EE final correct is ', correct/4448)

if __name__ == '__main__':
	model_name = sys.argv[1][15:]
	analyze_stanalone_accuracy(model_name)