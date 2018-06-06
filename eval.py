import sed_eval

module_name  = input("test name(e.g. 20180422_01_glass):")
truth_file = "./event_list/model_"+ module_name+'_test_truth.txt'
result_file= "./event_list/model_"+module_name+'_test.txt'

file_list = [
    {
     'reference_file': truth_file,
     'estimated_file': result_file
    }
]

data = []

all_data = sed_eval.io.load_event_list(file_list[0]['reference_file'])

for file_pair in file_list:
    reference_event_list = sed_eval.io.load_event_list(file_pair['reference_file'])
    estimated_event_list = sed_eval.io.load_event_list(file_pair['estimated_file'])
    data.append({'reference_event_list': reference_event_list,
                 'estimated_event_list': estimated_event_list})
    all_data += reference_event_list
    
# Get used event labels
event_labels = all_data.unique_event_labels

# Start evaluating

# Create metrics classes, define parameters
event_based_metrics = sed_eval.sound_event.EventBasedMetrics(event_label_list=event_labels,evaluate_onset=True, evaluate_offset=False, t_collar=0.5)

# Go through files
for file_pair in data:
    event_based_metrics.evaluate(file_pair['reference_event_list'],
                                 file_pair['estimated_event_list'])


# Or print all metrics as reports
savefile = print(event_based_metrics)

with open("./event_list/model_"+module_name+"_eval.out","w") as f:
	print(event_based_metrics,file=f)


