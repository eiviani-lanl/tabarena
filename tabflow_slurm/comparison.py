import pandas as pd
import numpy as np

'''elm_df = pd.read_csv('/vast/home/eiviani/tabarena/tabflow_slurm/evals/custom_elm/all/without/results_per_split.csv')
rvfl_df = pd.read_csv('/vast/home/eiviani/tabarena/tabflow_slurm/evals/custom_elm/all/with/results_per_split.csv')

elm_df['method'] = elm_df['method'].str.replace('gfdl', 'elm', case=False)
rvfl_df['method'] = rvfl_df['method'].str.replace('gfdl', 'rvfl', case=False)
print(elm_df.columns)

all_data = pd.concat([elm_df, rvfl_df], ignore_index=True)'''

all_data = pd.read_csv('/vast/home/eiviani/tabarena/tabflow_slurm/evals/custom_elm/all/results_per_split.csv')

agg_dict = {
    'time_train_s': 'mean',
    'time_infer_s': 'mean',
    'metric_error': 'mean'
}

agg_dict['metric_error_val'] = 'mean'

method_avg = all_data.groupby(['method','problem_type']).agg(agg_dict).reset_index()

tuned_methods = method_avg[method_avg['method'].str.contains('(tuned + ensemble)', case=False, regex=False)].copy()


all_results = []

for prob_type in ['binary','multiclass','regression']:
    print(f"Problem type: {prob_type}")
    
    prob_data = tuned_methods[tuned_methods['problem_type'] == prob_type].copy()
    
    prob_data['error_rank'] = prob_data['metric_error'].rank(method='average', ascending=True).astype(int)
    prob_data['val_error_rank'] = prob_data['metric_error_val'].rank(method='average', ascending=True).astype(int)
    
    prob_data = prob_data.sort_values('metric_error')
    
    print(f"Error averaged across all datasets and folds for {prob_type}")
    
    display_df = prob_data[['method', 'metric_error', 'error_rank', 
                                'metric_error_val', 'val_error_rank']].copy()
    display_df.columns = ['Method', 'Test error', 'Test rank', 'Val error', 'Val rank']
    
    print(display_df.to_string(index=False))
    
    prob_data['train_rank'] = prob_data['time_train_s'].rank(method='average', ascending=True).astype(int)
    prob_data['infer_rank'] = prob_data['time_infer_s'].rank(method='average', ascending=True).astype(int)
    
    prob_data = prob_data.sort_values('time_train_s')
    
    print(f"Times averaged across all datasets and folds for {prob_type}")
    
    display_df = prob_data[['method', 'time_train_s', 'train_rank', 
                                'time_infer_s', 'infer_rank']].copy()
    display_df.columns = ['Method', 'Train time (s)', 'Train Rank', 'Infer time (s)', 'Infer Rank']
    
    print(display_df.to_string(index=False))