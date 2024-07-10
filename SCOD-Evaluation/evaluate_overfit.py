from selective_function.LinearDoubleScore import LinearDoubleScore
from selective_function.SIRCDoubleScore import SIRCDoubleScore
from helpers.visualization import plot_scores
from helpers.utils import load_result, get_results
from helpers.constants import N_A_SAMPLES, N_B_SAMPLES, N_ALPHA_SAMPLES
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
import itertools
import time
import glob
import os

plt.style.use('sequential.mplstyle')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default='OpenOOD/results/imagenet_resnet50_tvsv1_base_default',
                        help="Directory name of the experiment.", type=str)
    parser.add_argument("--id_dataset", default='imagenet',
                        help="In distribution dataset name.", type=str)
    parser.add_argument("--main_method", default='msp',
                        help="Primary method to combine with others.")
    parser.add_argument('--ignored_methods', default=[], nargs='*',
                        help="Methods to not combine with the default one.")
    parser.add_argument("--beta", default=0.5,
                        help="Beta parameter of Augmented-Risk. Selective risk is multiplied by beta. FPR is multiplied by (1-beta).", type=float)
    parser.add_argument("--tpr", default=0.95,
                        help="TPR to evaluate metrics at, e.g., FPR@95TPR.", type=float)
    parser.add_argument("--multisplit", default=False, action='store_true', help="Signifies whether the experiment uses splits 's0', 's1' and 's3'.")
    parser.add_argument("--output_dir", default="output/", help="Directory to which results are exported.")
    parser.add_argument("--seed", default=123, help="Seed for random sampling of train/test data.")
    parser.add_argument("--fit_percentage", default=10, help="Percentage of OOD samples to use for fitting of the selective function.")

    args = parser.parse_args()
    
    np.random.seed(args.seed)

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    base_dir = args.base_dir
    id_dataset = args.id_dataset
    ignored_methods = args.ignored_methods + [args.main_method] # No reason to combine the main method with itself, add it to ignored.
    main_method = args.main_method
    beta = args.beta
    at_tpr = args.tpr
    multisplit = args.multisplit
    fit_percentage = args.fit_percentage

    splits = ['s0', 's1', 's2'] if multisplit else ['']
    base_result = load_result(f'{base_dir}/s0/scores/{main_method}.pkl') if multisplit else load_result(f'{base_dir}/scores/{main_method}.pkl')

    ood_datasets = []
    for nearfar in base_result['ood']:
        if base_result['ood'][nearfar] is not None:
            for dataset in base_result['ood'][nearfar]:
                ood_datasets.append((nearfar, dataset))

    print(
        f"For ID dataset {id_dataset}, the following OOD datasets were found:")
    for nearfar, ood_dataset in ood_datasets:
        print(f"\t {nearfar} \t {ood_dataset}")

    results = {}

    for split in splits:
        split_name = '_split_' + split if multisplit else ''
        scores_root = f'{base_dir}/{split}/scores/' if multisplit else f'{base_dir}/scores/'
        methods = [os.path.splitext(os.path.basename(i))[0]
               for i in glob.glob(scores_root + '*.pkl')]
        
        methods = [method for method in methods if 'lhood' not in method]
        
        if id_dataset == 'imagenet':        
            methods += ['lhood', 'mixture_lhood']

        elif id_dataset == 'gauss':
            methods += ['lhood_ratio', 'lhood_id']
        
        for nearfar, ood_dataset in tqdm(ood_datasets, desc="dataset", position=0):

            results[ood_dataset] = {} 

            for method in tqdm(methods, desc="method", position=1, leave=False):
                if method in ignored_methods:
                    continue
                
                y_pred_1, score_1, y_true_1 = get_results(os.path.join(scores_root, main_method + '.pkl'), nearfar=nearfar, ood_dataset=ood_dataset)

                if 'lhood' in method:
                    scores_path = os.path.join(
                        scores_root, ood_dataset + "_" + method + '.pkl')
                    if not os.path.exists(scores_path):
                        continue
                    y_pred_2, score_2, y_true_2, is_test = get_results(
                        scores_path, nearfar=nearfar, ood_dataset=ood_dataset, filter_paths=os.path.join(scores_root, ood_dataset + '_' + method + '_paths.json'))
                    
                    # Filter out OOD samples used for training of the score
                    y_pred_1 = y_pred_1[is_test]
                    y_true_1 = y_true_1[is_test]
                    score_1 = score_1[is_test]
                    y_pred_2 = y_pred_2[is_test]
                    y_true_2 = y_true_2[is_test]
                    score_2 = score_2[is_test]

                else:
                    y_pred_2, score_2, y_true_2 = get_results(os.path.join(
                        scores_root, method + '.pkl'), nearfar=nearfar, ood_dataset=ood_dataset)

                assert np.all(y_true_1 == y_true_2)
                
                results[ood_dataset][method] = {}
                
                # Get ID and OOD indexes
                ID_ixs = np.where((y_true_1 != -1))[0]
                OOD_ixs = np.where((y_true_1 == -1))[0]
                # Separate the OOD indexes into validation (for fitting selective function) and test (for evaluation) part
                np.random.shuffle(OOD_ixs)
                nr_ood_samples = len(OOD_ixs)
                nr_val_ood_samples = int(nr_ood_samples * fit_percentage / 100)
                val_OOD_ixs = OOD_ixs[:nr_val_ood_samples]
                test_OOD_ixs = OOD_ixs[nr_val_ood_samples:]
                # Separate the data into validation and test part; note that the ID data are used in full for both validation and testing
                val_ixs = np.concatenate([ID_ixs, val_OOD_ixs])
                test_ixs = np.concatenate([ID_ixs, test_OOD_ixs])

                # Ininitialize the selective functions, with validation data
                linear = LinearDoubleScore(y_pred=y_pred_1[val_ixs], y_true=y_true_1[val_ixs], score_1=score_1[val_ixs], score_2=score_2[val_ixs])
                linear_default = LinearDoubleScore(y_pred=y_pred_1[val_ixs], y_true=y_true_1[val_ixs], score_1=score_1[val_ixs], score_2=score_2[val_ixs], fast=True)
                sirc = SIRCDoubleScore(y_pred=y_pred_1[val_ixs], y_true=y_true_1[val_ixs], score_1=score_1[val_ixs], score_2=score_2[val_ixs])
                sirc_default = SIRCDoubleScore(y_pred=y_pred_1[val_ixs], y_true=y_true_1[val_ixs], score_1=score_1[val_ixs], score_2=score_2[val_ixs], default=True)

                # Fit the selective functions to minimize the joint risk at @TPR
                linear.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_alpha_samples=N_ALPHA_SAMPLES, beta=beta)                
                linear_default.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_alpha_samples=N_ALPHA_SAMPLES, beta=beta)                
                sirc.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES, beta=beta)
                sirc_default.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES, beta=beta)


                # Evaluate the selective functions on the validation data
                evaluation = linear.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['linear|val|val|'+key] = value
                evaluation = linear_default.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['linear_default|val|val|'+key] = value
                evaluation = sirc.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc|val|val|'+key] = value
                evaluation = sirc_default.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc_default|val|val|'+key] = value

                # Evaluate the selective functions on the test data
                evaluation = linear.evaluate(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs])
                for key, value in evaluation.items(): results[ood_dataset][method]['linear|val|test|'+key] = value
                evaluation = linear_default.evaluate(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs])
                for key, value in evaluation.items(): results[ood_dataset][method]['linear_default|val|test|'+key] = value
                evaluation = sirc.evaluate(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs])
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc|val|test|'+key] = value
                evaluation = sirc_default.evaluate(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs])
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc_default|val|test|'+key] = value

                # Ininitialize the selective functions, with test data
                linear = LinearDoubleScore(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs])
                linear_default = LinearDoubleScore(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs], fast=True)
                sirc = SIRCDoubleScore(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs])
                sirc_default = SIRCDoubleScore(y_pred=y_pred_1[test_ixs], y_true=y_true_1[test_ixs], score_1=score_1[test_ixs], score_2=score_2[test_ixs], default=True)
                
                # Fit the selective functions to minimize the joint risk at @TPR
                linear.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_alpha_samples=N_ALPHA_SAMPLES, beta=beta)                
                linear_default.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_alpha_samples=N_ALPHA_SAMPLES, beta=beta)                
                sirc.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES, beta=beta)
                sirc_default.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES, beta=beta)
                
                # Evaluate the selective functions on the test data
                evaluation = linear.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['linear|test|test|'+key] = value
                evaluation = linear_default.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['linear_default|test|test|'+key] = value
                evaluation = sirc.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc|test|test|'+key] = value
                evaluation = sirc_default.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc_default|test|test|'+key] = value
                                
                # Ininitialize the selective functions, with all data
                linear = LinearDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)
                linear_default = LinearDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2, fast=True)
                sirc = SIRCDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)
                sirc_default = SIRCDoubleScore(y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2, default=True)
                
                # Fit the selective functions to minimize the joint risk at @TPR
                linear.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_alpha_samples=N_ALPHA_SAMPLES, beta=beta)                
                linear_default.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_alpha_samples=N_ALPHA_SAMPLES, beta=beta)                
                sirc.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES, beta=beta)
                sirc_default.fit(minimize=True, optimization_objective = 'joint risk', optimization_condition='tpr', min_tpr = at_tpr, n_a_samples=N_A_SAMPLES, n_b_samples=N_B_SAMPLES, beta=beta)
                
                # Evaluate the selective functions on all data
                evaluation = linear.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['linear|all|all|'+key] = value
                evaluation = linear_default.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['linear_default|all|all|'+key] = value
                evaluation = sirc.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc|all|all|'+key] = value
                evaluation = sirc_default.self_evaluate()
                for key, value in evaluation.items(): results[ood_dataset][method]['sirc_default|all|all|'+key] = value
                            
            pd.DataFrame.from_dict(results[ood_dataset]).to_csv(os.path.join(args.output_dir, f"overfitting_{id_dataset}_{ood_dataset}_results{split_name}_{now}.csv"))

        with open(os.path.join(args.output_dir, f"overfitting_{id_dataset}_{ood_dataset}_results{split_name}_{now}.pickle"), "wb") as output_file:
            pickle.dump(results, output_file)