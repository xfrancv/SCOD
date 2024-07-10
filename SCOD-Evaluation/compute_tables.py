from selective_function.LinearDoubleScore import LinearDoubleScore
from selective_function.SIRCDoubleScore import SIRCDoubleScore
from selective_function.MultiplicationDoubleScore import MultiplicationDoubleScore
from helpers.visualization import plot_scores
from helpers.utils import load_result, get_results, get_paths
from helpers.constants import N_A_SAMPLES, N_B_SAMPLES, N_ALPHA_SAMPLES, BETA
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
    parser.add_argument("--multisplit", default=False, action='store_true')
    parser.add_argument("--likelihood", default=False, action='store_true')
    parser.add_argument("--output_dir", default="output/results/")

    args = parser.parse_args()

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    base_dir = args.base_dir
    id_dataset = args.id_dataset
    ignored_methods = args.ignored_methods + [args.main_method]
    main_method = args.main_method
    multisplit = args.multisplit
    ignore_likelihood = not args.likelihood
    
    os.makedirs(args.output_dir, exist_ok=True)

    splits = ['s0', 's1', 's2'] if multisplit else ['']
    base_result = load_result(
        f'{base_dir}/s0/scores/{main_method}.pkl') if multisplit else load_result(f'{base_dir}/scores/{main_method}.pkl')

    ood_datasets = []
    for nearfar in base_result['ood']:
        if base_result['ood'][nearfar] is not None:
            for dataset in base_result['ood'][nearfar]:
                if 'paths' not in dataset:
                    ood_datasets.append((nearfar, dataset))

    print(f"For ID dataset {id_dataset}, the following OOD datasets were found:")
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
            if not ignore_likelihood:
                methods += ['mix_sigmoid_lhood', 
                            'mix_fixed_sigmoid_lhood', 
                            'sigmoid_lhood']

        elif id_dataset == 'gauss':
            methods += ['lhood_ratio', 
                        'lhood_id']

        elif id_dataset == 'cifar10':
            if not ignore_likelihood:
                methods += ['cifar10_sigmoid_lhood',
                            'cifar10_mix_fixed_sigmoid_lhood']
            
        elif id_dataset == 'cifar100':
            if not ignore_likelihood:
                methods = ['cifar100_sigmoid_lhood', 
                        'cifar100_mix_fixed_sigmoid_lhood']
                
        print("Processing the following methods ...")
        for method in methods:
            print(f"\t {method}")

        for nearfar, ood_dataset in tqdm(ood_datasets, desc="Dataset", position=0):
            results[ood_dataset] = {}

            for method in tqdm(methods, desc="Method", position=1, leave=False):
                if method in ignored_methods:
                    continue

                y_pred_1, score_1, y_true_1 = get_results(os.path.join(
                    scores_root, main_method + '.pkl'), nearfar=nearfar, ood_dataset=ood_dataset)

                # The likelihood ratio score needs to be handled differently, specifically
                # one needs to also consider what samples the model was trained on and omit them from evaluation.
                # For this purpose, a JSON with the data split definitions is passed to `filter_paths` and the
                # appropriate samples are filtered out using the `is_test` boolean array.
                if 'lhood' in method and id_dataset != 'gauss':
                    # For CIFAR-10/100, the name of the exported results contains the number of the split. 
                    # This requires us to process CIFAR results a bit differently than ImageNet-1K
                    if 'cifar' in id_dataset and 'sigmoid_lhood' in method:
                        first, second = method.split('sigmoid')
                        method_to_load = first + f'sigmoid_fold_{int(split[1])}' + second
                    else:
                        method_to_load = method
                    
                    scores_path = os.path.join(
                        scores_root, ood_dataset + "_" + method_to_load + '.pkl')                    

                    if not os.path.exists(scores_path):
                        print(f"Could not find {scores_path}!")
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

                linear = LinearDoubleScore(
                    y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)
                linear_plugin = LinearDoubleScore(
                    y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2, fast=True)
                sirc = SIRCDoubleScore(
                    y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)
                sirc_plugin = SIRCDoubleScore(
                    y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2, default=True)
                multiplication = MultiplicationDoubleScore(
                    y_pred=y_pred_1, y_true=y_true_1, score_1=score_1, score_2=score_2)

                # SIRC
                sirc_risk, sirc_tpr, sirc_auc = sirc.compute_joint_risk_vs_tpr_curve()
                # SIRC plugin
                sirc_plugin.fit(minimize=True, optimization_objective='joint risk', optimization_condition='tpr', min_tpr=0.99)
                sirc_plugin_risk, sirc_plugin_tpr, sirc_plugin_auc = sirc_plugin.compute_joint_risk_vs_tpr_curve()
                # Linear
                linear_risk, linear_tpr, linear_auc = linear.compute_joint_risk_vs_tpr_curve()
                # Linear plugin
                linear_plugin_risk, linear_plugin_tpr, linear_plugin_auc = linear_plugin.compute_joint_risk_vs_tpr_curve()
                # Multiplication
                multiplication_risk, multiplication_tpr, multiplication_auc = multiplication.compute_joint_risk_vs_tpr_curve()
                
                metrics_1 = linear.get_score_1_metrics()
                metrics_2 = linear.get_score_2_metrics()

                results[ood_dataset][method] = {'score_1': main_method,
                                                'score_2': method,
                                                'beta': BETA,
                                                'score_1_AUROC': metrics_1.AUROC,
                                                'score_1_AURC': metrics_1.AURC,
                                                'score_1_AUSCOD': metrics_1.AUJR,
                                                'score_2_AUROC': metrics_2.AUROC,
                                                'score_2_AURC': metrics_2.AURC,
                                                'score_2_AUSCOD': metrics_2.AUJR,
                                                'SIRC-plugin-AUSCOD': sirc_plugin_auc,
                                                'SIRC-tuned-AUSCOD': sirc_auc,
                                                'Linear-tuned-AUSCOD': linear_auc,
                                                'Linear-plugin-AUSCOD': linear_plugin_auc,
                                                'multiplication-AUSCOD': multiplication_auc}

            pd.DataFrame.from_dict(results[ood_dataset]).to_csv(os.path.join(
                args.output_dir, f"{id_dataset}_{ood_dataset}_results{split_name}_{now}.csv"))

        with open(os.path.join(args.output_dir, f"{id_dataset}_results{split_name}_{now}.pickle"), "wb") as output_file:
            pickle.dump(results, output_file)
