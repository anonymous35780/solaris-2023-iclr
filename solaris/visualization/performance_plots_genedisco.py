"""
Copyright 2022 Pascal Notin, University of Oxford
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import ast 

result_location = 'output/Sep29_final_plots.csv' 
result_location_processed = 'output/Sep29_final_plots_processed.csv'
fig_suffix="_Sep29_final_plots_main_all_combined"
plot_all_on_1_fig = True

plt.rcParams.update({'font.size': 18})

def main():
    try:
        performance_data = pd.read_csv(result_location, low_memory=False)
        print("Loaded without issue")
    except:
        print("Need to escape the lists in file")
        with open(result_location, "r") as f:
            lines = f.readlines()
        for line_index,line in enumerate(lines):
            lines[line_index] = line.replace('[','"[').replace(']',']"')
        with open(result_location_processed, "w") as f:
            for line in lines:
                f.write(line)
        performance_data = pd.read_csv(result_location_processed, low_memory=False)

    performance_data = performance_data.loc[performance_data.acquisition_cycle==25, :]
    print("Check completion rate")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(performance_data.groupby(['feature_set_name','dataset_name','acquisition_function_name']).size())
    methods = ["random",
                "topk_bax",
                "levelset_bax",
                "subsetmax_bax_additive",
                #"subsetmax_bax_multiplicative",
                "ucb",
                "thompson_sampling",
                "topuncertain",
                "softuncertain",
                "marginsample",
                "badge",
                "coreset",
                "kmeans_embedding",
                "kmeans_data",
                "adversarialBIM"]
    #methods=[
    #    "subsetmax_bax_additive",
    #    "topk_bax",
    #    "levelset_bax",
    #    "ucb",
    #    "thompson_sampling",
    #    "coreset",
    #    "random"
    #]
    #feature_sets = ["achilles",
    #                "ccle",
    #                "string"]
    feature_sets = ["achilles"]
    target_datasets = ["schmidt_2021_ifng",
                        "schmidt_2021_il2",
                        "zhuang_2019_nk",
                        "sanchez_2021_tau",
                        "zhu_2021_sarscov2"]
    #target_datasets = [
    #    "schmidt_2021_ifng",
    #    "zhuang_2019_nk"
    #]
    seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    pretty_method_names = {"random":"Random",
                            "topk_bax":"Top-K BAX",
                            "levelset_bax":"Levelset BAX",
                            "subsetmax_bax_additive":"DiscoBAX",
                            "subsetmax_bax_multiplicative":"DiscoBAX_mult",
                            "ucb":"UCB",
                            "thompson_sampling":"Thompson sampling",
                            "topuncertain":"Top uncertainty",
                            "softuncertain":"Soft uncertainty",
                            "marginsample":"Margin sample",
                            "badge":"BADGE",
                            "coreset":"Coreset",
                            "kmeans_embedding":"Kmeans embedding",
                            "kmeans_data":"Kmeans data",
                            "adversarialBIM":"Adversarial BIM"
    }
    pretty_dataset_names = {"schmidt_2021_ifng":"Interferon Gamma",
                        "schmidt_2021_il2":"Interleukin 2",
                        "zhuang_2019_nk":"Leukemia / NK cells",
                        "sanchez_2021_tau": "Tau protein",
                        "zhu_2021_sarscov2":"SARS-CoV-2"

    }

    sns.set_style("whitegrid")
    if plot_all_on_1_fig:
        for feature_set in feature_sets:
            fig, axs = plt.subplots(len(target_datasets), 2, figsize=(22, 32))

            for dataset_index,target_dataset in enumerate(target_datasets):
                for m_index, m in enumerate(methods):
                    agg_topk_precision_all_seeds=[]
                    agg_topk_recall_all_seeds=[]
                    agg_topk_cluster_recall_all_seeds=[]
                    for seed in seeds:
                        try:
                            data_point = performance_data.loc[(performance_data.feature_set_name==feature_set)&(performance_data.dataset_name==target_dataset)&(performance_data.acquisition_function_name==m)&(performance_data.seed==seed)]
                            assert len(data_point)<=1, "Duplicate runs found in the performance data for params: {} {} {} {}".format(feature_set,target_dataset,m,seed)
                            assert len(data_point)>=1, "Seed run {} {} {} {} could not be found".format(feature_set,target_dataset,m,seed)
                            num_results = len(data_point)
                            if num_results > 1:
                                print("Duplicate runs found in the performance data for params: {} {} {} {}".format(feature_set,target_dataset,m,seed))
                                print(data_point)
                            while(num_results>0):
                                agg_topk_precision_all_seeds.append(ast.literal_eval(data_point['cumulative_precision_topk'].values[num_results-1]))
                                agg_topk_recall_all_seeds.append(ast.literal_eval(data_point['cumulative_recall_topk'].values[num_results-1]))
                                agg_topk_cluster_recall_all_seeds.append(ast.literal_eval(data_point['cumulative_proportion_top_clusters_recovered'].values[num_results-1]))
                                num_results-=1
                        except:
                            pass
                    num_seeds_completed = len(agg_topk_precision_all_seeds)
                    #mean_agg_topk_precision_all_seeds = np.mean(np.array(agg_topk_precision_all_seeds), axis=0)
                    mean_agg_topk_recall_all_seeds = np.mean(np.array(agg_topk_recall_all_seeds), axis=0)
                    mean_agg_topk_cluster_recall_all_seeds = np.mean(np.array(agg_topk_cluster_recall_all_seeds), axis=0)
                    #stde_agg_topk_precision_all_seeds = np.std(np.array(agg_topk_precision_all_seeds), axis=0) / (num_seeds_completed)**0.5
                    stde_agg_topk_recall_all_seeds = np.std(np.array(agg_topk_recall_all_seeds), axis=0) / (num_seeds_completed)**0.5
                    stde_agg_topk_cluster_recall_all_seeds = np.std(np.array(agg_topk_cluster_recall_all_seeds), axis=0) / (num_seeds_completed)**0.5
                    
                    if (m=='subsetmax_bax_additive'):
                        print(mean_agg_topk_cluster_recall_all_seeds)
                    #axs[dataset_index][0].plot(mean_agg_topk_precision_all_seeds, label=m)
                    #axs[dataset_index][0].fill_between(range(len(mean_agg_topk_precision_all_seeds)), mean_agg_topk_precision_all_seeds - stde_agg_topk_precision_all_seeds, mean_agg_topk_precision_all_seeds + stde_agg_topk_precision_all_seeds , alpha=0.2)
                    linestyle='dashed' if m_index >= 10 else 'solid'
                    axs[dataset_index][0].plot(mean_agg_topk_recall_all_seeds, label=pretty_method_names[m],linestyle=linestyle)
                    axs[dataset_index][0].fill_between(range(len(mean_agg_topk_recall_all_seeds)), mean_agg_topk_recall_all_seeds - stde_agg_topk_recall_all_seeds, mean_agg_topk_recall_all_seeds + stde_agg_topk_recall_all_seeds , alpha=0.2)
                    axs[dataset_index][1].plot(mean_agg_topk_cluster_recall_all_seeds, label=pretty_method_names[m],linestyle=linestyle)
                    axs[dataset_index][1].fill_between(range(len(mean_agg_topk_cluster_recall_all_seeds)), mean_agg_topk_cluster_recall_all_seeds - stde_agg_topk_cluster_recall_all_seeds, mean_agg_topk_cluster_recall_all_seeds + stde_agg_topk_cluster_recall_all_seeds , alpha=0.2)

                
                #axs[dataset_index][0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
                #axs[dataset_index][1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
                
                axs[dataset_index][0].set_yticklabels(['{:,.0%}'.format(x) for x in axs[dataset_index][0].get_yticks()])
                axs[dataset_index][1].set_yticklabels(['{:,.0%}'.format(x) for x in axs[dataset_index][1].get_yticks()])
                axs[dataset_index][0].title.set_text(pretty_dataset_names[target_dataset])
                axs[dataset_index][1].title.set_text(pretty_dataset_names[target_dataset])
                #axs[dataset_index][2].title.set_text(target_dataset)
                axs[dataset_index][0].set_xlabel("Active learning cycles")
                axs[dataset_index][1].set_xlabel("Active learning cycles")
                #axs[dataset_index][2].set_xlabel("Active learning cycles")
                #axs[dataset_index][0].set_ylabel("Cumulative Top-K precision")
                axs[dataset_index][0].set_ylabel("Cumulative Top-K recall")
                axs[dataset_index][1].set_ylabel("Cumulative Diversity score")
            
            axs[2][1].legend(
            loc="center right",   # Position of legend
            bbox_to_anchor=(1.55, 0.5),
            borderaxespad=0.1,    # Small spacing around legend box
            title="Acquisition methods",  # Title for the legend
            fontsize=14
            )
            #fig.suptitle(f"Performance of acquisition methods Vs # of cycles",fontsize=16)
            fig.tight_layout()
            plt.subplots_adjust(
                    wspace=0.25, 
                    hspace=0.3)
            plt.savefig(f"Performance_results_feature_set_{feature_set}{fig_suffix}.png")
            plt.clf()
    else:
        for feature_set in feature_sets:
            for dataset_index,target_dataset in enumerate(target_datasets):
                fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
                for m_index, m in enumerate(methods):
                    agg_topk_precision_all_seeds=[]
                    agg_topk_recall_all_seeds=[]
                    agg_topk_cluster_recall_all_seeds=[]
                    for seed in seeds:
                        try:
                            data_point = performance_data.loc[(performance_data.feature_set_name==feature_set)&(performance_data.dataset_name==target_dataset)&(performance_data.acquisition_function_name==m)&(performance_data.seed==seed)]
                            assert len(data_point)<=1, "Duplicate runs found in the performance data for params: {} {} {} {}".format()
                            assert len(data_point)>=1, "Seed run {} {} {} {} could not be found".format()
                            agg_topk_precision_all_seeds.append(ast.literal_eval(data_point['cumulative_precision_topk'].values[0]))
                            agg_topk_recall_all_seeds.append(ast.literal_eval(data_point['cumulative_recall_topk'].values[0]))
                            agg_topk_cluster_recall_all_seeds.append(ast.literal_eval(data_point['cumulative_proportion_top_clusters_recovered'].values[0]))
                        except:
                            pass
                    num_seeds_completed = len(agg_topk_precision_all_seeds)
                    
                    #mean_agg_topk_precision_all_seeds = np.mean(np.array(agg_topk_precision_all_seeds), axis=0)
                    mean_agg_topk_recall_all_seeds = np.mean(np.array(agg_topk_recall_all_seeds), axis=0)
                    mean_agg_topk_cluster_recall_all_seeds = np.mean(np.array(agg_topk_cluster_recall_all_seeds), axis=0)
                    #stde_agg_topk_precision_all_seeds = np.std(np.array(agg_topk_precision_all_seeds), axis=0) / (num_seeds_completed)**0.5
                    stde_agg_topk_recall_all_seeds = np.std(np.array(agg_topk_recall_all_seeds), axis=0) / (num_seeds_completed)**0.5
                    stde_agg_topk_cluster_recall_all_seeds = np.std(np.array(agg_topk_cluster_recall_all_seeds), axis=0) / (num_seeds_completed)**0.5
                    
                    #axs[dataset_index][0].plot(mean_agg_topk_precision_all_seeds, label=m)
                    #axs[dataset_index][0].fill_between(range(len(mean_agg_topk_precision_all_seeds)), mean_agg_topk_precision_all_seeds - stde_agg_topk_precision_all_seeds, mean_agg_topk_precision_all_seeds + stde_agg_topk_precision_all_seeds , alpha=0.2)
                    linestyle='dashed' if m_index >= 10 else 'solid'
                    axs[0].plot(mean_agg_topk_recall_all_seeds, label=pretty_method_names[m],linestyle=linestyle)
                    axs[0].fill_between(range(len(mean_agg_topk_recall_all_seeds)), mean_agg_topk_recall_all_seeds - stde_agg_topk_recall_all_seeds, mean_agg_topk_recall_all_seeds + stde_agg_topk_recall_all_seeds , alpha=0.2)
                    axs[1].plot(mean_agg_topk_cluster_recall_all_seeds, label=pretty_method_names[m],linestyle=linestyle)
                    axs[1].fill_between(range(len(mean_agg_topk_cluster_recall_all_seeds)), mean_agg_topk_cluster_recall_all_seeds - stde_agg_topk_cluster_recall_all_seeds, mean_agg_topk_cluster_recall_all_seeds + stde_agg_topk_cluster_recall_all_seeds , alpha=0.2)

                #axs[0].title.set_text(target_dataset)
                #axs[1].title.set_text(target_dataset)
                axs[0].set_yticklabels(['{:,.0%}'.format(x) for x in axs[0].get_yticks()])
                axs[1].set_yticklabels(['{:,.0%}'.format(x) for x in axs[1].get_yticks()])
                axs[0].set_xlabel("Active learning cycles")
                axs[1].set_xlabel("Active learning cycles")
                axs[0].set_ylabel("Cumulative Top-K recall")
                axs[1].set_ylabel("Cumulative Diversity score")
            
                axs[1].legend(
                loc="center right",   # Position of legend
                bbox_to_anchor=(1.85, 0.55),
                borderaxespad=0.1,    # Small spacing around legend box
                title="Acquisition methods",  # Title for the legend
                fontsize=13
                )
                #fig.suptitle(f"Performance of acquisition methods Vs # of cycles",fontsize=16)
                fig.tight_layout()
                plt.savefig(f"Performance_results_feature_set_{feature_set}_dataset_{target_dataset}{fig_suffix}.png")
                plt.clf()

if __name__ == "__main__":
    main()
