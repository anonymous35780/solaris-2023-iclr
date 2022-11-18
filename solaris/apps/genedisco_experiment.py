"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc, Clare Lyle, University of Oxford, Pascal Notin, University of Oxford
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

import pickle as pkl
import os
from collections import namedtuple
import random
import numpy as np
import pandas as pd
from typing import AnyStr, Dict, List, Optional
from collections import defaultdict

import slingpy as sp
from slingpy.utils.logging import info
#from slingpy.evaluation.evaluator import Evaluator
from slingpy.utils.path_tools import PathTools
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel

from solaris.apps import abstract_base_application as aba
from solaris.apps.genedisco_single_cycle_experiment import SingleCycleApplication
from solaris.models import clustering
from solaris.methods.entropic_ucb import ucb_sampling
from solaris.methods.bax_acquisition import bax_sampling_genedisco
from solaris.methods.thompson_sampling import thompson_sampling_genedisco
from solaris.methods.random import random
from solaris.methods.active_learning.kmeans import Kmeans
from solaris.methods.active_learning.core_set import CoreSet
from solaris.methods.active_learning.badge_sampling import BadgeSampling
from solaris.methods.active_learning.adversarial_BIM import AdversarialBIM
from solaris.methods.active_learning.top_soft_uncertainty import TopUncertainAcquisition
from solaris.methods.active_learning.top_soft_uncertainty import SoftUncertainAcquisition
from solaris.methods.active_learning.margin_sampling import MarginSamplingAcquisition
from solaris.methods.active_learning.base_acquisition_function import BaseBatchAcquisitionFunction

DataSet = namedtuple(
    "DataSet",
    "training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y",
)

class GeneDiscoLoop(aba.AbstractBaseApplication):
    
    def __init__(
        self,
        feature_set_name: AnyStr,
        dataset_name: AnyStr,
        cache_directory: AnyStr,
        output_directory: AnyStr,
        model_name: AnyStr = "gp",
        acquisition_function_name: AnyStr = "random",
        acquisition_function_path: AnyStr = "custom",
        acquisition_batch_size: int = 20,
        num_active_learning_cycles: int = 20,
        test_ratio: float = 0.0,
        hyperopt_children: bool = False,
        schedule_on_slurm: bool = False,
        schedule_children_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 1,
        remote_execution_mem_limit_in_mb: int = 5048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        seed: int = 1000,
        topk_percent: float = 0.1,
        performance_file_location: str = "output/performance_file.csv",
        bax_topk_kvalue: int = 3,
        bax_subset_select_subset_size: int = 5,
        bax_subset_level_c: float = 1.0,
        bax_batch_selection_mode: str = 'topk_EIG',
        num_topk_clusters: int = 20
    ):
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_path = acquisition_function_path
        self.run_name = '_'.join([dataset_name,feature_set_name,str(topk_percent),model_name,acquisition_function_name,str(seed)])
        PathTools.mkdir_if_not_exists(output_directory)
        output_directory = output_directory+os.sep+self.run_name
        PathTools.mkdir_if_not_exists(output_directory)
        self.temp_folder_name = output_directory+os.sep+'tmp'
        PathTools.mkdir_if_not_exists(self.temp_folder_name)
        self.performance_file_location = performance_file_location
        self.bax_topk_kvalue = bax_topk_kvalue
        self.bax_subset_select_subset_size = bax_subset_select_subset_size
        self.bax_subset_level_c = bax_subset_level_c
        self.bax_batch_selection_mode = bax_batch_selection_mode
        self.acquisition_function = self.get_acquisition_function(
            self.acquisition_function_name, self.acquisition_function_path, self.bax_topk_kvalue, self.bax_subset_level_c, self.bax_subset_select_subset_size, self.bax_batch_selection_mode
        )
        self.acquisition_batch_size = acquisition_batch_size
        self.num_active_learning_cycles = num_active_learning_cycles
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.model_name = model_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.cache_directory = cache_directory
        self.schedule_children_on_slurm = schedule_children_on_slurm
        self.topk_percent = topk_percent
        self.topk_indices = None
        self.topk_clusters_to_item = None
        self.topk_item_to_clusters = None
        self.num_topk_clusters = num_topk_clusters
        super(GeneDiscoLoop, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=False,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
            schedule_on_slurm=schedule_on_slurm,
            remote_execution_num_cpus=remote_execution_num_cpus,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path,
        )

    def get_acquisition_function(self, 
                                acquisition_function_name: AnyStr, 
                                acquisition_function_path: AnyStr, 
                                bax_topk_kvalue: int = 3, 
                                bax_subset_level_c: float = 1.0, 
                                bax_subset_select_subset_size: int = 5,
                                bax_batch_selection_mode: str = 'topk_EIG'
                ) -> BaseBatchAcquisitionFunction:

        if acquisition_function_name == "random":
            return random.RandomBatchAcquisitionFunction()
        elif acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        elif acquisition_function_name == "topk_bax":
            return bax_sampling_genedisco.BaxAcquisition(objective_function="topk", k = bax_topk_kvalue)
        elif acquisition_function_name == "levelset_bax":
            return bax_sampling_genedisco.BaxAcquisition(objective_function="level_set", c = bax_subset_level_c)
        elif acquisition_function_name == "subsetmax_bax_additive":
            return bax_sampling_genedisco.BaxAcquisition(objective_function="subset_select", noise_type="additive", subset_size=bax_subset_select_subset_size, batch_selection_mode=bax_batch_selection_mode)
        elif acquisition_function_name == "subsetmax_bax_multiplicative":
            return bax_sampling_genedisco.BaxAcquisition(objective_function="subset_select", noise_type="multiplicative", subset_size=bax_subset_select_subset_size, batch_selection_mode=bax_batch_selection_mode)
        elif acquisition_function_name == "thompson_sampling":
            return thompson_sampling_genedisco.ThompsonSamplingcquisition()
        elif acquisition_function_name == "topuncertain":
            return TopUncertainAcquisition()
        elif acquisition_function_name == "softuncertain":
            return SoftUncertainAcquisition()
        elif acquisition_function_name == "marginsample":
            return MarginSamplingAcquisition()
        elif acquisition_function_name == "badge":
            return BadgeSampling()
        elif acquisition_function_name == "coreset":
            return CoreSet()
        elif acquisition_function_name == "kmeans_embedding":
            return Kmeans(representation="linear", n_init=10)
        elif acquisition_function_name == "kmeans_data":
            return Kmeans(representation="raw", n_init=10)
        elif acquisition_function_name == "adversarialBIM":
            return AdversarialBIM()
        else:
            raise NotImplementedError()

    def initialize_pool(self):
        dataset_x = SingleCycleApplication.get_dataset_x(
            self.feature_set_name,
            self.cache_directory
        )
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, 
            self.cache_directory
        )
        available_indices = sorted(
            list(set(dataset_x.get_row_names()).intersection(set(dataset_y.get_row_names())))
        )
        test_indices = sorted(
            list(
                np.random.choice(
                    available_indices, 
                    size=int(self.test_ratio * len(available_indices)), 
                    replace=False)
            )
        )
        available_indices = list(set(available_indices) - set(test_indices))
        dataset_x = dataset_x.subset(available_indices)
        dataset_y = dataset_y.subset(available_indices)
        self.topk_indices = self.get_topk_indices(dataset_y,self.topk_percent)

        #Get basic stats
        try:
            df = pd.DataFrame(dataset_y.get_data()[0])
            print(df.describe())
        except:
            print("Errors getting stats")

        #Get clusters
        path_cache_cluster_files_prefix = self.cache_directory+os.sep+'_'.join(['clusters',self.dataset_name,self.feature_set_name,str(self.topk_percent)])
        if os.path.isfile(path_cache_cluster_files_prefix+f'_topk_{self.num_topk_clusters}_clusters_to_items.pkl') and os.path.isfile(path_cache_cluster_files_prefix+f'_topk_items_to_{self.num_topk_clusters}_clusters.pkl'):
            print("Loading clusters from cache")
            with open(path_cache_cluster_files_prefix+f'_topk_{self.num_topk_clusters}_clusters_to_items.pkl', "rb") as fp:
                self.topk_clusters_to_item = pkl.load(fp)
            with open(path_cache_cluster_files_prefix+f'_topk_items_to_{self.num_topk_clusters}_clusters.pkl', "rb") as fp:
                self.topk_item_to_clusters = pkl.load(fp)
        else:
            print("Did not find clusters in cache -- computing top cluster assignment")
            self.topk_clusters_to_item, self.topk_item_to_clusters = clustering.get_top_target_clusters(dataset_x, 
                                                                                                    self.topk_indices, 
                                                                                                    num_clusters = self.num_topk_clusters,
                                                                                                    plot_location=self.output_directory+os.sep+'plot_optimal_number_top_clusters.png', 
                                                                                                    path_cache_cluster_files_prefix=path_cache_cluster_files_prefix
                                                                                                )
        return dataset_x, available_indices, test_indices

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return []
    
    def get_model(self, **kwargs):
        return None

    def get_topk_indices(self, dataset_y: AbstractDataSource, topk_percent: float = 0.1):
        y_indices = dataset_y.get_row_names()
        y_values = pd.DataFrame(dataset_y.get_data()[0].flatten(), index = y_indices, columns=['y_values'])
        y_values.sort_values(by=['y_values'], ascending=False, axis=0, inplace=True)
        num_k = int(topk_percent * len(y_values.y_values))
        topk_indices = list(y_values.iloc[:num_k].index)
        return topk_indices

    def get_cumulative_precision_recall_topk(self,cumulative_indices: List[int], dataset_y: AbstractDataSource, topk_percent: float = 0.1):
        """
        topk_percent is the percent of top target recalled over the course of the different cycles
        """
        if self.topk_indices is None:
            self.topk_indices = self.get_topk_indices(dataset_y,topk_percent)
        precision, recall = len(set(self.topk_indices) & set(cumulative_indices)) / len(set(cumulative_indices)), len(set(self.topk_indices) & set(cumulative_indices)) / len(set(self.topk_indices))
        return precision, recall

    def get_cumulative_proportion_top_clusters_recovered(self,cumulative_indices):
        assert self.topk_clusters_to_item is not None, "Top clusters have not been computed yet"
        num_distinct_top_clusters = len(self.topk_clusters_to_item.keys())
        recovered_clusters_dict=defaultdict(int)
        for item in cumulative_indices:
            if item in self.topk_item_to_clusters:
                recovered_clusters_dict[self.topk_item_to_clusters[item]]+=1
        proportion_top_clusters_recovered = len(recovered_clusters_dict.keys()) / float(num_distinct_top_clusters)
        return proportion_top_clusters_recovered

    def get_precision_topk_last_selected_indices(self,last_selected_indices, dataset_y: AbstractDataSource, topk_percent: float = 0.1):
        if self.topk_indices is None:
            self.topk_indices = self.get_topk_indices(dataset_y,topk_percent)
        precision_last_selected = len(set(self.topk_indices) & set(last_selected_indices)) / len(set(last_selected_indices))
        return precision_last_selected

    def train_model(self) -> Optional[AbstractBaseModel]:
        single_cycle_application_args = {
            "model_name": self.model_name,
            "seed": self.seed,
            "remote_execution_time_limit_days": self.remote_execution_time_limit_days,
            "remote_execution_mem_limit_in_mb": self.remote_execution_mem_limit_in_mb,
            "remote_execution_virtualenv_path": self.remote_execution_virtualenv_path,
            "remote_execution_num_cpus": self.remote_execution_num_cpus,
            "schedule_on_slurm": self.schedule_children_on_slurm,
        }
        cumulative_indices = []
        dataset_x, available_indices, test_indices = self.initialize_pool()
        dataset_y = SingleCycleApplication.get_dataset_y(self.dataset_name, self.cache_directory)
        dataset_y = dataset_y.subset(available_indices)
        last_selected_indices = sorted(
            list(
                np.random.choice(
                    available_indices,
                    size=int(self.acquisition_batch_size),
                    replace=False,
                )
            )
        )
        cumulative_indices += last_selected_indices
        result_records = list()
        cumulative_recall_topk = list()
        cumulative_precision_topk = list()
        cumulative_proportion_top_clusters_recovered = list()
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(
                self.output_directory, f"cycle_{cycle_index}"
            )

            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(
                current_cycle_directory, "selected_indices.pkl"
            )
            with open(cumulative_indices_file_path, "wb") as fp:
                pkl.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(
                current_cycle_directory, "test_indices.pkl"
            )
            with open(test_indices_file_path, "wb") as fp:
                pkl.dump(test_indices, fp)
            app = SingleCycleApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=1.0, 
                hyperopt=self.hyperopt_children,
                selected_indices_file_path=cumulative_indices_file_path,
                test_indices_file_path=test_indices_file_path,
                **single_cycle_application_args,
            )
            results = app.run().run_result
            info(results.test_scores, "test scores")
            result_records.append(results.test_scores)
            available_indices = sorted(list(set(available_indices) - set(last_selected_indices)))
            
            trained_model = app.model.load(results.model_path)
            if trained_model is None:
                print("Could not find trained model at specified path in results")
                trained_model = app.get_model()
            
            last_selected_indices = self.acquisition_function(
                dataset_x=dataset_x,
                acquisition_batch_size=self.acquisition_batch_size,
                available_indices=available_indices,
                last_selected_indices=last_selected_indices,
                cumulative_indices=cumulative_indices, 
                model=trained_model,
                dataset_y=dataset_y,
                temp_folder_name=self.temp_folder_name
            )
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = list(set(cumulative_indices))
            assert len(last_selected_indices) == self.acquisition_batch_size
            last_batch_precision_topk = self.get_precision_topk_last_selected_indices(last_selected_indices=last_selected_indices, dataset_y=dataset_y, topk_percent=self.topk_percent)
            precision_topk, recall_topk = self.get_cumulative_precision_recall_topk(cumulative_indices=cumulative_indices, dataset_y=dataset_y, topk_percent=self.topk_percent)
            cumulative_precision_topk.append(precision_topk)
            cumulative_recall_topk.append(recall_topk)
            proportion_top_clusters_recovered = self.get_cumulative_proportion_top_clusters_recovered(cumulative_indices)
            cumulative_proportion_top_clusters_recovered.append(proportion_top_clusters_recovered)
            
            performance_file_exists = os.path.exists(self.performance_file_location)
            with open(self.performance_file_location, "a") as performance_record:
                if not performance_file_exists:
                    header = "dataset_name,feature_set_name,topk_percent,model_name,acquisition_function_name,acquisition_batch_size,num_active_learning_cycles,seed,num_total_items,num_topk_items,num_topk_clusters,acquisition_cycle,last_batch_precision_topk,aggregated_precision_topk,aggregated_recall_topk,aggregated_proportion_top_clusters_recovered,cumulative_precision_topk,cumulative_recall_topk,cumulative_proportion_top_clusters_recovered,bax_topk_kvalue,bax_subset_select_subset_size,bax_subset_level_c,bax_batch_selection_mode"
                    performance_record.write(header+"\n")
                record = ','.join([str(x) for x in [self.dataset_name,self.feature_set_name,self.topk_percent,self.model_name,self.acquisition_function_name,self.acquisition_batch_size,self.num_active_learning_cycles,self.seed,len(dataset_y),len(self.topk_indices),len(self.topk_clusters_to_item.keys()),cycle_index+1,last_batch_precision_topk,precision_topk,recall_topk,proportion_top_clusters_recovered,'"'+str(cumulative_precision_topk)+'"','"'+str(cumulative_recall_topk)+'"','"'+str(cumulative_proportion_top_clusters_recovered)+'"',self.bax_topk_kvalue,self.bax_subset_select_subset_size,self.bax_subset_level_c,self.bax_batch_selection_mode]])
                performance_record.write(record+"\n")

        print("Cumulative precision topk: {}".format(cumulative_precision_topk))
        print("Cumulative recall topk: {}".format(cumulative_recall_topk))
        print("cumulative proportion top clusters recovered: {}".format(cumulative_proportion_top_clusters_recovered))
        results_path = os.path.join(self.output_directory, "results.pkl")
        with open(results_path, "wb") as fp:
            pkl.dump(result_records, fp)
        return None


if __name__ == "__main__":
    GeneDisco_loop = sp.instantiate_from_command_line(GeneDiscoLoop)
    GeneDisco_loop.run()