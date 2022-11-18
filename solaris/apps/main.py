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
import os
import sys
import pickle
import inspect
from xml.sax.handler import feature_external_ges
import torch
import pickle as pkl
import numpy as np
import slingpy as sp
from slingpy.utils.logging import info, warn
from slingpy.utils.path_tools import PathTools
from typing import AnyStr, Dict, List, Optional
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from solaris.data import gt_data_synthesizer as ds
from solaris.methods.entropic_ucb import ucb_sampling
from solaris.models import toy_experiment_models
import genedisco.models.pytorch_models as pmd
from genedisco.models import meta_models
from slingpy.models import torch_model
from collections import namedtuple
from slingpy.evaluation.evaluator import Evaluator
import pdb

# delete
import matplotlib.pyplot as plt


DIMENSIONS = 50
DataSet = namedtuple(
    "DataSet",
    "training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y",
)


class RandomBatchAcquisitionFunction(object):
    def __call__(
        self,
        dataset_x,
        batch_size,
        available_indices,
        last_selected_indices,
        last_model,
    ):
        info(len(available_indices), batch_size, " randomacquisition")
        selected = np.random.choice(available_indices, size=batch_size, replace=False)
        return selected


class CustomLoss(sp.TorchLoss):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def __call__(
        self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = self.loss_fn(y_pred[0], y_true[0].float())
        return loss


class SingleCycleApplication(sp.AbstractBaseApplication):
    def __init__(
        self,
        model_name: AnyStr = "randomforest",
        acquisition_function_name: AnyStr = "random",
        acquisition_batch_size: int = 4,
        num_solaris_cycles: int = 10,
        output_directory: AnyStr = "",
        test_ratio: float = 0.2,
        hyperopt_children: bool = True,
        seed: int = 0,
        dataset_name: AnyStr = "synthetic",
        feature_set_name="",
        selected_indices_file_path="",
        test_indices_file_path="",
        **kwargs,
    ):
        self.model_name = model_name
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_batch_size = acquisition_batch_size
        self.num_solaris_cycles = num_solaris_cycles
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.selected_indices_file_path = selected_indices_file_path
        self.test_indices_file_path = test_indices_file_path
        with open(test_indices_file_path, "rb") as fp:
            self.test_indices = pickle.load(fp)
        with open(selected_indices_file_path, "rb") as fp:
            self.selected_indices = pkl.load(fp)

        super(SingleCycleApplication, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=True,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
        )
        self.initialize_pool()
        self.model = self.get_model()

    def initialize_pool(self):
        """Get the data as (x, value) pairs

        Raises:
            NotImplementedError: _description_
        """

        if self.dataset_name == "synthetic":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer(
                n_features=DIMENSIONS, n_informative=int(DIMENSIONS / 2)
            )
            self.data_synthesizer = gen_d

            gen_d._generate_data()

            dataset_x = gen_d.dataset[0]
            dataset_y = gen_d.dataset[1]

            available_indices = list(range(int(0.9 * len(dataset_y))))
            test_indices = list(range(int(0.9 * len(dataset_y)), len(dataset_y)))
            self.dataset = DataSet(**self.load_data())
            # self.datasets = DataSet(**self.load_data())

            return dataset_x, available_indices, test_indices
        else:
            raise NotImplementedError()

    def init_data(self):
        self.datasets = DataSet(**self.load_data())

    # This is what the active learning loop app does in genedisco as well
    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {
            "training_set_x": self.data_synthesizer.dataset[0][self.selected_indices],
            "training_set_y": self.data_synthesizer.dataset[1][self.selected_indices],
            "validation_set_x": self.data_synthesizer.dataset[0][self.test_indices],
            "validation_set_y": self.data_synthesizer.dataset[1][self.test_indices],
            "test_set_x": self.data_synthesizer.dataset[0][self.test_indices],
            "test_set_y": self.data_synthesizer.dataset[1][self.test_indices],
        }

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
        ]

    def train_model(self) -> Optional[AbstractBaseModel]:

        self.model.fit(
            self.data_synthesizer.dataset[0][self.selected_indices],
            self.data_synthesizer.dataset[1][self.selected_indices],
        )

        return self.model
        raise NotImplementedError

    def get_model(self) -> Optional[AbstractBaseModel]:
        # Hardcoded
        if self.model_name == "gp":
            sp_model = toy_experiment_models.GaussianProcessModel(
                None, noise_sigma=0.1, save_path_dir="./gpmodel", kernel_lengthscale=0.1
            )
        elif self.model_name == "linear":
            # Need to adaptively update num dimensions
            sp_model = toy_experiment_models.BayesianLinearModel(
                1, prior_sigma=1.0, noise_sigma=0.1, save_path_dir="./linearmodel"
            )
        return sp_model

    def get_acquisition_function(
        acquisition_function_name: AnyStr, acquisition_function_path: AnyStr
    ):
        if acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        else:
            return RandomBatchAcquisitionFunction()

    def evaluate_model(
        self,
        model: AbstractBaseModel,
        dataset_x: AbstractDataSource,
        dataset_y: AbstractDataSource,
        with_print: bool = True,
        set_name: AnyStr = "",
        threshold=None,
    ):
        """
        Evaluates model performance.
        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.
        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """

        preds, cov = model.predict(dataset_x)
        uncs = np.diag(cov)
        info(
            "predictions", preds[:10], dataset_y[:10], ((preds - dataset_y) ** 2).mean()
        )
        return {"MSE": ((preds - dataset_y) ** 2).mean()}
        # return Evaluator.evaluate(
        #     model,
        #     dataset_x,
        #     dataset_y,
        #     self.get_metrics(set_name),
        #     with_print=with_print,
        #     set_name=set_name,
        #     threshold=threshold,
        # )


class SolarisLoop(sp.AbstractBaseApplication):
    ACQUISITION_FUNCTIONS = ["random", "ucb"]

    def __init__(
        self,
        model_name: AnyStr = "gp",
        acquisition_function_name: AnyStr = "random",
        acquisition_function_path: AnyStr = "custom",
        acquisition_batch_size: int = 1,
        num_active_learning_cycles: int = 6,
        feature_set_name: AnyStr = None,
        dataset_name: AnyStr = "synthetic",
        cache_directory: AnyStr = "cache",
        output_directory: AnyStr = "output",
        test_ratio: float = 0.2,
        hyperopt_children: bool = True,
        schedule_on_slurm: bool = False,
        schedule_children_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 1,
        remote_execution_mem_limit_in_mb: int = 2048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        seed: int = 0,
    ):
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_path = acquisition_function_path
        PathTools.mkdir_if_not_exists(output_directory)
        self.acquisition_function = SolarisLoop.get_acquisition_function(
            self.acquisition_function_name, self.acquisition_function_path
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
        super(SolarisLoop, self).__init__(
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

    @staticmethod
    def get_acquisition_function(
        acquisition_function_name: AnyStr, acquisition_function_path: AnyStr
    ):

        if acquisition_function_name == "random":
            return RandomBatchAcquisitionFunction()
        elif acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        else:
            raise NotImplementedError()

    def initialize_pool(self):
        """Get the data as (x, value) pairs

        Raises:
            NotImplementedError: _description_
        """

        if self.dataset_name == "synthetic":
            # Generate synthetic dataset
            gen_d = ds.GTDataSynthesizer()
            self.data_synthesizer = gen_d
            gen_d._generate_data()
            dataset_x = gen_d.dataset[0]
            dataset_y = gen_d.dataset[1]
            available_indices = list(range(int(0.9 * len(dataset_y))))
            test_indices = list(range(int(0.9 * len(dataset_y)), len(dataset_y)))
            return dataset_x, available_indices, test_indices
        else:
            return NotImplementedError

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_model(self):
        return None

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return []

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
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(
                self.output_directory, f"cycle_{cycle_index}"
            )
            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(
                current_cycle_directory, "selected_indices.pickle"
            )
            with open(cumulative_indices_file_path, "wb") as fp:
                pickle.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(
                current_cycle_directory, "test_indices.pickle"
            )
            with open(test_indices_file_path, "wb") as fp:
                pickle.dump(test_indices, fp)

            app = SingleCycleApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=0.8,
                hyperopt=self.hyperopt_children,
                selected_indices_file_path=cumulative_indices_file_path,
                test_indices_file_path=test_indices_file_path,
                **single_cycle_application_args,
            )
            results = app.run().run_result
            info(results.test_scores, "test scores")
            result_records.append(results.test_scores)
            available_indices = list(
                set(available_indices) - set(last_selected_indices)
            )

            trained_model_path = results.model_path
            # Note: currently results.model_path does not equal the path
            # that gets saved to by GP or BLR models, so this will not load anything.
            # TODO: fix model checkpointing.
            trained_model = app.model.load(trained_model_path)
            if trained_model is None:
                trained_model = app.get_model()
            last_selected_indices = self.acquisition_function(
                dataset_x,
                self.acquisition_batch_size,
                available_indices,
                last_selected_indices,
                trained_model,
            )
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = list(set(cumulative_indices))
            assert len(last_selected_indices) == self.acquisition_batch_size

        results_path = os.path.join(self.output_directory, "results.pickle")
        with open(results_path, "wb") as fp:
            pickle.dump(result_records, fp)
        return None


def main():
    solaris_loop = sp.instantiate_from_command_line(SolarisLoop)
    data = solaris_loop.initialize_pool()

    solaris_loop.run()


if __name__ == "__main__":
    main()
