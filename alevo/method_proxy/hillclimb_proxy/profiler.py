# HillCLimb uses the default profilers

from ...tools.profiler import TensorboardProfiler
from ...tools.profiler import WandBProfiler

import sys

try:
    import wandb
    from torch.utils.tensorboard import SummaryWriter
except:
    pass

from ...base import Function
import os
import json


class HillClimbProxyTensorboardProfiler(TensorboardProfiler):

    def __init__(self,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex'):
        """
        Args:
            log_dir  : folder path for tensorboard log files.
            log_style: the output style in the terminal. Option in ['complex', 'simple']
        """
        super().__init__(log_dir=log_dir, initial_num_samples=initial_num_samples, log_style=log_style)
        self._cur_best_program_target_score = float('-inf')

    def _write_json(self, function: Function):
        if not self._log_dir:
            return

        sample_order = self.__class__._num_samples
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(function)
        score = function.score
        target_score = function.target_score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score,
            'target_score': target_score
        }
        path = os.path.join(self._samples_json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _record_and_verbose(self, function, *, resume_mode=False):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        target_score = function.target_score

        # update best function
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_function = function
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self.__class__._num_samples

        # update target score
        if target_score is not None and target_score > self._cur_best_program_target_score:
            self._cur_best_program_target_score = target_score

        if not resume_mode:
            # log attributes of the function
            if self._log_style == 'complex':
                print(f'================= Evaluated Function =================')
                print(f'{function_str}')
                print(f'------------------------------------------------------')
                print(f'Proxy Score  : {str(score)}')
                print(f'Target Score : {str(target_score)}')
                print(f'Sample time  : {str(sample_time)}')
                print(f'Evaluate time: {str(evaluate_time)}')
                print(f'Sample orders: {str(self.__class__._num_samples)}')
                print(f'------------------------------------------------------')
                print(f'Current best proxy score : {self._cur_best_program_score}')
                print(f'Current best target score: {self._cur_best_program_target_score}')
                print(f'======================================================\n')
            else:
                if score is None:
                    print(f'Sample{self.__class__._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}')
                else:
                    print(f'Sample{self.__class__._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _write_tensorboard(self, *args, **kwargs):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Proxy Score of Function',
            self._cur_best_program_score,
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalar(
            'Best Target Score of Function',
            self._cur_best_program_target_score,
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self.__class__._num_samples
        )


class HillClimbProxyWandBProfiler(WandBProfiler):

    def __init__(self,
                 wandb_project_name: str,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 **wandb_init_kwargs):
        """
        Args:
            wandb_project_name : the project name in which you sync your results.
            log_dir            : folder path for tensorboard log files.
            wandb_init_kwargs  : args used to init wandb project, such as name='funsearch_run1', group='funsearch'.
            log_style          : the output style in the terminal. Option in ['complex', 'simple'].
        """
        super().__init__(wandb_project_name=wandb_project_name,
                         log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         **wandb_init_kwargs)
        self._cur_best_program_target_score = float('-inf')

    def _write_wandb(self, *args, **kwargs):
        self._logger.log(
            {
                'Best Proxy Score of Function': self._cur_best_program_score
            },
            step=self.__class__._num_samples
        )
        self._logger.log(
            {
                'Best Target Score of Function': self._cur_best_program_target_score
            },
            step=self.__class__._num_samples
        )
        self._logger.log(
            {
                'Valid Function Num': self._evaluate_success_program_num,
                'Invalid Function Num': self._evaluate_failed_program_num
            },
            step=self.__class__._num_samples
        )
        self._logger.log(
            {
                'Total Sample Time': self._tot_sample_time,
                'Total Evaluate Time': self._tot_evaluate_time
            },
            step=self.__class__._num_samples
        )

    def _write_json(self, function: Function):
        if not self._log_dir:
            return

        sample_order = self.__class__._num_samples
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(function)
        score = function.score
        target_score = function.target_score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score,
            'target_score': target_score
        }
        path = os.path.join(self._samples_json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _record_and_verbose(self, function, *, resume_mode=False):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        target_score = function.target_score

        # update best function
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_function = function
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self.__class__._num_samples

        # update target score
        if target_score is not None and target_score > self._cur_best_program_target_score:
            self._cur_best_program_target_score = target_score

        if not resume_mode:
            # log attributes of the function
            if self._log_style == 'complex':
                print(f'================= Evaluated Function =================')
                print(f'{function_str}')
                print(f'------------------------------------------------------')
                print(f'Proxy Score  : {str(score)}')
                print(f'Target Score : {str(target_score)}')
                print(f'Sample time  : {str(sample_time)}')
                print(f'Evaluate time: {str(evaluate_time)}')
                print(f'Sample orders: {str(self.__class__._num_samples)}')
                print(f'------------------------------------------------------')
                print(f'Current best proxy score : {self._cur_best_program_score}')
                print(f'Current best target score: {self._cur_best_program_target_score}')
                print(f'======================================================\n')
            else:
                if score is None:
                    print(f'Sample{self.__class__._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}')
                else:
                    print(f'Sample{self.__class__._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
