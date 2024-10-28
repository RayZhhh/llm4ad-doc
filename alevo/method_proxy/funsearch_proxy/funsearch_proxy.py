from __future__ import annotations

import concurrent.futures
import time
from functools import partial
from threading import Thread
import traceback

import numpy as np

from . import programs_database
from .config import ProgramsDatabaseConfig
from ...base import *
from .profiler import FunSearchProxyProfiler


class FunSearchProxy:
    def __init__(self,
                 template_program: str,
                 sampler: Sampler,
                 proxy_evaluator: Evaluator,
                 evaluator: Evaluator,
                 profiler: FunSearchProxyProfiler = None,
                 n_program_evals: int = 1,
                 config: ProgramsDatabaseConfig = ProgramsDatabaseConfig(),
                 max_sample_nums: int | None = 20,
                 num_samplers: int = 4,
                 num_evaluators: int = 4,
                 samples_per_prompt: int = 4,
                 *,
                 valid_only: bool = False,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: str = 'thread',
                 **kwargs):
        """
        Args:
            template_program: the seed program (in str) as the initial function of the run.
                the template_program should be executable, i.e., incorporating package import, and function definition, and function body.
            sampler         : an instance of 'alevo.base.Sampler', which provides the way to query LLM.
            evaluator       : an instance of 'alevo.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'alevo.method.funsearch.FunSearchProfiler'. If you do not want to use it, you can pass a 'None'.
            config          : an instance of 'alevo.method.funsearch.config.ProgramDatabaseConfig'.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not).
            resume_mode     : in resume_mode, funsearch will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs        : some args pass to 'alevo.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        # arguments and keywords
        self._template_program_str = template_program
        self._max_sample_nums = max_sample_nums
        self._debug_mode = debug_mode
        self._resume_mode = resume_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(template_program)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(template_program)

        # population, sampler, and evaluator
        self._n_program_evals = n_program_evals
        self._database = programs_database.ProgramsDatabase(
            config,
            self._template_program,
            self._function_to_evolve_name
        )
        self._sampler = SamplerTrimmer(sampler)
        self._proxy_evaluator = SecureEvaluator(proxy_evaluator, debug_mode=debug_mode, **kwargs)
        self._evaluator = SecureEvaluator(evaluator, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        self._num_samplers = num_samplers
        self._valid_only = valid_only
        self._num_evaluators = num_evaluators
        self._samples_per_prompt = samples_per_prompt

        # statistics
        self._tot_sample_nums = 0

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_evaluators
            )

        # threads for sampling
        self._sampler_threads = [
            Thread(target=self._sample_evaluate_register) for _ in range(self._num_samplers)
        ]

    def _sample_evaluate_register(self):
        while (self._max_sample_nums is None) or (self._tot_sample_nums < self._max_sample_nums):
            try:
                # get prompt
                prompt = self._database.get_prompt()
                prompt_contents = [prompt.code for _ in range(self._samples_per_prompt)]

                # do sample
                draw_sample_start = time.time()
                sampled_funcs = self._sampler.draw_samples(prompt_contents)
                draw_sample_times = time.time() - draw_sample_start
                avg_time_for_each_sample = draw_sample_times / len(sampled_funcs)

                # convert samples to program instances
                programs_to_be_eval = []
                for func in sampled_funcs:
                    program = SamplerTrimmer.sample_to_program(func, self._template_program)
                    # if sample to program success
                    if program is not None:
                        programs_to_be_eval.append(program)

                # submit tasks to the thread pool and evaluate
                futures = []
                for program in programs_to_be_eval:
                    # future = self._evaluation_executor.submit(self._evaluator.evaluate_program_record_time, program)
                    future = self._evaluation_executor.submit(
                        partial(self._proxy_evaluator.evaluate_program_record_time,
                                program,
                                funsearch=self)  # pass self to evaluator
                    )
                    futures.append(future)
                # get evaluate scores and evaluate times
                scores_times = [f.result() for f in futures]
                scores, times = [i[0] for i in scores_times], [i[1] for i in scores_times]

                # if score is better than current best (on proxy data),
                # perform n_program_evals times re-evaluations on PROXY data.
                for i, (score, program_to_be_eval) in enumerate(zip(scores, programs_to_be_eval)):
                    if score is not None and score > self._database.best_target_func.score:
                        futures = []
                        for _ in range(self._n_program_evals):
                            future = self._evaluation_executor.submit(
                                partial(self._proxy_evaluator.evaluate_program_record_time,
                                        program_to_be_eval,
                                        hillclimb=self)  # pass self to evaluator
                            )
                            futures.append(future)
                        scores_times = [f.result() for f in futures]
                        pxy_scores, times = [i[0] for i in scores_times], [i[1] for i in scores_times]
                        scores[i] = None if None in pxy_scores else np.mean(pxy_scores)

                # if n_eval_avg_score_pxy better than current best (on proxy data),
                # perform additional five re-evaluations on TARGET data.
                target_scores = [None for _ in scores]
                for i, (score, program_to_be_eval) in enumerate(zip(scores, programs_to_be_eval)):
                    if score is not None and score > self._database.best_target_func.score:
                        futures = []
                        for _ in range(self._n_program_evals):
                            future = self._evaluation_executor.submit(
                                partial(self._evaluator.evaluate_program_record_time,
                                        program_to_be_eval,
                                        hillclimb=self)  # pass self to evaluator
                            )
                            futures.append(future)
                        scores_times = [f.result() for f in futures]
                        tar_scores, times = [i[0] for i in scores_times], [i[1] for i in scores_times]
                        target_scores[i] = None if None in tar_scores else np.mean(tar_scores)

                # register to program database and profiler
                island_id = prompt.island_id
                for program, score, target_score, eval_time in zip(programs_to_be_eval, scores, target_scores, times):
                    # convert to Function instance
                    function = TextFunctionProgramConverter.program_to_function(program)
                    # check if the function has converted to Function instance successfully
                    if function is None:
                        continue
                    # register to program database
                    function.score = score
                    function.target_score = target_score
                    function.sample_time = avg_time_for_each_sample
                    function.evaluate_time = eval_time
                    if score is not None:
                        self._database.register_function(
                            function=function,
                            island_id=island_id,
                            score=score
                        )
                    # register to profiler
                    if self._profiler is not None:
                        if self._valid_only:
                            if score is not None:
                                self._profiler.register_function(function)
                                self._profiler.register_program_db(self._database)
                        else:
                            self._profiler.register_function(function)
                            self._profiler.register_program_db(self._database)
                    # update
                    if self._valid_only:
                        if score is not None:
                            self._tot_sample_nums += 1
                    else:
                        self._tot_sample_nums += 1

            except Exception as e:
                if self._debug_mode:
                    print(traceback.format_exc())
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def run(self):
        if not self._resume_mode:
            # evaluate the template program on proxy evaluator, make sure the score of which is not 'None'
            score, eval_time = [], []
            for _ in range(self._n_program_evals):
                s, t = self._proxy_evaluator.evaluate_program_record_time(program=self._template_program, hillclimb=self)
                if s is None:
                    raise RuntimeError('The score of the template function on "proxy evaluator" must not be "None".')
                score.append(s)
                eval_time.append(t)

            score, eval_time = np.mean(score), np.mean(eval_time)

            # evaluate the template program on target evaluator
            target_score = []
            for _ in range(self._n_program_evals):
                s, _ = self._evaluator.evaluate_program_record_time(program=self._template_program, hillclimb=self)
                if s is None:
                    raise RuntimeError('The score of the template function on "evaluator" must not be "None".')
                target_score.append(s)
            target_score = np.mean(target_score)

            # register the template program to the program database
            self._function_to_evolve.score = score
            self._function_to_evolve.target_score = target_score
            self._function_to_evolve.evaluate_time = eval_time
            self._database.register_function(function=self._function_to_evolve, island_id=None, score=score)

            if self._profiler:
                self._profiler.register_function(self._function_to_evolve)

        # start sampling using multiple threads
        for t in self._sampler_threads:
            t.start()

        # join all threads to the main thread
        for t in self._sampler_threads:
            t.join()

        if self._profiler is not None:
            self._profiler.finish()
