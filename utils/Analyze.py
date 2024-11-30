import statistics
from collections.abc import Sequence
import numpy as np
from project.utils.welch import Welch
from matplotlib import pyplot as plt

from scipy import stats

from project.env_system import ManufacturingSystem


def t_student_critical_value(alpha: float, n: int) -> float:
    """return the critical value of a t-test for a given alpha and n"""
    return stats.t.ppf(1 - alpha, n - 1)


def analyze_mts(runs: Sequence[ManufacturingSystem], warmup_period: int, alpha: float = 0.05) -> tuple[
    float, float, float]:
    """Analyse  the Mean time in system of a product give a sequence of runs, a warmpup period and an alpha
    :returns
    mts_sample_mean: mean time in system of runs
    mts_sample_variance: variance of the mean time in system of runs
    half_interval: the half-interval of relevance of the measures
    """
    n = len(runs)
    sample = [
        statistics.mean([
            job.time_in_system
            for job in run.jobs
            if job.done[0] and job.arrival_time >= warmup_period])
        for run in runs
    ]
    throughput_sample_mean = statistics.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)
    return throughput_sample_mean, throughput_sample_variance, half_interval


def analyze_wip(runs: Sequence[ManufacturingSystem], warmup_period: int, alpha: float = 0.05) -> tuple[
    float, float, float]:
    """Analyse  the Mean time in system of a product give a sequence of runs, a warmpup period and an alpha
    :returns
    wip_sample_mean: mean time in system of runs
    wip_variance: variance of the mean time in system of runs
    half_interval: the half-interval of relevance of the measures
    """
    n = len(runs)
    sample = [
        statistics.mean(run.wip_tot_stats[warmup_period:])
        for run in runs
    ]
    throughput_sample_mean = statistics.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)
    return throughput_sample_mean, throughput_sample_variance, half_interval


def analyze_parameter(runs: list[dict], param: str, alpha: float = 0.05) -> tuple[float, float, float]:
    """Analyse  sample_mean,  sample_variance and half_interval of a generic Sequence of dict
    """
    n = len(runs)
    sample = [
        run[param] for run in runs
    ]
    throughput_sample_mean = statistics.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)

    return throughput_sample_mean, throughput_sample_variance, half_interval


def welch_Avg(sample, window_size) -> np.ndarray:
        averaged_process = []
        for i in range(1, sample.shape[0] - window_size):
            if i <= window_size:
                averaged_process.append(sample[:2 * i - 1].mean())
            else:
                averaged_process.append(
                    sample[i - window_size // 2:i + window_size // 2].mean())
        return np.array(averaged_process)

def plot_comparison(models: dict[str, list[dict]], param: str, window_size:int=50):
    """ Plot the average measures of a given parameter"""

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(111)
    # store the values of each model
    for model_name, runs in models.items():
        sample = np.array([
            run[param] for run in runs
        ])

        # do welch graph
        avg_sample = welch_Avg(sample,window_size=window_size)

        # plot it
        ax1.plot(avg_sample, label=model_name)

    # Set axes
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(param)
    ax1.set_title(f'{param} comparisons')
    ax1.legend()
    plt.show()


def compare_with_std(models: dict[str, list[dict]], standard: list[dict], param: str, alpha: float = 0.05):
    """"Compare the standard (PUSH) with all other models"""

    # we calculate the sample of the standard
    std_sample = np.array([
        run[param] for run in standard
    ])

    # for problem in my notebook view
    print(' ')
    # for every model
    for model_name, runs in models.items():
        print('-' * 35 + f' Model: {model_name} ' + '+' * 35)
        sample_model = np.array([
            run[param] for run in runs
        ])
        Z = std_sample - sample_model
        n = Z.shape[0]
        Z_mean = Z.mean()
        Z_variance = np.power(Z - Z.mean(), 2).sum() / (n * (n - 1))
        t = t_student_critical_value(alpha=alpha, n=n)
        half_interval = t * np.sqrt(Z_variance)
        print(f'Half interval of Model:{model_name} with respect to the standard:')
        print(f'{Z_mean - half_interval}, {Z_mean + half_interval}')
        print(f'-' * 35 + f' Model: {model_name} ' + '+' * 35)
