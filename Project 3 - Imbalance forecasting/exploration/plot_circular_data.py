from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.helpers.path import ROOT_PATH

def _plot_time_in_day(ax: plt.Axes):
    minutes_per_day = 60 * 24
    hours_extra = 3
    minutes = np.arange(-60 * hours_extra, minutes_per_day + 60 * hours_extra, 5)
    
    sin_values = np.sin(minutes * 2 * np.pi / minutes_per_day)
    cos_values = np.cos(minutes * 2 * np.pi / minutes_per_day)

    within_day = np.where((minutes >= 0) & (minutes <= minutes_per_day))
    after_day = minutes >= minutes_per_day
    before_day = minutes <= 0

    color_sin = plt.get_cmap("Paired").colors[0]
    color_cos = plt.get_cmap("Paired").colors[1]
    ax.plot(minutes[within_day], sin_values[within_day], label="sin(time of day)", color=color_sin)
    ax.plot(minutes[within_day], cos_values[within_day], label="cos(time of day)", color=color_cos)

    # Plot previous and future times
    ax.plot(minutes[before_day], sin_values[before_day], "--", color=color_sin)
    ax.plot(minutes[before_day], cos_values[before_day], "--", color=color_cos)
    ax.plot(minutes[after_day], sin_values[after_day], "--", color=color_sin)
    ax.plot(minutes[after_day], cos_values[after_day], "--", color=color_cos)

    # Add vertical line to signify day
    ax.vlines([0, minutes_per_day], colors=plt.get_cmap("Paired").colors[9], ymin=-1, ymax=1.2)

    # Add legend and title
    ax.set_title("Conversion of time of day")
    ax.set_xlabel("Minutes into the day")
    ax.legend()

def _plot_time_of_week(ax: plt.Axes):
    day_values = np.arange(-1, 8, 1)
    
    sin_values = np.sin(day_values * 2 * np.pi / 7)
    cos_values = np.cos(day_values * 2 * np.pi / 7)

    week = np.where((day_values >= 0) & (day_values <= 6))
    next_week = day_values >= 6
    last_week = day_values <= 0

    color_sin = plt.get_cmap("Paired").colors[2]
    color_cos = plt.get_cmap("Paired").colors[3]
    ax.plot(day_values[week], sin_values[week], label="sin(day of week)", color=color_sin)
    ax.plot(day_values[week], cos_values[week], label="cos(day of week)", color=color_cos)

    # Plot previous and future times
    ax.plot(day_values[last_week], sin_values[last_week], "--", color=color_sin)
    ax.plot(day_values[last_week], cos_values[last_week], "--", color=color_cos)
    ax.plot(day_values[next_week], sin_values[next_week], "--", color=color_sin)
    ax.plot(day_values[next_week], cos_values[next_week], "--", color=color_cos)

    # Add vertical line to show week
    ax.vlines([0, 6], colors=plt.get_cmap("Paired").colors[9], ymin=-1, ymax=1.2)

    # Add legend and title
    ax.set_title("Conversion of day of week")
    ax.set_xlabel("Days into the week")

    # Change x-ticks to be weekdays
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]
    ax.set_xticks(range(-1, len(days)-1), days)
    ax.legend()


def _plot_time_of_year(ax: plt.Axes):
    days_per_year = 365
    days_extra = 30
    days = np.arange(-days_extra, days_per_year + days_extra, 1)
    
    sin_values = np.sin(days * 2 * np.pi / days_per_year)
    cos_values = np.cos(days * 2 * np.pi / days_per_year)

    this_year = np.where((days >= 0) & (days <= days_per_year))
    next_year = days >= days_per_year
    last_year = days <= 0

    color_sin = plt.get_cmap("Paired").colors[4]
    color_cos = plt.get_cmap("Paired").colors[5]
    ax.plot(days[this_year], sin_values[this_year], label="sin(time of year)", color=color_sin)
    ax.plot(days[this_year], cos_values[this_year], label="cos(time of year)", color=color_cos)

    # Plot previous and future times
    ax.plot(days[last_year], sin_values[last_year], "--", color=color_sin)
    ax.plot(days[last_year], cos_values[last_year], "--", color=color_cos)
    ax.plot(days[next_year], sin_values[next_year], "--", color=color_sin)
    ax.plot(days[next_year], cos_values[next_year], "--", color=color_cos)

    # Add vertical line to signify day
    ax.vlines([0, days_per_year], colors=plt.get_cmap("Paired").colors[9], ymin=-1, ymax=1.2)

    # Add legend and title
    ax.set_title("Conversion of time of year")
    ax.set_xlabel("Days into the year")
    ax.legend()


if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 30))
    savepath = ROOT_PATH / "plots/circular_data.png"

    _plot_time_in_day(ax[0])
    _plot_time_of_week(ax[1])
    _plot_time_of_year(ax[2])
    fig.savefig(savepath)
