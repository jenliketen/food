#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numpy import percentile


def remove_outliers(data):
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    
    return outliers_removed


def plot_hist(data, xlab, name):
    fig = plt.figure(figsize=(5, 5))
    plt.hist(data, alpha=0.5, color="orange", edgecolor="dimgray")
    plt.xlabel("{}".format(xlab))
    plt.ylabel("Count")
    plt.title("Distribution of {}".format(name))


def plot_hist_outliers(outliers_keep, outliers_removed, xlab, name, remove_outliers=True):
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(outliers_keep, alpha=0.5, color="orange", edgecolor="dimgray")
    plt.xlabel("{}".format(xlab))
    plt.ylabel("Number of recipes")
    plt.title("Distribution of {} (including outliers)".format(name))
    plt.subplot(1, 2, 2)
    plt.hist(outliers_removed, alpha=0.5, color="orange", edgecolor="dimgray")
    plt.xlabel("{}".format(xlab))
    plt.ylabel("Number of recipes")
    plt.title("Distribution of {} (outliers removed)".format(name))
    
    if remove_outliers == False:
        fig = plt.figure(figsize=(5, 5))
        plt.hist(outliers_keep, alpha=0.5, color="orange", edgecolor="dimgray")
        plt.xlabel("{}".format(xlab))
        plt.ylabel("Number of recipes")
        plt.title("Distribution of {} (outliers removed)".format(name))