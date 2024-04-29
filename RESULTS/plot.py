import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kernel1 = pd.read_csv('kernel1.csv')
kernel2 = pd.read_csv('kernel2.csv')
kernel3 = pd.read_csv('kernel3.csv')
kernel4 = pd.read_csv('kernel4.csv')
kernel5 = pd.read_csv('kernel5.csv')
kernel6_d = pd.read_csv('kernel6_double.csv')
kernel6_s = pd.read_csv('kernel6_single.csv')
cub = pd.read_csv('cub.csv')
thrust = pd.read_csv('thrust.csv')
cpu_nothread = pd.read_csv('cpu_nothread.csv')
cpu_thread = pd.read_csv('cpu_thread.csv')

input_s = kernel1["input_single"]

# Single
time_single_kernel1 = kernel1["time_single_ms"]
time_single_kernel2 = kernel2["time_single_ms"]
time_single_kernel3 = kernel3["time_single_ms"]
time_single_kernel4 = kernel4["time_single_ms"]
time_single_kernel5 = kernel5["time_single_ms"]
time_single_cub = cub["time_single_ms"]
time_single_thrust = thrust["time_single_ms"]
time_single_cpu_nothread = cpu_nothread["time_single_ms"]
time_single_cpu_thread = cpu_thread["time_single_ms"]

time_single_kernel6_cf1 = kernel6_s["time_cf1"]
time_single_kernel6_cf2 = kernel6_s["time_cf2"]
time_single_kernel6_cf3 = kernel6_s["time_cf3"]
time_single_kernel6_cf4 = kernel6_s["time_cf4"]
time_single_kernel6_cf5 = kernel6_s["time_cf5"]
time_single_kernel6_cf6 = kernel6_s["time_cf6"]
time_single_kernel6_cf7 = kernel6_s["time_cf7"]
time_single_kernel6_cf8 = kernel6_s["time_cf8"]
time_single_kernel6_cf9 = kernel6_s["time_cf9"]
time_single_kernel6_cf10 = kernel6_s["time_cf10"]
time_single_kernel6_cf11 = kernel6_s["time_cf11"]
time_single_kernel6_cf12 = kernel6_s["time_cf12"]
time_single_kernel6_cf13 = kernel6_s["time_cf13"]
time_single_kernel6_cf14 = kernel6_s["time_cf14"]
time_single_kernel6_cf15 = kernel6_s["time_cf15"]
time_single_kernel6_cf16 = kernel6_s["time_cf16"]

# Double
time_double_kernel1 = kernel1["time_double_ms"]
time_double_kernel2 = kernel2["time_double_ms"]
time_double_kernel3 = kernel3["time_double_ms"]
time_double_kernel4 = kernel4["time_double_ms"]
time_double_kernel5 = kernel5["time_double_ms"]
time_double_cub = cub["time_double_ms"]
time_double_thrust = thrust["time_double_ms"]
time_double_cpu_nothread = cpu_nothread["time_double_ms"]
time_double_cpu_thread = cpu_thread["time_double_ms"]

time_double_kernel6_cf1 = kernel6_d["time_cf1"]
time_double_kernel6_cf2 = kernel6_d["time_cf2"]
time_double_kernel6_cf3 = kernel6_d["time_cf3"]
time_double_kernel6_cf4 = kernel6_d["time_cf4"]
time_double_kernel6_cf5 = kernel6_d["time_cf5"]
time_double_kernel6_cf6 = kernel6_d["time_cf6"]
time_double_kernel6_cf7 = kernel6_d["time_cf7"]
time_double_kernel6_cf8 = kernel6_d["time_cf8"]
time_double_kernel6_cf9 = kernel6_d["time_cf9"]
time_double_kernel6_cf10 = kernel6_d["time_cf10"]
time_double_kernel6_cf11 = kernel6_d["time_cf11"]
time_double_kernel6_cf12 = kernel6_d["time_cf12"]
time_double_kernel6_cf13 = kernel6_d["time_cf13"]
time_double_kernel6_cf14 = kernel6_d["time_cf14"]
time_double_kernel6_cf15 = kernel6_d["time_cf15"]
time_double_kernel6_cf16 = kernel6_d["time_cf16"]

barWidth = 0.05

# Set position of bar on X axis
br1 = np.arange(len(input_s))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
br7 = [x + barWidth for x in br6]
br8 = [x + barWidth for x in br7]
br9 = [x + barWidth for x in br8]
br10 = [x + barWidth for x in br9]
br11 = [x + barWidth for x in br10]
br12 = [x + barWidth for x in br11]
br13 = [x + barWidth for x in br12]
br14 = [x + barWidth for x in br13]
br15 = [x + barWidth for x in br14]
br16 = [x + barWidth for x in br15]

plt.figure(1, figsize=(12, 8))
plt.bar(br1, time_single_kernel1, color='blue', width=barWidth,
        edgecolor='grey', label='Kernel 1')
plt.bar(br2, time_single_kernel2, color='green', width=barWidth,
        edgecolor='grey', label='Kernel 2')
plt.bar(br3, time_single_kernel3, color='red', width=barWidth,
        edgecolor='grey', label='Kernel 3')
plt.bar(br4, time_single_kernel4, color='orange', width=barWidth,
        edgecolor='grey', label='Kernel 4')
plt.bar(br5, time_single_kernel5, color='pink', width=barWidth,
        edgecolor='grey', label='Kernel 5')
plt.bar(br6, time_single_kernel6_cf1, color='lightblue', width=barWidth,
        edgecolor='grey', label='Kernel 6 Coarse Factor 1')
plt.bar(br7, time_single_cub, color='purple', width=barWidth,
        edgecolor='grey', label='CUB Reduce')
plt.bar(br8, time_single_thrust, color='yellow', width=barWidth,
        edgecolor='grey', label='Thrust Reduce')
plt.bar(br9, time_single_cpu_nothread, color='darkorange', width=barWidth,
        edgecolor='grey', label='CPU Not Threaded')
plt.bar(br10, time_single_cpu_thread, color='black', width=barWidth,
        edgecolor='grey', label='CPE Threaded')
plt.xlabel('Input Size', fontweight='bold', fontsize=15)
plt.ylabel('Time in Milliseconds', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(input_s))],
           input_s)
plt.title("Performance Comparison Chart for Single Precision")
plt.legend()


plt.figure(2, figsize=(12, 8))
plt.bar(br1, time_double_kernel1, color='blue', width=barWidth,
        edgecolor='grey', label='Kernel 1')
plt.bar(br2, time_double_kernel2, color='green', width=barWidth,
        edgecolor='grey', label='Kernel 2')
plt.bar(br3, time_double_kernel3, color='red', width=barWidth,
        edgecolor='grey', label='Kernel 3')
plt.bar(br4, time_double_kernel4, color='orange', width=barWidth,
        edgecolor='grey', label='Kernel 4')
plt.bar(br5, time_double_kernel5, color='pink', width=barWidth,
        edgecolor='grey', label='Kernel 5')
plt.bar(br6, time_double_kernel6_cf1, color='lightblue', width=barWidth,
        edgecolor='grey', label='Kernel 6 Coarse Factor 1')
plt.bar(br7, time_double_cub, color='purple', width=barWidth,
        edgecolor='grey', label='CUB Reduce')
plt.bar(br8, time_double_thrust, color='yellow', width=barWidth,
        edgecolor='grey', label='Thrust Reduce')
plt.bar(br9, time_double_cpu_nothread, color='darkorange', width=barWidth,
        edgecolor='grey', label='CPU Not Threaded')
plt.bar(br10, time_double_cpu_thread, color='black', width=barWidth,
        edgecolor='grey', label='CPU Threaded')
plt.xlabel('Input Size', fontweight='bold', fontsize=15)
plt.ylabel('Time in Milliseconds', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(input_s))],
           input_s)
plt.title("Performance Comparison Chart for Double Precision")
plt.legend()

plt.figure(3, figsize=(12, 8))
plt.bar(br1, time_single_kernel6_cf1, color='blue', width=barWidth,
        edgecolor='grey', label='Course Factor 1')
plt.bar(br2, time_single_kernel6_cf2, color='red', width=barWidth,
        edgecolor='grey', label='Course Factor 2')
plt.bar(br3, time_single_kernel6_cf3, color='green', width=barWidth,
        edgecolor='grey', label='Course Factor 3')
plt.bar(br4, time_single_kernel6_cf4, color='purple', width=barWidth,
        edgecolor='grey', label='Course Factor 4')
plt.bar(br5, time_single_kernel6_cf5, color='black', width=barWidth,
        edgecolor='grey', label='Course Factor 5')
plt.bar(br6, time_single_kernel6_cf6, color='darkred', width=barWidth,
        edgecolor='grey', label='Course Factor 6')
plt.bar(br7, time_single_kernel6_cf7, color='orange', width=barWidth,
        edgecolor='grey', label='Course Factor 7')
plt.bar(br8, time_single_kernel6_cf8, color='darkgreen', width=barWidth,
        edgecolor='grey', label='Course Factor 8')
plt.bar(br9, time_single_kernel6_cf9, color='lightblue', width=barWidth,
        edgecolor='grey', label='Course Factor 9')
plt.bar(br10, time_single_kernel6_cf10, color='violet', width=barWidth,
        edgecolor='grey', label='Course Factor 10')
plt.bar(br11, time_single_kernel6_cf11, color='darkorange', width=barWidth,
        edgecolor='grey', label='Course Factor 11')
plt.bar(br12, time_single_kernel6_cf12, color='pink', width=barWidth,
        edgecolor='grey', label='Course Factor 12')
plt.bar(br13, time_single_kernel6_cf13, color='brown', width=barWidth,
        edgecolor='grey', label='Course Factor 13')
plt.bar(br14, time_single_kernel6_cf14, color='lightgreen', width=barWidth,
        edgecolor='grey', label='Course Factor 14')
plt.bar(br15, time_single_kernel6_cf15, color='yellow', width=barWidth,
        edgecolor='grey', label='Course Factor 15')
plt.bar(br16, time_single_kernel6_cf16, color='magenta', width=barWidth,
        edgecolor='grey', label='Course Factor 16')
plt.xlabel('Input Size', fontweight='bold', fontsize=15)
plt.ylabel('Time in Milliseconds', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(input_s))],
           input_s)
plt.title("Performance Comparison Chart for Single Precision")
plt.legend()

plt.figure(4, figsize=(12, 8))
plt.bar(br1, time_double_kernel6_cf1, color='blue', width=barWidth,
        edgecolor='grey', label='Course Factor 1')
plt.bar(br2, time_double_kernel6_cf2, color='red', width=barWidth,
        edgecolor='grey', label='Course Factor 2')
plt.bar(br3, time_double_kernel6_cf3, color='green', width=barWidth,
        edgecolor='grey', label='Course Factor 3')
plt.bar(br4, time_double_kernel6_cf4, color='purple', width=barWidth,
        edgecolor='grey', label='Course Factor 4')
plt.bar(br5, time_double_kernel6_cf5, color='black', width=barWidth,
        edgecolor='grey', label='Course Factor 5')
plt.bar(br6, time_double_kernel6_cf6, color='darkred', width=barWidth,
        edgecolor='grey', label='Course Factor 6')
plt.bar(br7, time_double_kernel6_cf7, color='orange', width=barWidth,
        edgecolor='grey', label='Course Factor 7')
plt.bar(br8, time_double_kernel6_cf8, color='darkgreen', width=barWidth,
        edgecolor='grey', label='Course Factor 8')
plt.bar(br9, time_double_kernel6_cf9, color='lightblue', width=barWidth,
        edgecolor='grey', label='Course Factor 9')
plt.bar(br10, time_double_kernel6_cf10, color='violet', width=barWidth,
        edgecolor='grey', label='Course Factor 10')
plt.bar(br11, time_double_kernel6_cf11, color='darkorange', width=barWidth,
        edgecolor='grey', label='Course Factor 11')
plt.bar(br12, time_double_kernel6_cf12, color='pink', width=barWidth,
        edgecolor='grey', label='Course Factor 12')
plt.bar(br13, time_double_kernel6_cf13, color='brown', width=barWidth,
        edgecolor='grey', label='Course Factor 13')
plt.bar(br14, time_double_kernel6_cf14, color='lightgreen', width=barWidth,
        edgecolor='grey', label='Course Factor 14')
plt.bar(br15, time_double_kernel6_cf15, color='yellow', width=barWidth,
        edgecolor='grey', label='Course Factor 15')
plt.bar(br16, time_double_kernel6_cf16, color='magenta', width=barWidth,
        edgecolor='grey', label='Course Factor 16')
plt.xlabel('Input Size', fontweight='bold', fontsize=15)
plt.ylabel('Time in Milliseconds', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(input_s))],
           input_s)
plt.title("Performance Comparison Chart for Double Precision")
plt.legend()

plt.show()
