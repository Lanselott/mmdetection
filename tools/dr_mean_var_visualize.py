import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_axes([0.05, 0.09, 0.9, 0.9])

x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
iou_mean = [
    0.3465, 0.6872, 0.7098, 0.7229, 0.7315, 0.7277, 0.7340, 0.7547, 0.7652,
    0.7705, 0.7830, 0.7729
]

dr_iou_mean = [
    0.4734, 0.7961, 0.8122, 0.8219, 0.8319, 0.8280, 0.8357, 0.8486, 0.8553,
    0.8596, 0.8688, 0.8621
]

iou_var = [
    0.047, 0.045, 0.033, 0.043, 0.046, 0.056, 0.032, 0.037, 0.031, 0.041,
    0.038, 0.044
]
dr_iou_var = [
    0.02, 0.024, 0.017, 0.013, 0.015, 0.022, 0.010, 0.009, 0.011, 0.015, 0.011,
    0.012
]
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# ax1.set_title("sss")
ax1.set_xlabel("epoch", fontsize=28)
ax1.set_ylabel("mean of IoUs", fontsize=28)
iou_line = ax1.plot(x_axis, iou_mean, color='blue')
dr_iou_line = ax1.plot(x_axis, dr_iou_mean, color='red')

plt.errorbar(x_axis, iou_mean, iou_var, capsize=4)
plt.errorbar(x_axis, dr_iou_mean, dr_iou_var, capsize=4)

plt.legend(['IoU without D&R', 'IoU with D&R'], prop={'size': 24})
plt.grid(True)
plt.savefig('dr-compare-appendix.pdf')
# plt.show()