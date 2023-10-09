import numpy as np
import statistics

# datasets
data = np.array([27, 27, 27, 30, 31, 32, 33, 40, 44, 44, 
                 44, 45, 45, 48, 48, 51, 51, 53, 55, 55,
                 55, 57, 61, 62, 65, 65, 65, 65, 65, 65,
                 65, 65, 65, 65, 65, 67, 69, 70, 70, 70,
                 70, 73, 73, 79, 80, 80, 81, 81, 83, 83,
                 85, 85, 87, 88, 88, 88, 90, 91, 93, 93
                 ])
jumlah = len(data)

f_20_to_29 = 0
f_30_to_39 = 0
f_40_to_49 = 0
f_50_to_59 = 0
f_60_to_69 = 0
f_70_to_79 = 0
f_80_to_89 = 0
f_90_to_99 = 0

# making frequency table
for i in range(jumlah):
    if data[i] >= 20 and data[i] <= 29:
        f_20_to_29 += 1
    elif data[i] >= 30 and data[i] <= 39:
        f_30_to_39 += 1
    elif data[i] >= 40 and data[i] <= 49:
        f_40_to_49 += 1
    elif data[i] >= 50 and data[i] <= 59:
        f_50_to_59 += 1
    elif data[i] >= 60 and data[i] <= 69:
        f_60_to_69 += 1
    elif data[i] >= 70 and data[i] <= 79:
        f_70_to_79 += 1
    elif data[i] >= 80 and data[i] <= 89:
        f_80_to_89 += 1
    elif data[i] >= 90 and data[i] <= 99:
        f_90_to_99 += 1

print(f"|20 - 29 | {f_20_to_29}\t|\n"
      f"|30 - 39 | {f_30_to_39}\t|\n"
      f"|40 - 49 | {f_40_to_49}\t|\n"
      f"|50 - 59 | {f_50_to_59}\t|\n"
      f"|60 - 69 | {f_60_to_69}\t|\n"
      f"|70 - 79 | {f_70_to_79}\t|\n"
      f"|80 - 89 | {f_80_to_89}\t|\n"
      f"|90 - 99 | {f_90_to_99}\t|")

# find mean
mean = np.mean(data)
print(f"mean = {mean:.2f}")

# find median
median = np.median(data)
print(f"median = {median}")

# find modus
modus = statistics.mode(data)
print(f"Modus = {modus}")

# find quartil(Q3)
q3 = np.percentile(data, 75)
print(f"Q3 (Kuartil Ketiga) = {q3}")