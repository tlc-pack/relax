import csv
import sys
from collections import defaultdict

FILE = "ncu_output.csv"

records = defaultdict(list)

with open(FILE, "r", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        if len(lines) > 12:
            name, metric_name, metric_val = lines[4], lines[9], lines[11]
            if metric_name == "Duration":
                metric_val = float(metric_val) * 0.001
                records[name].append(metric_val)
                # print(name, metric_val)
            # if metric_name in metrics:
            #     val = metric_val.replace(',',"")
            #     metrics_val[metric_name].append(float(val))

for name, vals in records.items():
    print(f"\"{name}\",{sum(vals)}")
