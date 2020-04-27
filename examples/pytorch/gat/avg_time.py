import sys

count = 0
s = 0.
fs = 0.
skip_first = True
first_iter_time = 0.
for line in sys.stdin:
    if "It takes" in line:
        if skip_first:
            skip_first = False
            first_iter_time = float(line.split()[2]) 
            continue
        s += float(line.split()[2])
        fs += float(line.split()[9])
        count += 1
    if "rst2(gat)" in line:
        print(line)
print("On average it takes ", s/count, "s"," avg foward fraction " ,fs/count*100,"% first iter time ", first_iter_time, "s")
