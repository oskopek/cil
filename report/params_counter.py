import fileinput
from functools import reduce

total_num = 0
for line in fileinput.input():
    line = line.replace("shape=(", "").replace(",)","").replace(")", "").replace(" ", "")
    numbers = [int(num) for num in line.split(",")]
    product = reduce(lambda x, y: x*y, numbers)
    total_num += product

print("Total num of vars: {}".format(total_num))
