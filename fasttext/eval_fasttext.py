eval_file = 'pred_eval.txt'
test_file = 'pred_test.txt'

with open("twitter-datasets/eval_data.txt", "r") as f:
    labels = []
    for line in f.readlines():
        labels.append(int(line.strip().split(",")[0]))

with open(eval_file, "r") as f:
    correct = 0
    total = 0
    for i, line in enumerate(f.readlines()):
        label = int(line.strip()[-1])
        if label == labels[i]:
            correct += 1
        total += 1
    print("Eval acc", correct / float(total))

with open(test_file, 'r') as f:
    with open(test_file + '.kaggle.csv', 'w') as w:
        print('Id,Prediction', file=w)
        for i, line in enumerate(f.readlines()):
            label = 2 * int(line.strip()[-1]) - 1
            print('{},{}'.format(i + 1, label), file=w)
