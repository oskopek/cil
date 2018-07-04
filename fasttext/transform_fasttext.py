for dataset in ["train", "eval", "test"]:
    lines = []
    with open("{}_data.txt".format(dataset), "r") as f:
        for line in f.readlines():
            label, words = line.strip().split(",", maxsplit=1)
            words = words.split(" ")
            words.append("__label__{}".format(label))
            lines.append(" ".join(words))

    with open("fasttext_{}_data.txt".format(dataset), "w") as f:
        for line in lines:
            print(line, file=f)
