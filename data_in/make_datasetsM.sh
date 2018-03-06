printf '1\n%.0s' {1..1250000} > labels.txt
printf '0\n%.0s' {1..1250000} >> labels.txt

head -n 1250000 labels.txt >labels_pos.txt
tail -n 1250000 labels.txt >labels_neg.txt

head -n 1200000 labels_pos.txt > t_pos_labels.txt
head -n 1200000 labels_neg.txt > t_neg_labels.txt

head -n 1200000 train_pos_full.txt > t_pos.txt0
./preprocess.sh t_pos.txt0 > t_pos.txt
head -n 1200000 train_neg_full.txt > t_neg.txt0
./preprocess.sh t_neg.txt0 > t_neg.txt

tail -n 50000 train_pos_full.txt > valid.tmp0
tail -n 50000 train_neg_full.txt >> valid.tmp0
./preprocess.sh valid.tmp0 > valid.tmp 

head -n 50000 labels_pos.txt > valid_labels.tmp
head -n 50000 labels_neg.txt >> valid_labels.tmp

paste -d 'M' valid.tmp valid_labels.tmp | shuf | awk -v FS="M" '{ print $1 > "valid.txt" ; print $2 > "valid_labels.txt" }'

rm t_pos.txt0
rm t_neg.txt0
rm valid.tmp0
rm valid.tmp
rm valid_labels.tmp