source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate deepseek;

cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;

langs=("en" "ch")

for lang in "${langs[@]}" 
do
    python test_exit.py
done