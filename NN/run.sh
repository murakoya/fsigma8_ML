OUTPUT_LOG=./results/log_train.out

date >> $OUTPUT_LOG
python -u calc.py >> $OUTPUT_LOG

date >> $OUTPUT_LOG

echo "##################################\n" >> $OUTPUT_LOG
