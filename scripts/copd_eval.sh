model_name="operaCT"
dim=768
python src/eval/copd_processing.py --pretrain $model_name --dim $dim
python src/eval/linear_eval.py --task copd --pretrain $model_name --dim $dim

model_name="operaCE"
dim=1280
python src/eval/copd_processing.py --pretrain $model_name --dim $dim
python src/eval/linear_eval.py --task copd --pretrain $model_name --dim $dim


# list="opensmile vggish clap audiomae"  
# for i in $list;  
# do  
# python src/eval/copd_processing.py --pretrain $i 
# python src/eval/linear_eval.py --task copd --pretrain $i
# done 
