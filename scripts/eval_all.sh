# to run the script, copy feature, scripts datasets and src directories

# script example for running: sh scripts/eval_all.sh multiple150 1280  >> out/multiple150.txt
pretrain_model=$1
dim=$2
echo extracting feature from $pretrain_model for downstream tasks;

python -u src/eval/coviduk_processing.py \
        --pretrain $pretrain_model\
        --modality exhalation\
        --dim $dim

python -u src/eval/coviduk_processing.py \
        --pretrain $pretrain_model\
        --modality cough\
        --dim $dim
        
python -u src/eval/covid19sounds_processing.py --task 1\
        --pretrain $pretrain_model\
        --modality breath\
        --dim $dim

python -u src/eval/covid19sounds_processing.py --task 1\
        --pretrain $pretrain_model\
        --modality cough\
        --dim $dim

python -u src/eval/coswara_processing.py --pretrain $pretrain_model\
        --modality cough-shallow\
        --label smoker\
        --input_sec 2\
        --dim $dim

python -u src/eval/coswara_processing.py --pretrain $pretrain_model\
        --modality cough-shallow\
        --label sex\
        --input_sec 2\
        --dim $dim

python -u src/eval/coughvid_processing.py \
        --pretrain $pretrain_model\
        --label covid\
        --dim $dim

python -u src/eval/coughvid_processing.py \
        --pretrain $pretrain_model\
        --label gender\
        --dim $dim

python -u src/eval/ssbpr_processing.py \
        --pretrain $pretrain_model\
        --dim $dim

python -u src/eval/icbhi_processing.py  --pretrain $pretrain_model --dim $dim

python -u src/eval/kauh_processing.py --pretrain $pretrain_model --dim $dim

python -u src/eval/copd_processing.py --pretrain $pretrain_model --dim $dim

python -u src/eval/mmlung_processing.py --pretrain $pretrain_model --dim $dim

echo linear evaluation of $pretrain_model on all downstream tasks;


python src/eval/linear_eval.py --task coviduk\
        --pretrain $pretrain_model\
        --modality exhalation\
        --dim $dim

python src/eval/linear_eval.py --task coviduk\
        --pretrain $pretrain_model\
        --modality cough\
        --dim $dim

python src/eval/linear_eval.py --task covidtask1\
        --modality breath\
        --pretrain $pretrain_model\
        --dim $dim

python src/eval/linear_eval.py --task covidtask1\
        --modality cough\
        --pretrain $pretrain_model\
        --dim $dim

python src/eval/linear_eval.py --task coswarasmoker\
        --modality cough-shallow\
        --pretrain $pretrain_model\
        --mapgoogle True\
        --dim $dim

python src/eval/linear_eval.py --task coswarasex\
        --modality cough-shallow\
        --pretrain $pretrain_model\
        --mapgoogle True\
        --dim $dim

python src/eval/linear_eval.py --task coughvidcovid --pretrain $pretrain_model --dim $dim

python src/eval/linear_eval.py --task coughvidsex --pretrain $pretrain_model --dim $dim

python src/eval/linear_eval.py --task icbhi  --pretrain $pretrain_model --dim $dim

python src/eval/linear_eval.py --task kauh --pretrain $pretrain_model  --dim $dim

python src/eval/linear_eval.py --task copd --pretrain $pretrain_model  --dim $dim

python src/eval/linear_eval.py --task snoring --pretrain $pretrain_model  --dim $dim


modality="cough breath vowels" 
label="FVC FEV1 FEV1_FVC"
for m in $modality;  
do
for y in $label;
do
echo $pretrain_model is being evaluated on mmlung data - $m for $y;  
python src/eval/linear_eval.py --pretrain $pretrain_model --task spirometry --label $y --modality $m --LOOCV True  --dim $dim
done
done
