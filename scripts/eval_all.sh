# to run the script, copy feature, scripts datasets and src directories
# script example for running: sh scripts/eval_all.sh OPERA-GT 1280  >> out/multiple150.txt

pretrain_model=$1
if [ $# -gt 1 ]; then
        dim=$2
        echo 'Feature dimension:' $dim
else
        dim=0
        echo 'Baseline: no need to specify dimension'
fi

echo extracting feature from $pretrain_model for downstream tasks;
python -u src/benchmark/processing/nosemic_processing.py --pretrain $pretrain_model --dim $dim

echo linear evaluation of $pretrain_model on downstream tasks;
python -u src/benchmark/linear_eval.py --pretrain $pretrain_model --task rr --LOOCV True --dim $dim




# python -u src/benchmark/coviduk_processing.py \
#         --pretrain $pretrain_model\
#         --modality exhalation\
#         --dim $dim

# python -u src/benchmark/coviduk_processing.py \
#         --pretrain $pretrain_model\
#         --modality cough\
#         --dim $dim
        
# python -u src/benchmark/covid19sounds_processing.py --task 1\
#         --pretrain $pretrain_model\
#         --modality breath\
#         --dim $dim

# python -u src/benchmark/covid19sounds_processing.py --task 1\
#         --pretrain $pretrain_model\
#         --modality cough\
#         --dim $dim

# python -u src/benchmark/coswara_processing.py --pretrain $pretrain_model\
#         --modality cough-shallow\
#         --label smoker\
#         --input_sec 2\
#         --dim $dim

# python -u src/benchmark/coswara_processing.py --pretrain $pretrain_model\
#         --modality cough-shallow\
#         --label sex\
#         --input_sec 2\
#         --dim $dim

# python -u src/benchmark/coughvid_processing.py \
#         --pretrain $pretrain_model\
#         --label covid\
#         --dim $dim

# python -u src/benchmark/coughvid_processing.py \
#         --pretrain $pretrain_model\
#         --label gender\
#         --dim $dim

# python -u src/benchmark/ssbpr_processing.py \
#         --pretrain $pretrain_model\
#         --dim $dim

# python -u src/benchmark/icbhi_processing.py  --pretrain $pretrain_model --dim $dim

# python -u src/benchmark/kauh_processing.py --pretrain $pretrain_model --dim $dim

# python -u src/benchmark/copd_processing.py --pretrain $pretrain_model --dim $dim

# python -u src/benchmark/mmlung_processing.py --pretrain $pretrain_model --dim $dim

# echo linear evaluation of $pretrain_model on all downstream tasks;


# python src/benchmark/linear_eval.py --task coviduk\
#         --pretrain $pretrain_model\
#         --modality exhalation\
#         --dim $dim

# python src/benchmark/linear_eval.py --task coviduk\
#         --pretrain $pretrain_model\
#         --modality cough\
#         --dim $dim

# python src/benchmark/linear_eval.py --task covid19sounds\
#         --modality breath\
#         --pretrain $pretrain_model\
#         --dim $dim

# python src/benchmark/linear_eval.py --task covid19sounds\
#         --modality cough\
#         --pretrain $pretrain_model\
#         --dim $dim

# python src/benchmark/linear_eval.py --task coswarasmoker\
#         --modality cough-shallow\
#         --pretrain $pretrain_model\
#         --mapgoogle True\
#         --dim $dim

# python src/benchmark/linear_eval.py --task coswarasex\
#         --modality cough-shallow\
#         --pretrain $pretrain_model\
#         --mapgoogle True\
#         --dim $dim

# python src/benchmark/linear_eval.py --task coughvidcovid --pretrain $pretrain_model --dim $dim

# python src/benchmark/linear_eval.py --task coughvidsex --pretrain $pretrain_model --dim $dim

# python src/benchmark/linear_eval.py --task icbhi  --pretrain $pretrain_model --dim $dim

# python src/benchmark/linear_eval.py --task kauh --pretrain $pretrain_model  --dim $dim

# python src/benchmark/linear_eval.py --task copd --pretrain $pretrain_model  --dim $dim

# python src/benchmark/linear_eval.py --task snoring --pretrain $pretrain_model  --dim $dim


# modality="cough breath vowels" 
# label="FVC FEV1 FEV1_FVC"
# for m in $modality;  
# do
# for y in $label;
# do
# echo $pretrain_model is being evaluated on mmlung data - $m for $y;  
# python src/benchmark/linear_eval.py --pretrain $pretrain_model --task spirometry --label $y --modality $m --LOOCV True  --dim $dim
# done
# done
