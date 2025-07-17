eval_type=$1

if [ "$eval_type" == "noisy" ]; then
    echo "Evaluation on **NOISY DATASET**"
    model_name="operaCT"
    dim=768
    python -u src/benchmark/linear_noisy_eval.py --task coughvidcovid --pretrain $model_name --dim $dim

    model_name="operaCE"
    dim=1280
    python -u src/benchmark/linear_noisy_eval.py --task coughvidcovid --pretrain $model_name --dim $dim

    model_name="operaGT"
    dim=384
    python -u src/benchmark/linear_noisy_eval.py --task coughvidcovid --pretrain $model_name --dim $dim

    list="opensmile vggish clap audiomae"
    for i in $list;
    do
    python -u src/benchmark/linear_noisy_eval.py --task coughvidcovid --pretrain $i
    done

else
    echo "Evaluation on **CLEAN DATASET**"
    model_name="operaCT"
    dim=768
    python -u src/benchmark/processing/coughvid_processing.py --pretrain $model_name --dim $dim
    python -u src/benchmark/linear_eval.py --task coughvidcovid --pretrain $model_name --dim $dim

    model_name="operaCE"
    dim=1280
    python -u src/benchmark/processing/coughvid_processing.py --pretrain $model_name --dim $dim
    python -u src/benchmark/linear_eval.py --task coughvidcovid --pretrain $model_name --dim $dim

    model_name="operaGT"
    dim=384
    python -u src/benchmark/processing/coughvid_processing.py --pretrain $model_name --dim $dim
    python -u src/benchmark/linear_eval.py --task coughvidcovid --pretrain $model_name --dim $dim

    list="opensmile vggish clap audiomae"
    for i in $list;
    do
    python -u src/benchmark/processing/coughvid_processing.py --pretrain $i 
    python -u src/benchmark/linear_eval.py --task coughvidcovid --pretrain $i
    done
fi
