python -u src/benchmark/other_eval/finetuning.py --task coughvidcovid --pretrain operaCT --dim 768
python -u src/benchmark/other_eval/finetuning.py --task coughvidcovid --pretrain operaCE --dim 1280
python -u src/benchmark/other_eval/finetuning.py --task coughvidcovid --pretrain operaGT --dim 384
python -u src/benchmark/other_eval/finetuning.py --task coughvidcovid --pretrain audiomae --dim 768 
python -u src/benchmark/other_eval/finetuning.py --task coughvidcovid --pretrain clap --dim 1024

# python -u src/benchmark/other_eval/finetuning.py --task icbhidisease --pretrain operaCT --dim 768
# python -u src/benchmark/other_eval/finetuning.py --task icbhidisease --pretrain operaCE --dim 1280
# python -u src/benchmark/other_eval/finetuning.py --task icbhidisease --pretrain operaGT --dim 384
# python -u src/benchmark/other_eval/finetuning.py --task icbhidisease --pretrain audiomae --dim 768 
# python -u src/benchmark/other_eval/finetuning.py --task icbhidisease --pretrain clap --dim 1024

# python -u src/benchmark/other_eval/finetuning.py --task snoring --pretrain operaCT --dim 768
# python -u src/benchmark/other_eval/finetuning.py --task snoring --pretrain operaCE --dim 1280
# python -u src/benchmark/other_eval/finetuning.py --task snoring --pretrain operaGT --dim 384
# python -u src/benchmark/other_eval/finetuning.py --task snoring --pretrain audiomae --dim 768
# python -u src/benchmark/other_eval/finetuning.py --task snoring --pretrain clap --dim 1024

# python -u src/benchmark/other_eval/finetuning.py --task covid19soundsdownsample --modality cough --pretrain audiomae --dim 768 
# python -u src/benchmark/other_eval/finetuning.py --task covid19soundsdownsample --modality cough --pretrain clap --dim 1024
# python -u src/benchmark/other_eval/finetuning.py --task covid19soundsdownsample --modality cough --pretrain operaCT --dim 768 
# python -u src/benchmark/other_eval/finetuning.py --task covid19soundsdownsample --modality cough --pretrain  operaCE --dim 1280 
# python -u src/benchmark/other_eval/finetuning.py --task covid19soundsdownsample --modality cough --pretrain operaGT --dim 384

# python -u src/benchmark/other_eval/finetuning.py --task covid19sounds --modality cough --pretrain audiomae --dim 768
# python -u src/benchmark/other_eval/finetuning.py --task covid19sounds --modality cough --pretrain clap --dim 1024
# python -u src/benchmark/other_eval/finetuning.py --task covid19sounds --modality cough --pretrain operaCT --dim 768 
# python -u src/benchmark/other_eval/finetuning.py --task covid19sounds --modality cough --pretrain  operaCE --dim 1280
# python -u src/benchmark/other_eval/finetuning.py --task covid19sounds --modality cough --pretrain operaGT --dim 384


