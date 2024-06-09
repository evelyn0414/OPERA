python -u src/eval/cola_finetuning.py --task icbhidisease --pretrain operaCT --dim 768
python -u src/eval/cola_finetuning.py --task icbhidisease --pretrain operaCE --dim 1280
python -u src/eval/cola_finetuning.py --task icbhidisease --pretrain operaGT --dim 384
python -u src/eval/cola_finetuning.py --task icbhidisease --pretrain audiomae --dim 768 
python -u src/eval/cola_finetuning.py --task icbhidisease --pretrain clap --dim 1024

# python -u src/eval/cola_finetuning.py --task snoring --pretrain operaCT --dim 768
# python -u src/eval/cola_finetuning.py --task snoring --pretrain operaCE --dim 1280
# python -u src/eval/cola_finetuning.py --task snoring --pretrain operaGT --dim 384
# python -u src/eval/cola_finetuning.py --task snoring --pretrain audiomae --dim 768
# python -u src/eval/cola_finetuning.py --task snoring --pretrain clap --dim 1024

# python -u src/eval/cola_finetuning.py --task covidtask1downsample --modality cough --pretrain audiomae --dim 768 
# python -u src/eval/cola_finetuning.py --task covidtask1downsample --modality cough --pretrain clap --dim 1024
# python -u src/eval/cola_finetuning.py --task covidtask1downsample --modality cough --pretrain operaCT --dim 768 
# python -u src/eval/cola_finetuning.py --task covidtask1downsample --modality cough --pretrain  operaCE --dim 1280 
# python -u src/eval/cola_finetuning.py --task covidtask1downsample --modality cough --pretrain operaGT --dim 384

# python -u src/eval/cola_finetuning.py --task covidtask1 --modality cough --pretrain audiomae --dim 768
# python -u src/eval/cola_finetuning.py --task covidtask1 --modality cough --pretrain clap --dim 1024
# python -u src/eval/cola_finetuning.py --task covidtask1 --modality cough --pretrain operaCT --dim 768 
# python -u src/eval/cola_finetuning.py --task covidtask1 --modality cough --pretrain  operaCE --dim 1280
# python -u src/eval/cola_finetuning.py --task covidtask1 --modality cough --pretrain operaGT --dim 384


