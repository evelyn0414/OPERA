python -u src/pretrain/prepare_data/icbhi_pressl.py

list="breath cough"  
for i in $list;  
do  
echo $i preprocessing to fmax8000 spec;  
python -u src/pretrain/prepare_data/covid19sounds_pressl.py --modality $i

done 
