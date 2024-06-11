python -u src/pretrain/cola_training.py --data multiple\
        --covidbreath True\
        --covidcough True\
        --icbhi True\
        --coughvid True\
        --hf_lung True\
        --covidUKexhalation True\
        --covidUKcough True\
        --encoder efficientnet\
        --title operaCE-test\
        --epoches 150


python -u src/pretrain/cola_training.py --data multiple\
        --covidbreath True\
        --covidcough True\
        --icbhi True\
        --coughvid True\
        --hf_lung True\
        --covidUKexhalation True\
        --covidUKcough True\
        --encoder htsat\
        --title operaCT-test\
        --epoches 250


python -u src/pretrain/mae_training.py --data multiple\
        --covidbreath True\
        --covidcough True\
        --icbhicycle True\
        --coughvid True\
        --hf_lung True\
        --covidUKexhalation True\
        --covidUKcough True\
        --encoder vit\
        --title operaGT-test\
        --epoches 100