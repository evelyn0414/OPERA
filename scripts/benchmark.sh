
## runing baselines
sh scripts/eval_all.sh opensmile  >> cks/logs/result_opensmile.log
sh scripts/eval_all.sh vggish  >> cks/logs/result_vggish.log
sh scripts/eval_all.sh audiomae  >> cks/logs/result_audiomae.log
sh scripts/eval_all.sh clap  >> cks/logs/result_clap.log


## runing opera-X
sh scripts/eval_all.sh operaCT 768  >> cks/logs/result_operaCT.log
sh scripts/eval_all.sh operaCE 1280  >> cks/logs/result_operaCE.log
sh scripts/eval_all.sh operaGT 384 >> cks/logs/result_operaGT.log
