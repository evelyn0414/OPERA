
## runing baselines
sh scripts/eval_all.sh opensmile  >> result_opensmile.log
sh scripts/eval_all.sh vggish  >> result_vggish.log
sh scripts/eval_all.sh clap  >> result_clap.log
sh scripts/eval_all.sh audiomae  >> result_audiomae.log

## runing opera-X
sh scripts/eval_all.sh operaCT 768  >> result_operaCT.log
sh scripts/eval_all.sh operaCE 1280  >> result_operaCE.log
sh scripts/eval_all.sh operaGT 384 >> result_operaGT.log