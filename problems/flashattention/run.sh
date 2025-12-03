export PYTHONPATH="/data/jiaqi/juntong/asst5-kernels/problems:$PYTHONPATH"
python wrap_cuda_submission.py
mode=$1
debug=$2
gpuid=6
if [ "$debug" = "true" ]; then
    CUDA_VISIBLE_DEVICES=${gpuid} python ../eval.py ${mode} test_cases/test_correctness.txt
else
    CUDA_VISIBLE_DEVICES=${gpuid} python ../eval.py ${mode} test_cases/test.txt
fi