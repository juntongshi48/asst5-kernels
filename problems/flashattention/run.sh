export PYTHONPATH="<path-to-asst5>/problems:$PYTHONPATH"
python wrap_cuda_submission.py
CUDA_VISIBLE_DEVICES=3 python ../eval.py test test_cases/test.txt