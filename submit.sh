output_path=problems/flashattention/outputs/
if [ ! -d ${output_path} ]; then
    mkdir -p ${output_path}
fi

export PATH="$PATH:/Users/juntongshi/Desktop/cs149/asst5-kernels/binary"
cd problems/flashattention
python wrap_cuda_submission.py
cd ../../
popcorn-cli submit --leaderboard flashattention --mode profile problems/flashattention/submission.py