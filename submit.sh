output_path=problems/flashattention/outputs/
if [ ! -d ${output_path} ]; then
    mkdir -p ${output_path}
fi

export PATH="$PATH:/data/jiaqi/juntong/asst5-kernels/binarys"
popcorn-cli submit --leaderboard flash_attn_turbo --mode profile problems/flashattention/wrap_cuda_submission.py