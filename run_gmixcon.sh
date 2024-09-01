
for seed in 1314
do 
    CUDA_VISIBLE_DEVICES=0  python -u ./src/g_mixcon.py --data_path . --model GIN --dataset IMDB-BINARY \
        --epoch 20 --lr 0.001 --saliency  False --seed=$seed  --log_screen True --batch_size 128 --num_hidden 64 \
        --lambda_value 0.5 --saliency_ratio 0.8  --criterion  SupConLoss --warmup 0 --augsup False --num_layers 4 
done
