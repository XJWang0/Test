for ((i=20;i<=450;i+=20))
do
  python -u main_smart_informer.py \
        --model cp_informer \
        --data ETTh1 \
        --features S \
        --seq_len 720 \
        --label_len 168 \
        --pred_len 24 \
        --rank $i \
        --e_layers 2 \
        --d_layers 1 \
        --attn prob \
        --des 'CP_Exp' \
        --itr 5
done