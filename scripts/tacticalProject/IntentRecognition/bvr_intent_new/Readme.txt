命令行示意：
（1）python "make_intent_features_NEU_wingmask.py" 
--input "radar_pairs_with_wing_traj.csv" 
--output "features_NEU_wingmask.csv"
（2）python "prepare_intent_dataset_13.py" 
--input "features_NEU_wingmask.csv" 
--out_dir "xxx" 
--T 32 
--S 1 
--use-status
-embed 
--perclass-step "防御=4,攻击=2,侦察=1,规避=1,撤退=1"
（3）python "train_intent_bilstm_from_npz.py" 
--data_dir "xxx" 
--out_dir "train_result" 
--epochs 100 
--batch_size 128 
--hidden_dim 512 
--num_layers 1 
--lr 8e-4 
--use-status-embed 
--use-self-attn 
--attn-dim 64 
--dropout 0.1 
--use-class-weight