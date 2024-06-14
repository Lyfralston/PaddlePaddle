local task = "ED";
local dataset = "ace05-en";
local split = 1;
local model_type = "RCEE";
local pretrained_model_name = "roberta-large";
local pretrained_model_alias = {
    "roberta-large": "roberta-large", 
};

{
    // general config
    "task": task, 
    "dataset": dataset,
    "model_type": model_type, 
    "gpu_device": 0, 
    "seed": 0, 
    "cache_dir": "./cache", 
    "output_dir": "./outputs/%s_%s_%s_split%s_%s" % [model_type, task, dataset, split, pretrained_model_alias[pretrained_model_name]], 
    "train_file": "./data/processed_data/%s/split%s/train.json" % [dataset, split],
    "dev_file": "./data/processed_data/%s/split%s/dev.json" % [dataset, split],
    "test_file": "./data/processed_data/%s/split%s/test.json" % [dataset, split],
    
    // model config
    "pretrained_model_name": pretrained_model_name,
    "base_model_dropout": 0.2,
    "linear_dropout": 0.2,
    "linear_bias": true, 
    "linear_activation": "relu",
    "multi_piece_strategy": "average", 
    "max_length": 200, 
    "question_type": "[EVENT]",
    
    // train config
    "max_epoch": 30,
    "warmup_epoch": 5,
    "accumulate_step": 1,
    "train_batch_size": 6,
    "eval_batch_size": 12,
    "learning_rate": 0.001,
    "base_model_learning_rate": 1e-05,
    "weight_decay": 0.001,
    "base_model_weight_decay": 1e-05,
    "grad_clipping": 5.0,
}
