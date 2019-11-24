# VLN PPO Agent Details

## Configurations
`configs/tasks/vln_r2r.yaml`
`habitat_baselines/config/ppo_vln.yaml`

`DATA_PATH: "./data/datasets/vln/r2r/v1/preprocessed/{split}/{split}.json.gz"`  - This uses a dataset preprocessed to work with glove.6B.50d.txt embeddings.

`embedding_file: "./data/datasets/vln/r2r/v1/preprocessed/embeddings.json"`
- This file has the embedddings for all R2R words pulled from glove.6B.50d.txt.
- Must be used with DATA_PATH specified above.
