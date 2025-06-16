echo "Running inference for NBDM..."
python code/processing/sanity_datasets/nbdm_gamma_sanity_datasets.py
python code/processing/sanity_datasets/nbdm_lognormal_sanity_datasets.py
echo "Running inference for ZINB..."
python code/processing/sanity_datasets/zinb_gamma_sanity_datasets.py
python code/processing/sanity_datasets/zinb_lognormal_sanity_datasets.py
echo "Running inference for NBVCP..."
python code/processing/sanity_datasets/nbvcp_gamma_sanity_datasets.py
python code/processing/sanity_datasets/nbvcp_lognormal_sanity_datasets.py
echo "Running inference for ZINBVCP..."
python code/processing/sanity_datasets/zinbvcp_gamma_sanity_datasets.py
python code/processing/sanity_datasets/zinbvcp_lognormal_sanity_datasets.py
