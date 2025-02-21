echo "Generating plots for NBDM..."
python code/processing/sanity_datasets/nbdm_gamma_sanity_viz.py
python code/processing/sanity_datasets/nbdm_lognormal_sanity_viz.py
echo "Generating plots for ZINB..."
python code/processing/sanity_datasets/zinb_gamma_sanity_viz.py
python code/processing/sanity_datasets/zinb_lognormal_sanity_viz.py
echo "Generating plots for NBVCP..."
python code/processing/sanity_datasets/nbvcp_gamma_sanity_viz.py
python code/processing/sanity_datasets/nbvcp_lognormal_sanity_viz.py
echo "Generating plots for ZINBVCP..."
python code/processing/sanity_datasets/zinbvcp_gamma_sanity_viz.py
python code/processing/sanity_datasets/zinbvcp_lognormal_sanity_viz.py
