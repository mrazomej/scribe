echo "Running inference for NBDM..."
python code/processing/sanity_datasets/svi_odds-ratio_nbdm_sanity_fit.py
python code/processing/sanity_datasets/svi_odds-ratio_nbdm_sanity_viz.py
echo "Running inference for ZINB..."
python code/processing/sanity_datasets/svi_odds-ratio_zinb_sanity_fit.py
python code/processing/sanity_datasets/svi_odds-ratio_zinb_sanity_viz.py
echo "Running inference for NBVCP..."
python code/processing/sanity_datasets/svi_odds-ratio_nbvcp_sanity_fit.py
python code/processing/sanity_datasets/svi_odds-ratio_nbvcp_sanity_viz.py
echo "Running inference for ZINBVCP..."
python code/processing/sanity_datasets/svi_odds-ratio_zinbvcp_sanity_fit.py
python code/processing/sanity_datasets/svi_odds-ratio_zinbvcp_sanity_viz.py
