# Multi-seed summary

Equivalence margin: ±0.02 balanced accuracy (TOST, 90% CI).

## confirmatory

| Model | Bal. acc. | SD seeds | SD folds | Rec. mal. | AUC |
|---|---|---|---|---|---|
| SVM_RBF | 0.975 | 0.000 | 0.018 | 0.962 | 0.994 |
| SVM_LIN | 0.959 | 0.000 | 0.031 | 0.944 | 0.991 |
| SVM_LIN_PCA4 | 0.959 | 0.000 | 0.036 | 0.934 | 0.994 |
| MLP_PCA8 | 0.956 | 0.002 | 0.035 | 0.930 | 0.993 |
| RF | 0.952 | 0.005 | 0.019 | 0.935 | 0.990 |
| SVM_RBF_PCA4 | 0.948 | 0.000 | 0.017 | 0.929 | 0.989 |
| MLP_PCA4 | 0.946 | 0.002 | 0.025 | 0.922 | 0.982 |
| MLP_30D | 0.937 | 0.003 | 0.030 | 0.918 | 0.982 |
| MLP_H6 | 0.933 | 0.016 | 0.039 | 0.892 | 0.980 |
| MLP_extractor | 0.912 | 0.032 | 0.056 | 0.857 | 0.972 |
| HYBRID_MLP_VQC | 0.904 | 0.014 | 0.025 | 0.854 | 0.968 |
| QSVM_ZZ4 | 0.859 | 0.000 | 0.030 | 0.797 | 0.940 |
| VQC_8 | 0.842 | 0.006 | 0.029 | 0.694 | 0.973 |

## vqc_ce80

| Model | Bal. acc. | SD seeds | SD folds | Rec. mal. | AUC |
|---|---|---|---|---|---|
| SVM_RBF | 0.975 | 0.000 | 0.018 | 0.962 | 0.994 |
| SVM_LIN | 0.959 | 0.000 | 0.031 | 0.944 | 0.991 |
| MLP_PCA8 | 0.956 | 0.002 | 0.035 | 0.930 | 0.993 |
| RF | 0.952 | 0.005 | 0.019 | 0.935 | 0.990 |
| MLP_30D | 0.937 | 0.003 | 0.030 | 0.918 | 0.982 |
| MLP_H6 | 0.933 | 0.016 | 0.039 | 0.892 | 0.980 |
| MLP_extractor | 0.912 | 0.032 | 0.056 | 0.857 | 0.972 |
| HYBRID_MLP_VQC | 0.903 | 0.008 | 0.027 | 0.856 | 0.969 |
| VQC_8 | 0.837 | 0.004 | 0.028 | 0.679 | 0.972 |

## vqc_boost

| Model | Bal. acc. | SD seeds | SD folds | Rec. mal. | AUC |
|---|---|---|---|---|---|
| vqc_allz_ce_L4 | 0.965 | 0.009 | 0.014 | 0.953 | 0.995 |
| vqc_reup2_allz_ce_L2 | 0.965 | 0.007 | 0.008 | 0.953 | 0.994 |
| linear_pca8 | 0.960 | 0.004 | 0.032 | 0.924 | 0.994 |
| mlp_pca8 | 0.956 | 0.002 | 0.035 | 0.930 | 0.993 |
| vqc_allz_ce_L2 | 0.953 | 0.002 | 0.026 | 0.929 | 0.992 |
| vqc_z0_ce_head_L2 | 0.922 | 0.007 | 0.026 | 0.890 | 0.984 |
| baseline_z0_mse_L2 | 0.842 | 0.006 | 0.029 | 0.694 | 0.973 |

## Contrasts (seed-averaged fold scores, Nadeau–Bengio corrected)

| Contrast | Δ | 95% CI | p | Equivalent (±margin) |
|---|---|---|---|---|
| confirmatory:VQC_8-MLP_PCA8:balanced_accuracy | -0.113 | [-0.143, -0.084] | 0.00046 | no |
| confirmatory:VQC_8-MLP_PCA8:recall_malignant | -0.237 | [-0.331, -0.143] | 0.0022 | no |
| confirmatory:VQC_8-SVM_RBF:balanced_accuracy | -0.133 | [-0.179, -0.087] | 0.0013 | no |
| confirmatory:VQC_8-SVM_RBF:recall_malignant | -0.269 | [-0.387, -0.150] | 0.0032 | no |
| confirmatory:HYBRID_MLP_VQC-MLP_H6:balanced_accuracy | -0.028 | [-0.065, +0.009] | 0.1 | no |
| confirmatory:HYBRID_MLP_VQC-MLP_H6:recall_malignant | -0.038 | [-0.125, +0.050] | 0.3 | no |
| confirmatory:HYBRID_MLP_VQC-MLP_30D:balanced_accuracy | -0.033 | [-0.055, -0.011] | 0.014 | no |
| confirmatory:HYBRID_MLP_VQC-MLP_30D:recall_malignant | -0.064 | [-0.138, +0.010] | 0.075 | no |
| confirmatory:QSVM_ZZ4-SVM_LIN_PCA4:balanced_accuracy | -0.099 | [-0.153, -0.046] | 0.0068 | no |
| confirmatory:QSVM_ZZ4-SVM_LIN_PCA4:recall_malignant | -0.137 | [-0.244, -0.030] | 0.024 | no |
| confirmatory:MLP_PCA8-SVM_RBF:balanced_accuracy | -0.020 | [-0.066, +0.027] | 0.31 | no |
| confirmatory:MLP_PCA8-SVM_RBF:recall_malignant | -0.032 | [-0.140, +0.076] | 0.46 | no |
| confirmatory:MLP_30D-SVM_RBF:balanced_accuracy | -0.038 | [-0.083, +0.007] | 0.078 | no |
| confirmatory:MLP_30D-SVM_RBF:recall_malignant | -0.044 | [-0.094, +0.005] | 0.068 | no |
| vqc_ce80:VQC_8-MLP_PCA8:balanced_accuracy | -0.119 | [-0.138, -0.100] | 6.6e-05 | no |
| vqc_ce80:HYBRID_MLP_VQC-MLP_H6:balanced_accuracy | -0.030 | [-0.067, +0.007] | 0.088 | no |
| vqc_boost:baseline_z0_mse_L2-linear_pca8:balanced_accuracy | -0.118 | [-0.159, -0.076] | 0.0014 | no |
| vqc_boost:baseline_z0_mse_L2-linear_pca8:recall_malignant | -0.231 | [-0.335, -0.126] | 0.0036 | no |
| vqc_boost:baseline_z0_mse_L2-mlp_pca8:balanced_accuracy | -0.113 | [-0.143, -0.084] | 0.00046 | no |
| vqc_boost:baseline_z0_mse_L2-mlp_pca8:recall_malignant | -0.237 | [-0.331, -0.143] | 0.0022 | no |
| vqc_boost:baseline_z0_mse_L2-SVM_RBF:balanced_accuracy | -0.133 | [-0.179, -0.087] | 0.0013 | no |
| vqc_boost:vqc_allz_ce_L2-linear_pca8:balanced_accuracy | -0.007 | [-0.029, +0.016] | 0.46 | no |
| vqc_boost:vqc_allz_ce_L2-linear_pca8:recall_malignant | +0.005 | [-0.042, +0.053] | 0.77 | no |
| vqc_boost:vqc_allz_ce_L2-mlp_pca8:balanced_accuracy | -0.002 | [-0.021, +0.016] | 0.74 | yes |
| vqc_boost:vqc_allz_ce_L2-mlp_pca8:recall_malignant | -0.001 | [-0.031, +0.029] | 0.93 | no |
| vqc_boost:vqc_allz_ce_L2-SVM_RBF:balanced_accuracy | -0.022 | [-0.059, +0.015] | 0.17 | no |
| vqc_boost:vqc_allz_ce_L4-linear_pca8:balanced_accuracy | +0.005 | [-0.033, +0.044] | 0.71 | no |
| vqc_boost:vqc_allz_ce_L4-linear_pca8:recall_malignant | +0.029 | [-0.043, +0.101] | 0.33 | no |
| vqc_boost:vqc_allz_ce_L4-mlp_pca8:balanced_accuracy | +0.010 | [-0.032, +0.051] | 0.56 | no |
| vqc_boost:vqc_allz_ce_L4-mlp_pca8:recall_malignant | +0.023 | [-0.053, +0.098] | 0.46 | no |
| vqc_boost:vqc_allz_ce_L4-SVM_RBF:balanced_accuracy | -0.010 | [-0.034, +0.014] | 0.31 | no |
| vqc_boost:vqc_reup2_allz_ce_L2-linear_pca8:balanced_accuracy | +0.005 | [-0.043, +0.054] | 0.77 | no |
| vqc_boost:vqc_reup2_allz_ce_L2-linear_pca8:recall_malignant | +0.029 | [-0.066, +0.124] | 0.45 | no |
| vqc_boost:vqc_reup2_allz_ce_L2-mlp_pca8:balanced_accuracy | +0.010 | [-0.043, +0.062] | 0.64 | no |
| vqc_boost:vqc_reup2_allz_ce_L2-mlp_pca8:recall_malignant | +0.022 | [-0.080, +0.125] | 0.58 | no |
| vqc_boost:vqc_reup2_allz_ce_L2-SVM_RBF:balanced_accuracy | -0.010 | [-0.038, +0.018] | 0.37 | no |
| vqc_boost:vqc_z0_ce_head_L2-linear_pca8:balanced_accuracy | -0.038 | [-0.079, +0.002] | 0.06 | no |
| vqc_boost:vqc_z0_ce_head_L2-linear_pca8:recall_malignant | -0.034 | [-0.087, +0.018] | 0.14 | no |
| vqc_boost:vqc_z0_ce_head_L2-mlp_pca8:balanced_accuracy | -0.034 | [-0.064, -0.004] | 0.033 | no |
| vqc_boost:vqc_z0_ce_head_L2-mlp_pca8:recall_malignant | -0.041 | [-0.078, -0.003] | 0.039 | no |
| vqc_boost:vqc_z0_ce_head_L2-SVM_RBF:balanced_accuracy | -0.054 | [-0.098, -0.009] | 0.028 | no |
| vqc_boost:linear_pca8-SVM_RBF:balanced_accuracy | -0.016 | [-0.064, +0.033] | 0.42 | no |
