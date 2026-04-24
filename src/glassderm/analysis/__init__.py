from .case_studies import build_case_studies
from .correct_features import write_correct_prediction_summary
from .plots import (
    plot_confusion_matrices,
    plot_correct_prediction_features,
    plot_dataset_scale_distribution,
    plot_feature_importance_or_shape_functions,
    plot_metrics_vs_data_size,
    plot_pipeline_comparison_full,
    plot_transparent_explanation_example,
    plot_transparent_lr_vs_tree_tradeoff,
)
from .reports import (
    write_hard_cbm_report,
    write_readme_results,
    write_transparent_lr_readouts,
    write_transparent_tree_readouts,
)

__all__ = [
    "build_case_studies",
    "plot_confusion_matrices",
    "plot_correct_prediction_features",
    "plot_dataset_scale_distribution",
    "plot_feature_importance_or_shape_functions",
    "plot_metrics_vs_data_size",
    "plot_pipeline_comparison_full",
    "plot_transparent_explanation_example",
    "plot_transparent_lr_vs_tree_tradeoff",
    "write_correct_prediction_summary",
    "write_hard_cbm_report",
    "write_readme_results",
    "write_transparent_lr_readouts",
    "write_transparent_tree_readouts",
]
