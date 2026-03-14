"""
PyTorch implementation of OpenNILM

This package provides PyTorch implementations of NILM (Non-Intrusive Load Monitoring)
models including CNN, GRU, and TCN architectures.
"""

from .models import CNN_NILM, GRU_NILM, TCN_NILM, get_model
from .data_loader import NILMDataset, DataLoaderNILM, SimpleNILMDataLoader
from .trainer import Trainer, train_model, EarlyStopping, ModelCheckpoint, TrainingHistory
from .tester import Tester, SimpleTester, load_model
from .config import (
    MODEL_CONFIGS,
    TRAINING,
    CALLBACKS,
    DATASET_CONFIGS,
    DATASET_SPLITS,
    REFIT_PARAMS,
    PLEGMA_PARAMS,
    get_appliance_params,
    get_model_config,
    get_dataset_config,
    get_dataset_split
)
from .utils import (
    set_seeds,
    create_experiment_directories,
    get_device,
    count_parameters,
    print_model_summary,
    save_checkpoint,
    load_checkpoint
)


from .evaluator import (
    run_predictions,
    compute_status,
    compute_metrics,
    evaluate_model,
)
from .pipeline import (
    build_nilm_model,
    get_data_loader,
    run_training,
    run_evaluation,
    save_pipeline_results,
)

# quantizer is imported lazily (TF may not be installed in every environment)
# Users who need it should import directly:
#   from src_pytorch.quantizer import rebuild_pruned_tcn, build_tcn_keras, ...

__version__ = '1.0.0'
__all__ = [
    # Models
    'CNN_NILM',
    'GRU_NILM',
    'TCN_NILM',
    'get_model',

    # Data
    'NILMDataset',
    'DataLoaderNILM',
    'SimpleNILMDataLoader',

    # Training
    'Trainer',
    'train_model',
    'EarlyStopping',
    'ModelCheckpoint',
    'TrainingHistory',

    # Testing
    'Tester',
    'SimpleTester',
    'load_model',

    # Config
    'MODEL_CONFIGS',
    'TRAINING',
    'CALLBACKS',
    'DATASET_CONFIGS',
    'DATASET_SPLITS',
    'REFIT_PARAMS',
    'PLEGMA_PARAMS',
    'get_appliance_params',
    'get_model_config',
    'get_dataset_config',
    'get_dataset_split',

    # Utils
    'set_seeds',
    'create_experiment_directories',
    'get_device',
    'count_parameters',
    'print_model_summary',
    'save_checkpoint',
    'load_checkpoint',

    # Evaluation
    'run_predictions',
    'compute_status',
    'compute_metrics',
    'evaluate_model',

    # Pipeline
    'build_nilm_model',
    'get_data_loader',
    'run_training',
    'run_evaluation',
    'save_pipeline_results',
    # run_quantization intentionally omitted from __all__ (lazy TF import)
]
