from config import PipelineConfig
from engine import DataBuilder, ModelEngine
from cache import MatrixCache
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Initializing Memory-Optimized Aortic Dissection Pipeline...")
    cfg = PipelineConfig()

    # 1. Build Data (Returns paths to temporary files rather than passing DataFrames)
    matrix_cache = MatrixCache(cfg)
    temp_matrix_path = matrix_cache.restore_raw_matrix()
    if temp_matrix_path is None:
        builder = DataBuilder(cfg)
        temp_spine_path = builder.build_spine_and_demographics()
        temp_vitals_path = builder.add_vitals(temp_spine_path)
        temp_matrix_path = builder.add_labs(temp_vitals_path)
        matrix_cache.store_raw_matrix(temp_matrix_path)

    # Add raw feature builders before matrix_cache.store_raw_matrix and include
    # their inputs in MatrixCache._signature before caching them.

    # IF BUILT NEW FEATURES, UNCOMMENT FOLLOWING LINE
    # builder = DataBuilder(cfg)
    # temp_matrix_path = builder.engineer_features(temp_matrix_path)

    # 2. Preprocess and Train (Cleans up the final temporary file automatically)
    engine = ModelEngine(cfg)
    X_train, y_train, X_test, y_test = engine.preprocess(temp_matrix_path)
    engine.train_and_eval(X_train, y_train, X_test, y_test)
 
if __name__ == "__main__":
    main()
 
