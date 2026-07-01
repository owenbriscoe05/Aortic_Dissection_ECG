from config import PipelineConfig
from engine import DataBuilder, ModelEngine
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Initializing Memory-Optimized Aortic Dissection Pipeline...")
    cfg = PipelineConfig()

    # 1. Build Data (Returns paths to temporary files rather than passing DataFrames)
    builder = DataBuilder(cfg)
    temp_spine_path = builder.build_spine_and_demographics()
    temp_vitals_path = builder.add_vitals(temp_spine_path)
    temp_matrix_path = builder.add_labs(temp_vitals_path)

    # IF USING ADDITIONAL INHERENT FEATURES, UNCOMMENT FOLLOWING LINE (and implement function in engine.py)
    # temp_matrix_path = builder.add_medications(temp_vitals_path)

    # IF BUILT NEW FEATURES, UNCOMMENT FOLLOWING LINE
    # temp_matrix_path = builder.engineer_features(temp_matrix_path)

    # 2. Preprocess and Train (Cleans up the final temporary file automatically)
    engine = ModelEngine(cfg)
    X, y = engine.preprocess(temp_matrix_path)
    engine.train_and_eval(X, y)
 
if __name__ == "__main__":
    main()
 