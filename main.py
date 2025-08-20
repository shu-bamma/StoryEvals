from data_processor import DataProcessor
from utils import save_results_to_json


def main():
    print("Starting StoryEvals evaluation pipeline...")
    
    # Step 1: Data Processing (Supporting function for evaluations)
    print("\n=== Step 1: Data Processing ===")
    processor = DataProcessor()
    all_results = processor.process_all_projects()
    save_results_to_json(all_results)
    
    # Step 2: Run Evaluations
    print("\n=== Step 2: Running Evaluations ===")

    
    # Step 3: Save Results
    print("\n=== Step 3: Saving Results ===")

    
    # Step 4: Display Evaluation Summary
    print("\n=== Evaluation Summary ===")

    
    print("\nEvaluation pipeline complete!")


if __name__ == "__main__":
    main()
