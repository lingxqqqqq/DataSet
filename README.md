Running Steps

1. Prepare Input Files
   Ensure you have prepared the target protein's PDB file (e.g. `example.pdb`). This file will serve as the basis for the input data.

2. Data Preprocessing
   Use `data_utils.py` to extract structural information, SASA (Solvent Accessible Surface Area), and charge information from the PDB file, and save it as a CSV file.

3. Modify Configuration
   Before running the prediction script, ensure the following settings are modified:
   Model Path: EMOCPD/models/model_save/best_6071model.pth.tar
   Input Data Path: Specify the path to the preprocessed CSV file.
   Output Path: Define where the prediction results should be stored.

   Open the `predict.py` file and modify the following code segment accordingly:
   MODEL_PATH = "EMOCPD/models/model_save/best_6071model.pth.tar" 
   INPUT_CSV = "path/to/your/input.csv"                          
   OUTPUT_DIR = "path/to/your/output.csv"                          

5. Run Prediction
   Run the `predict.py` script to generate the final prediction results.

   python predict.py


Output
   The prediction results will be saved in the specified output directory, typically as a CSV file, containing the following information:
   Predicted probabilities of amino acids at each position of the protein.

