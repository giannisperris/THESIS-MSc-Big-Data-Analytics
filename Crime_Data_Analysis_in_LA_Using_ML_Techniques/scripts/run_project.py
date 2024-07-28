from data_loader import load
from data_cleaning import clean
from data_preprocessing import preprocess
from eda_analysis import eda
from ml_analysis import ml_analysis

# XXX: Some paths are relative to current and looking for Crime_Data_Analysis_in_LA_Using_ML_Techniques
# Hence I need to run the code from the root directory: 
# python Crime_Data_Analysis_in_LA_Using_ML_Techniques\scripts\run_project.py

def main():
    # 
    print('Running project...')
    print("loading")
    load()
    print("cleaning")	
    clean()
    print("preprocessing")
    preprocess()
    print("eda")
    eda()
    print("ml analysis")
    ml_analysis()

if __name__ == '__main__':
    main()