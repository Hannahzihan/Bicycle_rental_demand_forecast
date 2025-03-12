# Project for D100 and D400
This is a project for D100 and D400 courses.

Installation Guidance

# create a virtual environment and activate it
python -m venv venv
.\venv\Scripts\activate
# install all the dependencies
pip install -r requirements.txt
# install the project locally
pip install e .

# or

# create the environment
conda env create -f environment.yml
# activate the environment 
conda init powershell
conda activate myenv
# install the project locally
pip install -e .
